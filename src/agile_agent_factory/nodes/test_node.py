"""Testing node — runs pytest with an LLM-driven correction loop for ONE story."""

from __future__ import annotations

import re

from agile_agent_factory.config import (
    MAX_CORRECTION_FAILURES, MAX_RETRIES_DEV, MAX_STRATEGY_RETRIES, PRODUCT_ROOT, TEST_MODEL,
)
from agile_agent_factory.tools.jira_client import (
    JiraClient, make_adf_bullet_list, make_adf_doc, make_adf_heading,
)
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.dependencies import resolve_dependencies
from agile_agent_factory.tools.pytest_runner import run_pytest
from agile_agent_factory.tools.workflow import WorkflowState
from agile_agent_factory.state import PipelineState
from agile_agent_factory.nodes.helpers import _active_story, _safe_transition, _to_legacy_state, raise_quota_interrupt
from agile_agent_factory.nodes.dev_node import _load_dev_context, _correct_code, _extract_error_summary
from agile_agent_factory.nodes.failure_recovery import (
    classify_failure,
    _scaffold_missing_module,
    _scaffold_fixture,
    _scaffold_missing_test_function,
)


def test_node(state: PipelineState) -> dict:
    """Run pytest with a correction loop for ONE story.

    On HITL resume, re-runs pytest from scratch (retries reset to 0).

    Retry budget separation (Milestone 8):
      retries             — genuine LLM-reasoning failures; escalates to intervention HITL
      correction_failures — _correct_code producing zero usable files; its own HITL trigger
      strategy_retries    — mechanical (dep re-resolve, stub scaffold, truncation retry);
                            these do NOT consume retries or correction_failures
    """
    from langgraph.types import interrupt

    jira = JiraClient()
    sk, story = _active_story(state)
    log(f"Test: running pytest for {sk}.")

    blueprint = _load_dev_context(sk)
    deps = resolve_dependencies(_to_legacy_state(state, sk), PRODUCT_ROOT)

    # Derive story-scoped test target from test_contract (Milestone 2)
    tc = story.get("test_contract", {})
    story_test_file = tc.get("test_file") if tc else None
    write_scope: list[str] = []
    if tc:
        if story_test_file:
            write_scope.append(story_test_file)
        for imp in (tc.get("target_imports") or []):
            if isinstance(imp, str) and imp.strip():
                m = re.match(r"from (app(?:\.\w+)+) import", imp)
                if m:
                    path_str = m.group(1).replace(".", "/") + ".py"
                    if path_str not in write_scope:
                        write_scope.append(path_str)

    retries = 0
    correction_failures = 0
    strategy_retries = 0
    last_output: str | None = None

    while True:
        # --- Run pytest (targeted first, then full suite) ---
        if last_output is None:
            if story_test_file:
                # Stage 1: targeted run (story-scoped, fast signal)
                target_path = str(PRODUCT_ROOT / story_test_file)
                t_exit, t_output = run_pytest(deps, test_targets=[target_path])
                if t_exit != 0:
                    # Targeted failed — feed to correction loop
                    exit_code, output = t_exit, t_output
                else:
                    # Stage 2: full suite to catch regressions
                    exit_code, output = run_pytest(deps)
                    if exit_code != 0:
                        # Targeted green, full suite red — classify the regression
                        failure_class = classify_failure(exit_code, output)
                        tb_files = _failing_files_in_output(output)
                        in_scope = any(f in write_scope for f in tb_files)
                        if in_scope:
                            # Our write-scope still has issues — keep iterating correction
                            log(f"Full suite regression in write-scope for {sk} — continuing correction.")
                        else:
                            # Cross-story regression — quarantine and advance
                            log(f"Cross-story regression detected for {sk}; quarantining.")
                            blocker_ids = [f for f in tb_files if f not in write_scope] or ["unclassified"]
                            try:
                                jira.add_comment_adf(
                                    sk,
                                    make_adf_doc(
                                        f"Story targeted tests passed. Full suite has cross-story regression "
                                        f"in: {', '.join(blocker_ids)}. Story advancing (quarantined)."
                                    ),
                                )
                            except Exception:
                                pass
                            _safe_transition(jira, sk, WorkflowState.TO_CODE_REVIEW)
                            for ek in state.get("epic_keys", []):
                                _safe_transition(jira, ek, WorkflowState.TO_CODE_REVIEW)
                            return {
                                "stories": {sk: {
                                    "column": "code_review",
                                    "regression_blockers": blocker_ids,
                                }}
                            }
            else:
                exit_code, output = run_pytest(deps)
        else:
            exit_code, output = 1, last_output

        if exit_code == 0:
            log(f"All tests passed for {sk}. Moving to code review.")
            if retries > 0:
                jira.add_comment_adf(
                    sk,
                    make_adf_doc(
                        f"Tests passing after {retries} correction attempt(s). Proceeding to code review."
                    ),
                )
            _safe_transition(jira, sk, WorkflowState.TO_CODE_REVIEW)
            for ek in state.get("epic_keys", []):
                _safe_transition(jira, ek, WorkflowState.TO_CODE_REVIEW)
            return {"stories": {sk: {"column": "code_review", "regression_blockers": []}}}

        if exit_code in (4, 5) and last_output is None:
            log(f"pytest exit code {exit_code}: collection error.")
            output += (
                f"\n\nNote: pytest exit code {exit_code} — tests could not be collected. "
                "This is an import or syntax error in the test or app modules shown above. "
                "Fix the import/definition mismatch (or generate the missing module/test) so collection succeeds."
            )

        # --- Mechanical failure recovery (before consuming LLM reasoning budget) ---
        failure_class = classify_failure(exit_code, output)
        log(f"Failure class for {sk}: {failure_class}.")

        if failure_class == "missing_dependency":
            if strategy_retries < MAX_STRATEGY_RETRIES:
                log(f"Missing dependency detected — re-resolving deps for {sk}.")
                deps = resolve_dependencies(_to_legacy_state(state, sk), PRODUCT_ROOT)
                strategy_retries += 1
                last_output = None
                continue  # re-run pytest with updated deps; does NOT consume retries

        elif failure_class == "missing_module":
            if strategy_retries < MAX_STRATEGY_RETRIES:
                scaffolded = _scaffold_missing_module(output)
                if scaffolded:
                    log(f"Scaffolded missing module(s) for {sk}: {scaffolded}.")
                    strategy_retries += 1
                    last_output = None
                    continue  # re-run pytest; does NOT consume retries
                # No "No module named" pattern found → fall through to LLM

        elif failure_class == "fixture_not_found":
            if strategy_retries < MAX_STRATEGY_RETRIES:
                scaffolded = _scaffold_fixture(output)
                if scaffolded:
                    log(f"Scaffolded fixture stub(s) for {sk}: {scaffolded}.")
                    strategy_retries += 1
                    last_output = None
                    continue  # re-run pytest; does NOT consume retries
                # Nothing scaffolded → fall through to LLM

        elif failure_class == "missing_test_function":
            if strategy_retries < MAX_STRATEGY_RETRIES:
                scaffolded = _scaffold_missing_test_function(output, story_test_file)
                if scaffolded:
                    log(f"Scaffolded test function stub(s) for {sk}: {scaffolded}.")
                    strategy_retries += 1
                    last_output = None
                    continue  # re-run pytest; does NOT consume retries
                # Nothing scaffolded → fall through to LLM

        elif failure_class == "namespace_collision":
            if strategy_retries < MAX_STRATEGY_RETRIES:
                log(f"Namespace collision detected for {sk} — re-resolving deps and retrying.")
                deps = resolve_dependencies(_to_legacy_state(state, sk), PRODUCT_ROOT)
                strategy_retries += 1
                last_output = None
                continue  # re-run pytest with fresh deps; does NOT consume retries

        elif failure_class in ("syntax_error", "collection_error_generic", "bad_import_signature"):
            if strategy_retries < MAX_STRATEGY_RETRIES:
                hint = _failure_hint(failure_class)
                log(f"Deterministic failure ({failure_class}) for {sk} — sending targeted hint to LLM.")
                correction_scope = _augment_scope_for_conftest(write_scope, failure_class, output)
                try:
                    correction_status, corrected = _correct_code(
                        blueprint, hint + output, model=TEST_MODEL or None, write_scope=correction_scope or None,
                        test_contract=story.get("test_contract"),
                        gherkin_criteria=story.get("gherkin_criteria"),
                    )
                except LLMQuotaExceeded as e:
                    patch = raise_quota_interrupt(jira, sk, e, state=state)
                    if patch:
                        return patch
                strategy_retries += 1

                if correction_status == "truncated":
                    log(f"Targeted hint correction truncated for {sk} — will retry.")
                    last_output = output
                    continue

                if corrected:
                    log(f"Targeted hint correction applied for {sk}: {corrected}.")
                    last_output = None
                    continue
                # Nothing produced → fall through to ordinary LLM loop below

        attempt = retries + 1
        log(f"pytest failed (attempt {attempt}/{MAX_RETRIES_DEV + 1}) for {sk}.")

        if retries >= MAX_RETRIES_DEV:
            log(f"Max retries exceeded for {sk}. Triggering HITL.")
            summary = (
                f"Downstream HITL: pytest failed after {attempt} attempts.\n\n"
                f"Final traceback:\n\n{output[-3000:]}"
            )
            jira.add_comment_adf(sk, make_adf_doc(summary))
            jira.set_flag(sk)
            interrupt({"type": "intervention", "blocking_key": sk})
            try:
                jira.clear_flag(sk)
            except Exception:
                pass
            # On resume: node will be re-entered from scratch; return hitl_type so the
            # state captures why this story is blocked.
            return {"stories": {sk: {"hitl_type": "intervention"}}}

        log("Requesting LLM-driven correction.")
        correction_scope = _augment_scope_for_conftest(write_scope, failure_class, output)
        try:
            correction_status, corrected = _correct_code(
                blueprint, output, model=TEST_MODEL or None, write_scope=correction_scope or None,
                test_contract=story.get("test_contract"),
                gherkin_criteria=story.get("gherkin_criteria"),
            )
        except LLMQuotaExceeded as e:
            patch = raise_quota_interrupt(jira, sk, e, state=state)
            if patch:
                return patch

        # Truncation: strategy retry budget (not the LLM reasoning budget)
        if correction_status == "truncated":
            if strategy_retries < MAX_STRATEGY_RETRIES:
                log(f"Correction truncated for {sk} — using strategy retry budget.")
                strategy_retries += 1
                last_output = output
                continue
            # Exhausted strategy budget — treat as correction failure
            log(f"Truncation strategy budget exhausted for {sk} — treating as correction failure.")
            correction_status = "empty"
            corrected = []

        if not corrected:
            if correction_failures >= MAX_CORRECTION_FAILURES:
                log(f"Correction failed {correction_failures + 1}x for {sk} — escalating to HITL.")
                summary = (
                    f"HITL: correction loop failed {correction_failures + 1} times without producing valid files.\n\n"
                    f"Last traceback:\n\n{output[-3000:]}"
                )
                jira.add_comment_adf(sk, make_adf_doc(summary))
                jira.set_flag(sk)
                interrupt({"type": "intervention", "blocking_key": sk})
                try:
                    jira.clear_flag(sk)
                except Exception:
                    pass
                # On resume: node will be re-entered from scratch; return hitl_type so the
                # state captures why this story is blocked.
                return {"stories": {sk: {"hitl_type": "intervention"}}}
            correction_failures += 1
            log(f"No correction produced (failure {correction_failures}/{MAX_CORRECTION_FAILURES + 1}) — retrying.")
            last_output = output
            continue

        if sk:
            error_summary = _extract_error_summary(output)
            comment_nodes = [
                make_adf_heading(f"Correction applied — attempt {attempt} of {MAX_RETRIES_DEV + 1}"),
                {"type": "paragraph", "content": [{"type": "text", "text": f"Error addressed:\n{error_summary}"}]},
                make_adf_heading("Files updated", level=4),
                make_adf_bullet_list(corrected),
            ]
            jira.add_comment_adf(sk, {"version": 1, "type": "doc", "content": comment_nodes})

        retries += 1
        correction_failures = 0
        last_output = None


def _failing_files_in_output(output: str) -> list[str]:
    """Extract relative file paths (app/*, tests/*) from a pytest failure output."""
    import re
    pattern = re.compile(r"\b((?:app|tests)/[\w/]+\.py)\b")
    seen = dict.fromkeys(pattern.findall(output))
    return list(seen)


def _failure_hint(failure_class: str) -> str:
    """Return a short targeted hint string to prepend to the LLM correction prompt."""
    hints = {
        "syntax_error": (
            "TARGETED FIX REQUIRED: The pytest output below contains a SyntaxError. "
            "Locate the exact file and line number, fix only the syntax error, "
            "and do not alter any other logic.\n\n"
        ),
        "bad_import_signature": (
            "TARGETED FIX REQUIRED: The pytest output below shows "
            "'ImportError: cannot import name'. "
            "Add or expose the missing name in the source module "
            "(stub it if necessary) so the import succeeds. "
            "Do not rename or remove existing names.\n\n"
        ),
        "collection_error_generic": (
            "TARGETED FIX REQUIRED: pytest could not collect tests (exit code 4 or 5). "
            "The output below describes the collection error. "
            "Fix the import, syntax, or definition issue so pytest can collect the test file.\n\n"
        ),
    }
    return hints.get(failure_class, "")


def _augment_scope_for_conftest(
    write_scope: list[str], failure_class: str, output: str
) -> list[str]:
    """Return write_scope extended with tests/conftest.py when the failure involves conftest.

    conftest.py is only added when the failure explicitly implicates it — either the
    failure class is fixture_not_found, or the pytest output mentions conftest.py.
    This prevents unrelated stories from accidentally overwriting shared fixtures.
    """
    conftest = "tests/conftest.py"
    if conftest in write_scope:
        return write_scope
    if failure_class == "fixture_not_found" or conftest in output:
        return list(write_scope) + [conftest]
    return write_scope
