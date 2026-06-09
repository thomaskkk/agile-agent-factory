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
from agile_agent_factory.nodes.helpers import (
    _active_story,
    _changed_product_files,
    derive_story_write_scope,
    _path_in_write_scope,
    _restore_product_files,
    _safe_transition,
    _snapshot_product_files,
    _to_legacy_state,
    _write_scope_violations,
    raise_quota_interrupt,
)
from agile_agent_factory.nodes.dev_node import _load_dev_context, _correct_code, _extract_error_summary
from agile_agent_factory.nodes.failure_recovery import (
    classify_failure,
    _scaffold_missing_module,
    _scaffold_fixture,
    _scaffold_missing_test_function,
)


def _failure_signature(failure_class: str, output: str) -> str:
    files = ",".join(_failing_files_in_output(output)[:5])
    summary = _extract_error_summary(output)
    return f"{failure_class}|{files}|{summary}"


def _repeated_failure_hint(failure_class: str, streak: int) -> str:
    if streak < 2:
        return ""
    return (
        "REPEATED IDENTICAL FAILURE: the previous correction strategy did not resolve this. "
        f"This is recurrence #{streak} for failure class '{failure_class}'. "
        "Change approach. Re-check imports, ownership boundaries, and exact public names before editing.\n\n"
    )


def _story_write_scope(story: dict) -> list[str]:
    story_key = story.get("story_key", "")
    return derive_story_write_scope(story_key, story)


def _owner_for_file(stories: dict[str, dict], target_path: str) -> str | None:
    for candidate_sk, candidate_story in stories.items():
        scope = _story_write_scope(candidate_story)
        if _path_in_write_scope(target_path, scope):
            return candidate_sk
    return None


def _merge_unique_paths(existing: list[str], incoming: list[str]) -> list[str]:
    return list(dict.fromkeys([*(existing or []), *(incoming or [])]))


def _merge_regression_output(existing: str, incoming: str) -> str:
    existing = (existing or "").strip()
    incoming = (incoming or "").strip()
    if not existing:
        return incoming
    if not incoming or incoming in existing:
        return existing
    return f"{existing}\n\n---\n\n{incoming}"


def _route_cross_story_regression(
    *,
    jira: JiraClient,
    state: PipelineState,
    quarantine_story_key: str,
    blocker_files: list[str],
    output: str,
) -> dict:
    stories = state.get("stories", {})

    story_updates: dict[str, dict] = {
        quarantine_story_key: {
            "column": "code_review",
            "regression_blockers": [],
            "incoming_regression_files": [],
            "incoming_regression_output": "",
        }
    }
    requeued_owners: list[str] = []
    natural_owners: list[str] = []
    unresolved: list[str] = []

    for blocker_file in blocker_files:
        owner_sk = _owner_for_file(stories, blocker_file)
        if not owner_sk or owner_sk == quarantine_story_key:
            unresolved.append(blocker_file)
            continue

        owner_story = stories.get(owner_sk, {})
        owner_column = owner_story.get("column")
        if owner_column in {"done", "code_review"}:
            owner_update = story_updates.setdefault(owner_sk, {
                "column": "testing",
                "review_status": None,
                "review_retries": 0,
                "review_rejection_reason": "",
                "hitl_type": None,
                "failure_streak": 0,
                "last_failure_signature": "",
                "last_failure_class": "",
                "incoming_regression_files": list(owner_story.get("incoming_regression_files", [])),
                "incoming_regression_output": owner_story.get("incoming_regression_output", ""),
            })
            owner_update["incoming_regression_files"] = _merge_unique_paths(
                owner_update.get("incoming_regression_files", []),
                [blocker_file],
            )
            owner_update["incoming_regression_output"] = _merge_regression_output(
                owner_update.get("incoming_regression_output", ""),
                output,
            )
            requeued_owners.append(owner_sk)
            continue

        natural_owners.append(owner_sk)

    unresolved = list(dict.fromkeys(unresolved))
    natural_owners = list(dict.fromkeys(natural_owners))
    requeued_owners = list(dict.fromkeys(requeued_owners))
    story_updates[quarantine_story_key]["regression_blockers"] = unresolved

    if requeued_owners:
        for owner_sk in requeued_owners:
            owner_files = story_updates[owner_sk]["incoming_regression_files"]
            try:
                jira.add_comment_adf(
                    owner_sk,
                    make_adf_doc(
                        "Cross-story regression detected while validating "
                        f"{quarantine_story_key}. Automatically requeued to testing for: "
                        f"{', '.join(owner_files)}."
                    ),
                )
            except Exception:
                pass

    status_parts = []
    if requeued_owners:
        status_parts.append(f"Automatically requeued owner stories: {', '.join(requeued_owners)}.")
    if natural_owners:
        status_parts.append(f"Owner stories already in progress: {', '.join(natural_owners)}.")
    if unresolved:
        status_parts.append(f"No owning story found for: {', '.join(unresolved)}.")

    try:
        jira.add_comment_adf(
            quarantine_story_key,
            make_adf_doc(
                "Story targeted tests passed, but the full suite exposed cross-story regressions in: "
                f"{', '.join(blocker_files)}. "
                + (" ".join(status_parts) if status_parts else "Quarantined for later resolution.")
            ),
        )
    except Exception:
        pass

    _safe_transition(jira, quarantine_story_key, WorkflowState.TO_CODE_REVIEW)
    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, WorkflowState.TO_CODE_REVIEW)
    return {"stories": story_updates}


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
    write_scope = _story_write_scope(story)
    full_suite_first = bool(
        story.get("incoming_regression_files") or story.get("incoming_regression_output")
    )
    if full_suite_first:
        log(
            f"Test: {sk} was requeued for incoming regression "
            f"{story.get('incoming_regression_files', [])}; running full suite first."
        )

    retries = 0
    correction_failures = 0
    strategy_retries = 0
    last_output: str | None = None
    last_failure_signature = story.get("last_failure_signature", "")
    last_failure_class = story.get("last_failure_class", "")
    failure_streak = story.get("failure_streak", 0)

    while True:
        # --- Run pytest (targeted first, then full suite) ---
        if last_output is None:
            if full_suite_first:
                exit_code, output = run_pytest(deps)
                if exit_code != 0 and write_scope:
                    tb_files = _regression_blocker_files(output)
                    in_scope = any(_path_in_write_scope(f, write_scope) for f in tb_files)
                    if tb_files and not in_scope:
                        log(f"Cross-story regression detected for {sk}; re-routing owners.")
                        return _route_cross_story_regression(
                            jira=jira,
                            state=state,
                            quarantine_story_key=sk,
                            blocker_files=tb_files,
                            output=output,
                        )
            elif story_test_file:
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
                        tb_files = _regression_blocker_files(output)
                        in_scope = any(_path_in_write_scope(f, write_scope) for f in tb_files)
                        if in_scope:
                            # Our write-scope still has issues — keep iterating correction
                            log(f"Full suite regression in write-scope for {sk} — continuing correction.")
                        else:
                            # Cross-story regression — route to the owning story when possible.
                            log(f"Cross-story regression detected for {sk}; re-routing owners.")
                            blocker_ids = [f for f in tb_files if not _path_in_write_scope(f, write_scope)] or ["unclassified"]
                            return _route_cross_story_regression(
                                jira=jira,
                                state=state,
                                quarantine_story_key=sk,
                                blocker_files=blocker_ids,
                                output=output,
                            )
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
            return {"stories": {sk: {
                "column": "code_review",
                "regression_blockers": [],
                "incoming_regression_files": [],
                "incoming_regression_output": "",
                "failure_streak": 0,
                "last_failure_signature": "",
                "last_failure_class": "",
                "last_changed_files": [],
            }}}

        if exit_code in (4, 5) and last_output is None:
            log(f"pytest exit code {exit_code}: collection error.")
            output += (
                f"\n\nNote: pytest exit code {exit_code} — tests could not be collected. "
                "This is an import or syntax error in the test or app modules shown above. "
                "Fix the import/definition mismatch (or generate the missing module/test) so collection succeeds."
            )

        # --- Mechanical failure recovery (before consuming LLM reasoning budget) ---
        failure_class = classify_failure(exit_code, output)
        current_signature = _failure_signature(failure_class, output)
        if current_signature == last_failure_signature:
            failure_streak += 1
        else:
            last_failure_signature = current_signature
            failure_streak = 1
        last_failure_class = failure_class
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
                before_snapshot = _snapshot_product_files()
                scaffolded = _scaffold_missing_module(output)
                changed_after = _changed_product_files(before_snapshot)
                violations = _write_scope_violations(changed_after, write_scope or None)
                if violations:
                    _restore_product_files(before_snapshot, changed_after)
                    log(f"Missing-module scaffold exceeded write_scope for {sk}: {violations}")
                    scaffolded = []
                if scaffolded:
                    log(f"Scaffolded missing module(s) for {sk}: {scaffolded}.")
                    strategy_retries += 1
                    last_output = None
                    continue  # re-run pytest; does NOT consume retries
                # No "No module named" pattern found → fall through to LLM

        elif failure_class == "fixture_not_found":
            if strategy_retries < MAX_STRATEGY_RETRIES:
                before_snapshot = _snapshot_product_files()
                scaffolded = _scaffold_fixture(output)
                changed_after = _changed_product_files(before_snapshot)
                violations = _write_scope_violations(changed_after, write_scope or None)
                if violations:
                    _restore_product_files(before_snapshot, changed_after)
                    log(f"Fixture scaffold exceeded write_scope for {sk}: {violations}")
                    scaffolded = []
                if scaffolded:
                    log(f"Scaffolded fixture stub(s) for {sk}: {scaffolded}.")
                    strategy_retries += 1
                    last_output = None
                    continue  # re-run pytest; does NOT consume retries
                # Nothing scaffolded → fall through to LLM

        elif failure_class == "missing_test_function":
            if strategy_retries < MAX_STRATEGY_RETRIES:
                before_snapshot = _snapshot_product_files()
                scaffolded = _scaffold_missing_test_function(output, story_test_file)
                changed_after = _changed_product_files(before_snapshot)
                violations = _write_scope_violations(changed_after, write_scope or None)
                if violations:
                    _restore_product_files(before_snapshot, changed_after)
                    log(f"Test-function scaffold exceeded write_scope for {sk}: {violations}")
                    scaffolded = []
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
                repeated_hint = _repeated_failure_hint(failure_class, failure_streak)
                # Bind defaults so a non-raising/None quota return can't fall through unbound.
                correction_status, corrected = "empty", []
                before_snapshot = _snapshot_product_files()
                try:
                    correction_status, corrected = _correct_code(
                        blueprint,
                        repeated_hint + hint + output,
                        model=TEST_MODEL or None,
                        write_scope=correction_scope or None,
                        test_contract=story.get("test_contract"),
                        gherkin_criteria=story.get("gherkin_criteria"),
                    )
                except LLMQuotaExceeded as e:
                    patch = raise_quota_interrupt(jira, sk, e, state=state)
                    if patch:
                        return patch
                    correction_status, corrected = "empty", []
                changed_after = _changed_product_files(before_snapshot)
                violations = _write_scope_violations(changed_after, correction_scope or None)
                if violations:
                    _restore_product_files(before_snapshot, changed_after)
                    log(f"Targeted hint correction exceeded write_scope for {sk}: {violations}")
                    correction_status, corrected = "empty", []
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
            # On resume: node will be re-entered from scratch; return hitl_type so the
            # state captures why this story is blocked.
            return {"stories": {sk: {
                "hitl_type": "intervention",
                "last_failure_signature": last_failure_signature,
                "last_failure_class": last_failure_class,
                "failure_streak": failure_streak,
            }}}

        log("Requesting LLM-driven correction.")
        correction_scope = _augment_scope_for_conftest(write_scope, failure_class, output)
        if failure_streak >= 2:
            focused_scope = [path for path in _failing_files_in_output(output) if _path_in_write_scope(path, correction_scope)]
            if focused_scope:
                correction_scope = focused_scope
                log(f"Repeated failure for {sk} — narrowing correction scope to {focused_scope}.")
        repeated_hint = _repeated_failure_hint(failure_class, failure_streak)
        # Bind defaults so a non-raising/None quota return can't fall through unbound.
        correction_status, corrected = "empty", []
        before_snapshot = _snapshot_product_files()
        try:
            correction_status, corrected = _correct_code(
                blueprint,
                repeated_hint + output,
                model=TEST_MODEL or None,
                write_scope=correction_scope or None,
                test_contract=story.get("test_contract"),
                gherkin_criteria=story.get("gherkin_criteria"),
            )
        except LLMQuotaExceeded as e:
            patch = raise_quota_interrupt(jira, sk, e, state=state)
            if patch:
                return patch
            correction_status, corrected = "empty", []
        changed_after = _changed_product_files(before_snapshot)
        violations = _write_scope_violations(changed_after, correction_scope or None)
        if violations:
            _restore_product_files(before_snapshot, changed_after)
            log(f"Correction exceeded write_scope for {sk}: {violations}")
            correction_status, corrected = "empty", []

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
                # On resume: node will be re-entered from scratch; return hitl_type so the
                # state captures why this story is blocked.
                return {"stories": {sk: {
                    "hitl_type": "intervention",
                    "last_failure_signature": last_failure_signature,
                    "last_failure_class": last_failure_class,
                    "failure_streak": failure_streak,
                }}}
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


_NODE_ID_RE = re.compile(
    r"^(?:FAILED|ERROR)\s+((?:app|tests)/[^\s:]+\.py)", re.MULTILINE
)


def _failing_files_from_node_ids(output: str) -> list[str]:
    """Recover owning test files from pytest node ids.

    Handles ``FAILED tests/...py::test_...`` and ``ERROR tests/...py`` lines whose
    paths the plain path regex may miss (e.g. hyphens, dotted names). Used as a
    fallback so a full-suite regression with node-id-only output can still be
    attributed to a file instead of the unownable ``"unclassified"`` sentinel.
    """
    seen = dict.fromkeys(_NODE_ID_RE.findall(output))
    return list(seen)


def _regression_blocker_files(output: str) -> list[str]:
    """Prefer failing test node ids over traceback paths for cross-story ownership.

    Full-suite regressions frequently mention shared implementation files in the
    traceback (for example ``app/routes.py``), even when the actual failing test
    belongs to a different story. Node ids point at the failing test files, which
    are the most reliable ownership signal for deciding whether to reroute a
    regression instead of holding the current story in correction/HITL.
    """
    return _failing_files_from_node_ids(output) or _failing_files_in_output(output)


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
    if _path_in_write_scope(conftest, write_scope):
        return write_scope
    if failure_class == "fixture_not_found" or conftest in output:
        return list(write_scope) + [conftest]
    return write_scope
