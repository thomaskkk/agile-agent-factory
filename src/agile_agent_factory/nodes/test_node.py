"""Testing node — runs pytest with an LLM-driven correction loop for ONE story."""

from __future__ import annotations

from agile_agent_factory.config import (
    MAX_CORRECTION_FAILURES, MAX_RETRIES_DEV, PRODUCT_ROOT, TEST_MODEL,
)
from agile_agent_factory.tools.jira_client import (
    JiraClient, make_adf_bullet_list, make_adf_doc, make_adf_heading,
)
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.dependencies import resolve_dependencies
from agile_agent_factory.tools.pytest_runner import run_pytest
from agile_agent_factory.state import PipelineState
from agile_agent_factory.nodes.helpers import _active_story, _safe_transition, _to_legacy_state
from agile_agent_factory.nodes.dev_node import _load_dev_context, _correct_code, _extract_error_summary


def test_node(state: PipelineState) -> dict:
    """Run pytest with a correction loop for ONE story.

    On HITL resume, re-runs pytest from scratch (retries reset to 0).
    """
    from langgraph.types import interrupt

    jira = JiraClient()
    sk, story = _active_story(state)
    log(f"Test: running pytest for {sk}.")

    blueprint = _load_dev_context(sk)
    deps = resolve_dependencies(_to_legacy_state(state, sk), PRODUCT_ROOT)

    retries = 0
    correction_failures = 0
    last_output: str | None = None

    while True:
        if last_output is None:
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
            _safe_transition(jira, sk, "To Code Review")
            for ek in state.get("epic_keys", []):
                _safe_transition(jira, ek, "To Code Review")
            return {"stories": {sk: {"column": "code_review"}}}

        if exit_code in (4, 5) and last_output is None:
            log(f"pytest exit code {exit_code}: collection error.")
            output += (
                f"\n\nNote: pytest exit code {exit_code} — tests could not be collected. "
                "This is an import or syntax error in the test or app modules shown above. "
                "Fix the import/definition mismatch (or generate the missing module/test) so collection succeeds."
            )

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
            retries = 0
            correction_failures = 0
            last_output = None
            continue

        log("Requesting LLM-driven correction.")
        corrected = _correct_code(blueprint, output, model=TEST_MODEL or None)

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
                retries = 0
                correction_failures = 0
                last_output = None
                continue
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
