"""Code-review node — LLM audit of generated code against the DoD for ONE story."""

from __future__ import annotations

import re

from agile_agent_factory.config import MAX_REVIEW_RETRIES
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_doc, make_adf_heading
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded
from agile_agent_factory.tools.logger import log
from agile_agent_factory.state import PipelineState
from agile_agent_factory.nodes.helpers import _active_story, _safe_transition, _notify_quota


def review_node(state: PipelineState) -> dict:
    """Reviewer agent: LLM audit of generated code against the DoD for ONE story."""
    from langgraph.types import interrupt
    from agile_agent_factory.agents.reviewer_agent import review_patch

    jira = JiraClient()
    sk, story = _active_story(state)
    review_retries = story.get("review_retries", state.get("review_retries", 0))
    log(f"Review: auditing code for {sk}.")

    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, "In Code Review")

    story_criteria = state.get("gherkin_criteria", {}).get(sk, [])

    # Derive write scope from test_contract so the reviewer only verdicts on owned files
    tc = story.get("test_contract", {})
    write_scope: list[str] = []
    if tc:
        if tc.get("test_file"):
            write_scope.append(tc["test_file"])
        for imp in (tc.get("target_imports") or []):
            m = re.match(r"from (app(?:\.\w+)+) import", imp)
            if m:
                path_str = m.group(1).replace(".", "/") + ".py"
                if path_str not in write_scope:
                    write_scope.append(path_str)

    try:
        result = review_patch(jira, [sk], story_criteria=story_criteria or None, story_key=sk, write_scope=write_scope or None)
    except LLMQuotaExceeded as e:
        _notify_quota(jira, sk, e)
        interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": sk})
        result = review_patch(jira, [sk], story_criteria=story_criteria or None, story_key=sk, write_scope=write_scope or None)

    approved = result.get("approved", False)
    reason = result.get("reason", "")

    if not approved and review_retries < MAX_REVIEW_RETRIES:
        cycle = review_retries + 1
        log(f"Review rejected for {sk} (retry {cycle}/{MAX_REVIEW_RETRIES}). Keeping in code_review for rework.")
        kickback_doc = {
            "version": 1,
            "type": "doc",
            "content": [
                make_adf_heading(
                    f"Code review rejected — keeping in code_review for rework (cycle {cycle} of {MAX_REVIEW_RETRIES})"
                ),
                {"type": "paragraph", "content": [{"type": "text", "text": f"Reason: {reason}"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": "The downstream agent will attempt to fix the issues above."}]},
            ],
        }
        jira.add_comment_adf(sk, kickback_doc)
        return {
            "review_approved": False,
            "review_retries": cycle,
            "stories": {sk: {
                "column": "code_review",
                "review_status": "rework_needed",
                "review_retries": cycle,
                "review_rejection_reason": reason,
            }},
        }

    if not approved and review_retries >= MAX_REVIEW_RETRIES:
        log(f"Max review retries ({MAX_REVIEW_RETRIES}) exhausted for {sk}. Triggering HITL.")
        summary = (
            f"Code review HITL: review rejected after {review_retries + 1} attempts.\n\n"
            f"Final rejection reason:\n\n{reason}"
        )
        jira.add_comment_adf(sk, make_adf_doc(summary))
        jira.set_flag(sk)
        interrupt({"type": "intervention", "blocking_key": sk})
        try:
            jira.clear_flag(sk)
        except Exception:
            pass
        # After human resume: fresh review cycle with full retry budget
        return {
            "review_approved": False,
            "review_retries": 0,
            "stories": {sk: {"review_retries": 0, "review_status": "pending_review"}},
        }

    log(f"Code review: APPROVED for {sk}.")

    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, "To QA")

    return {
        "review_approved": True,
        "stories": {sk: {"column": "done", "review_retries": review_retries, "review_status": None}},
    }
