"""Shared helpers for the LangGraph node modules.

These small utilities are used across the lifecycle nodes (pipeline.py) and the
dev/test/review node modules. They contain no node entrypoints themselves.
"""

from __future__ import annotations

from langgraph.types import interrupt

from agile_agent_factory.tools.jira_client import JiraClient
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.workflow import WorkflowState
from agile_agent_factory.state import PipelineState


def _story_keys(state: PipelineState) -> list[str]:
    return list(state.get("stories", {}).keys())


def _active_story(state: PipelineState) -> tuple[str, dict]:
    """Return (story_key, story_dict) for the active story in this node invocation."""
    sk = state.get("active_story_key") or _story_keys(state)[0]
    story = state.get("stories", {}).get(sk, {})
    return sk, story


def _safe_transition(jira: JiraClient, key: str, target: WorkflowState) -> None:
    try:
        jira.transition_to(key, target)
    except ValueError as e:
        log(str(e))


def _to_legacy_state(state: PipelineState, story_key: str | None = None) -> dict:
    """Build a legacy flat-state dict compatible with existing agent interfaces."""
    stories = state.get("stories", {})
    story: dict = {}
    if story_key:
        story = stories.get(story_key, {})

    return {
        "status": "READY",
        "current_phase": None,
        "blocking_issue_key": None,
        "hitl_feedback": story.get("hitl_feedback", ""),
        "epic_keys": state.get("epic_keys", []),
        "story_keys": _story_keys(state),
        "subtasks": story.get("subtasks") or state.get("subtasks", {}),
        "review_retries": story.get("review_retries", state.get("review_retries", 0)),
        "has_ui": story.get("has_ui", state.get("has_ui", False)),
        "dependencies": story.get("dependencies") or state.get("dependencies", []),
        "gherkin_criteria": {story_key: story.get("gherkin_criteria", [])} if story_key else state.get("gherkin_criteria", {}),
        "ux_spec": story.get("ux_spec") or state.get("ux_spec", {}),
        "architecture": story.get("architecture") or state.get("architecture", {}),
    }


def _notify_quota(jira: JiraClient, issue_key: str | None, exc: LLMQuotaExceeded) -> None:
    from agile_agent_factory.config import JIRA_HUMAN_ACCOUNT_ID
    from agile_agent_factory.tools.jira_client import make_adf_mention_doc
    provider = getattr(exc, "provider", "unknown")
    if issue_key:
        try:
            jira.set_flag(issue_key)
            jira.add_comment_adf(
                issue_key,
                make_adf_mention_doc(
                    JIRA_HUMAN_ACCOUNT_ID,
                    f"LLM quota exceeded (provider: {provider}). "
                    "Resolve the quota limit and clear this flag to resume the pipeline.",
                ),
            )
        except Exception as notify_err:
            log(f"Failed to post quota notification on {issue_key}: {notify_err}")
    log(f"Quota exceeded ({provider}). Pipeline pausing for human intervention.")


def _story_summary(jira: JiraClient, story_key: str) -> str:
    try:
        issue = jira._request("GET", f"issue/{story_key}?fields=summary")
        return issue.get("fields", {}).get("summary", "") or ""
    except Exception as e:
        log(f"Could not fetch story summary for {story_key}: {e}")
        return ""


def raise_quota_interrupt(jira: JiraClient, blocking_key: str | None, exc: LLMQuotaExceeded) -> None:
    """Notify Jira of the quota block and suspend the graph for HITL resume.

    Encapsulates the repeated `_notify_quota(...)` + `interrupt({"type": "quota", ...})`
    triplet shared by every node that calls an LLM.
    """
    _notify_quota(jira, blocking_key, exc)
    interrupt({
        "type": "quota",
        "provider": getattr(exc, "provider", "unknown"),
        "blocking_key": blocking_key,
    })
