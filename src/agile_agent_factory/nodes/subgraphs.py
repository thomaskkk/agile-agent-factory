"""LangGraph subgraphs for the agile-agent-factory pipeline.

Phase 5: The pytest retry loop is implemented as a proper LangGraph subgraph
so that each retry attempt is individually checkpointed. This means a crash
mid-retry resumes at the start of the next attempt, not from scratch.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agile_agent_factory.config import MAX_CORRECTION_FAILURES, MAX_RETRIES_DEV, PRODUCT_ROOT
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_bullet_list, make_adf_doc, make_adf_heading
from agile_agent_factory.tools.logger import log
from agile_agent_factory.nodes.pipeline import (
    _correct_code,
    _extract_error_summary,
    _to_legacy_state,
)
from agile_agent_factory.tools.pytest_runner import run_pytest
from agile_agent_factory.state import PipelineState


# ---------------------------------------------------------------------------
# Retry subgraph state
# ---------------------------------------------------------------------------

class RetryState(TypedDict, total=False):
    """State threaded through the pytest retry subgraph."""

    # Inputs from the parent graph (set by test_node before invoking subgraph)
    story_key: str
    blueprint: str
    extra_deps: list

    # Mutable retry counters
    retries: int
    correction_failures: int
    last_test_output: str  # "" means "run pytest fresh"; non-empty means reuse traceback
    passed: bool  # True when all tests green


# ---------------------------------------------------------------------------
# Subgraph node functions
# ---------------------------------------------------------------------------

def _run_pytest_node(state: RetryState) -> dict:
    """Run pytest (or reuse last_test_output if correction produced no files)."""
    from dependencies import resolve_dependencies

    last = state.get("last_test_output", "")
    if last:
        # Correction produced no new files — reuse the traceback rather than running again
        log("Reusing last traceback (correction produced no files).")
        return {"last_test_output": last}

    legacy = {
        "dependencies": state.get("extra_deps", []),
        "story_keys": [],
        "epic_keys": [],
        "subtasks": {},
        "review_retries": 0,
        "has_ui": False,
        "gherkin_criteria": {},
        "ux_spec": {},
        "architecture": {},
        "hitl_feedback": "",
        "blocking_issue_key": None,
        "status": "READY",
        "current_phase": None,
    }
    exit_code, output = run_pytest(state.get("extra_deps", []))

    if exit_code == 0:
        return {"passed": True, "last_test_output": ""}

    if exit_code in (4, 5):
        output += (
            f"\n\nNote: pytest exit code {exit_code} — tests could not be collected. "
            "This is an import or syntax error. "
            "Fix the import/definition mismatch so collection succeeds."
        )

    return {"passed": False, "last_test_output": output}


def _check_budget_node(state: RetryState) -> dict:
    """No state changes — routing only (see _route_after_pytest)."""
    return {}


def _correct_code_node(state: RetryState) -> dict:
    """Ask the LLM to fix the failing tests."""
    blueprint = state.get("blueprint", "")
    traceback = state.get("last_test_output", "")
    corrected = _correct_code(blueprint, traceback)
    if corrected:
        return {
            "retries": state.get("retries", 0) + 1,
            "correction_failures": 0,
            "last_test_output": "",  # run pytest fresh next iteration
        }
    # Correction produced no files
    return {
        "correction_failures": state.get("correction_failures", 0) + 1,
        "last_test_output": traceback,  # reuse traceback (skip re-running pytest)
    }


def _hitl_pause_node(state: RetryState) -> dict:
    """Interrupt for human intervention when budgets are exhausted."""
    from langgraph.types import interrupt

    jira = JiraClient()
    sk = state.get("story_key", "")
    output = state.get("last_test_output", "")
    retries = state.get("retries", 0)
    correction_failures = state.get("correction_failures", 0)

    if retries >= MAX_RETRIES_DEV:
        summary = (
            f"Downstream HITL: pytest failed after {retries + 1} attempts.\n\n"
            f"Final traceback:\n\n{output[-3000:]}"
        )
    else:
        summary = (
            f"HITL: correction loop failed {correction_failures + 1} times without producing valid files.\n\n"
            f"Last traceback:\n\n{output[-3000:]}"
        )

    if sk:
        jira.add_comment_adf(sk, make_adf_doc(summary))
        jira.set_flag(sk)

    interrupt({"type": "intervention", "blocking_key": sk})

    if sk:
        try:
            jira.clear_flag(sk)
        except Exception:
            pass

    # On resume: reset counters and run pytest fresh
    return {"retries": 0, "correction_failures": 0, "last_test_output": ""}


def _post_correction_comment_node(state: RetryState) -> dict:
    """Post a Jira comment summarising the correction attempt."""
    jira = JiraClient()
    sk = state.get("story_key", "")
    retries = state.get("retries", 0)
    traceback = state.get("last_test_output", "")

    if not sk:
        return {}

    # Find the files that were just changed (we don't have the list here, so use generic comment)
    error_summary = _extract_error_summary(traceback)
    comment_nodes = [
        make_adf_heading(f"Correction applied — attempt {retries} of {MAX_RETRIES_DEV + 1}"),
        {"type": "paragraph", "content": [{"type": "text", "text": f"Error addressed:\n{error_summary}"}]},
    ]
    jira.add_comment_adf(sk, {"version": 1, "type": "doc", "content": comment_nodes})
    return {}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_pytest(state: RetryState) -> str:
    if state.get("passed"):
        return "done"
    retries = state.get("retries", 0)
    if retries >= MAX_RETRIES_DEV:
        return "hitl"
    return "correct"


def _route_after_correct(state: RetryState) -> str:
    correction_failures = state.get("correction_failures", 0)
    if correction_failures > MAX_CORRECTION_FAILURES:
        return "hitl"
    return "comment"


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------

def build_retry_subgraph(checkpointer=None):
    """Build and return the pytest retry subgraph.

    The subgraph is intended to be embedded in the main graph as the "test" node.
    When used standalone (e.g., in tests), pass an InMemorySaver checkpointer.
    """
    g = StateGraph(RetryState)

    g.add_node("run_pytest", _run_pytest_node)
    g.add_node("check_budget", _check_budget_node)
    g.add_node("correct_code", _correct_code_node)
    g.add_node("hitl_pause", _hitl_pause_node)
    g.add_node("post_comment", _post_correction_comment_node)

    g.add_edge(START, "run_pytest")
    g.add_edge("run_pytest", "check_budget")
    g.add_conditional_edges(
        "check_budget",
        _route_after_pytest,
        {"done": END, "hitl": "hitl_pause", "correct": "correct_code"},
    )
    g.add_conditional_edges(
        "correct_code",
        _route_after_correct,
        {"hitl": "hitl_pause", "comment": "post_comment"},
    )
    # After posting comment, loop back to run pytest with fresh state
    g.add_edge("post_comment", "run_pytest")
    # After HITL, loop back to start (human may have fixed the code)
    g.add_edge("hitl_pause", "run_pytest")

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()
