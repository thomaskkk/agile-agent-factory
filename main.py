"""Agile-Agent-Factory entry point — LangGraph edition.

The pipeline is a StateGraph persisted via SqliteSaver (pipeline_checkpoint.db).
State is never written to state_manager.json from here; the checkpointer handles
crash recovery and HITL resume transparently.

Usage:
    uv run main.py               # start or resume pipeline
    uv run main.py --reset-state # delete checkpoint DB and exit
"""

from __future__ import annotations

import argparse
import sys
import time

from agile_agent_factory.config import DRY_RUN, CHECKPOINT_DB
from agile_agent_factory.graph import build_graph
from agile_agent_factory.tools.logger import log


THREAD_ID = "pipeline"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agile-Agent-Factory: Autonomous Software Development Pipeline"
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete checkpoint database and exit (does not touch ../app/ or ../tests/)",
    )
    args = parser.parse_args()

    if args.reset_state:
        if CHECKPOINT_DB.exists():
            CHECKPOINT_DB.unlink()
            log("Checkpoint database deleted. State reset.")
        else:
            log("No checkpoint database found. Nothing to reset.")
        return

    log("Starting Agile-Agent-Factory.")
    if DRY_RUN:
        log("DRY_RUN mode enabled — no production Jira writes.")

    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    snapshot = graph.get_state(config)

    # Detect an active interrupt (pipeline paused for human input)
    if _is_interrupted(snapshot):
        log("Pipeline paused. Checking for human response.")
        _handle_resume(graph, config, snapshot)
        return

    # Detect a completed graph
    if _is_complete(snapshot):
        log("Pipeline already complete. Run --reset-state to start fresh.")
        return

    # Check for a pending autonomous quota backoff before invoking the graph
    _handle_quota_backoff(graph, config, snapshot)

    # Fresh start or crash recovery — invoke the graph
    log("Reading business idea from ../business_idea.md.")
    graph.invoke({}, config)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _handle_quota_backoff(graph, config: dict, snapshot) -> None:
    """If the last node returned a quota_retry_after timestamp, sleep until it passes
    and then clear the field so the next graph.invoke() runs a fresh attempt.

    This implements the autonomous backoff loop for transient quota errors without
    requiring human intervention.
    """
    if not snapshot or not snapshot.values:
        return
    quota_retry_after = snapshot.values.get("quota_retry_after")
    if not quota_retry_after:
        return

    wait = quota_retry_after - time.time()
    if wait > 0:
        log(f"Quota backoff: sleeping {wait:.0f}s before autonomous retry.")
        time.sleep(wait)
    else:
        log("Quota backoff timestamp already passed — retrying immediately.")

    # Clear the timestamp and reset the autonomous retry counter so the next
    # graph.invoke() runs as though quota is resolved.
    try:
        graph.update_state(config, {"quota_retry_after": None, "quota_autonomous_retries": 0})
    except Exception as e:
        log(f"Quota backoff: failed to clear state ({e}); will retry on next run")
        return
    log("Quota backoff cleared. Resuming pipeline.")


def _is_interrupted(snapshot) -> bool:
    """True when the graph is paused at an interrupt() call."""
    return bool(snapshot and snapshot.next)


def _is_complete(snapshot) -> bool:
    """True when all stories have reached the 'done' column."""
    if not snapshot or not snapshot.values:
        return False
    stories = snapshot.values.get("stories", {})
    if not stories:
        return False
    return all(s.get("column") == "done" for s in stories.values())


def _interrupt_value(snapshot) -> dict | None:
    """Extract the interrupt payload from the paused graph snapshot."""
    try:
        for task in (snapshot.tasks or []):
            interrupts = getattr(task, "interrupts", [])
            if interrupts:
                return interrupts[0].value
    except Exception:
        pass
    return None


def _handle_resume(graph, config: dict, snapshot) -> None:
    """Inspect the interrupt type and resume with human feedback if ready."""
    from langgraph.types import Command
    from agile_agent_factory.tools.jira_client import JiraClient

    info = _interrupt_value(snapshot)
    if not info:
        log("No interrupt value found. Resuming with empty feedback.")
        graph.invoke(Command(resume=""), config)
        return

    interrupt_type = info.get("type")
    blocking_key = info.get("blocking_key")
    jira = JiraClient()

    if interrupt_type == "quota":
        if blocking_key and jira.is_flagged(blocking_key):
            log(f"{blocking_key} is still flagged. Quota not yet resolved.")
            return
        log("Quota flag cleared. Resuming pipeline.")
        # Reset autonomous retry counter and clear hitl_type so the story re-enters dispatch
        graph.update_state(config, {
            "quota_autonomous_retries": 0,
            "stories": {blocking_key: {"hitl_type": None}} if blocking_key else {},
        })
        graph.invoke(Command(resume="quota_resolved"), config)
        return

    if interrupt_type in ("refinement", "intervention"):
        if blocking_key and jira.is_flagged(blocking_key):
            log(f"{blocking_key} is still flagged. Awaiting human resolution.")
            return
        feedback = ""
        if blocking_key:
            feedback = jira.get_last_comment_text(blocking_key) or ""
            try:
                jira.clear_flag(blocking_key)
            except Exception:
                pass
        # Clear hitl_type and reset review_retries so the dispatcher re-enters the node
        # fresh with a full retry budget (the node is re-entered from scratch on Send()-resume,
        # so the return value inside the HITL path never updates the checkpoint).
        if blocking_key:
            graph.update_state(config, {"stories": {blocking_key: {"hitl_type": None, "review_retries": 0}}})
        log(f"Flag cleared. Human feedback: {feedback[:200]}")
        graph.invoke(Command(resume=feedback), config)
        return

    if interrupt_type == "blocked":
        # Dispatcher fires this when all dispatchable stories are blocked on HITL.
        # Check each blocked story: if its Jira flag is cleared, unblock it.
        blocked_stories = info.get("blocked_stories", [])
        cleared = {
            sk: {"hitl_type": None, "review_retries": 0}
            for sk in blocked_stories
            if sk and not jira.is_flagged(sk)
        }
        if not cleared:
            log(f"All {len(blocked_stories)} blocked story/stories still flagged. Awaiting human resolution.")
            return
        graph.update_state(config, {"stories": cleared})
        log(f"Unblocked {len(cleared)} story/stories (flag cleared by human).")
        graph.invoke(Command(resume=""), config)
        return

    log(f"Unknown interrupt type '{interrupt_type}'. Resuming with empty feedback.")
    graph.invoke(Command(resume=""), config)


if __name__ == "__main__":
    main()
