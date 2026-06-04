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

    # Fresh start or crash recovery — invoke the graph
    log("Reading business idea from ../business_idea.md.")
    graph.invoke({}, config)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

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
        log(f"Flag cleared. Human feedback: {feedback[:200]}")
        graph.invoke(Command(resume=feedback), config)
        return

    log(f"Unknown interrupt type '{interrupt_type}'. Resuming with empty feedback.")
    graph.invoke(Command(resume=""), config)


if __name__ == "__main__":
    main()
