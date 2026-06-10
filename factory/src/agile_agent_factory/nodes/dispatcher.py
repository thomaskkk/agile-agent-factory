"""Kanban dispatcher — routes stories to agent nodes with right-to-left priority.

The dispatcher is a deterministic routing function (no LLM calls). It scans
columns from right to left (closest to Done first), respects WIP limits, and
returns Send() commands that fan stories out to the appropriate agent nodes.

Column flow:
  backlog → refinement → tech_design → development → testing → code_review → done
"""

from __future__ import annotations

import re

from langgraph.types import Send

from agile_agent_factory.tools.logger import log
from agile_agent_factory.state import PipelineState

# Right-to-left scan order (skip backlog — PO handles that)
COLUMNS_RTL = ["code_review", "testing", "development", "tech_design", "refinement"]

NEXT_COLUMN = {
    "refinement": "tech_design",
    "tech_design": "development",
    "development": "testing",
    "testing": "code_review",
    "code_review": "done",
}

# Maps the column a story is currently IN to the agent node that processes it
COLUMN_TO_AGENT = {
    "refinement": None,       # special: QA + optional UX (see _dispatch_refinement)
    "tech_design": "tl",
    "development": "dev",
    "testing": "test",
    "code_review": "review",
}


def _wip_count(stories: dict, column: str) -> int:
    return sum(1 for s in stories.values() if s.get("column") == column)


def _story_dispatch_order(story: dict) -> tuple[str, int, str]:
    """Sort oldest-first within a column using Jira issue-key sequence as a stable proxy."""
    story_key = str(story.get("story_key") or "")
    match = re.match(r"^(.*?)-(\d+)$", story_key)
    if match:
        return (match.group(1), int(match.group(2)), story_key)
    return ("~", 10**12, story_key)


def _dispatch_refinement(stories: dict, story: dict, state: PipelineState) -> list[Send]:
    """
    Refinement is special: QA always runs; UX only runs when has_ui=True.
    Both must complete before the story advances to tech_design.

    qa_node and ux_node only set flags — they do NOT advance the column.
    When both flags are set, dispatch to refinement_gate to advance the column.
    This avoids a race condition where parallel qa/ux both try to write column="tech_design".
    """
    sk = story["story_key"]
    has_ui = story.get("has_ui", state.get("has_ui", False))
    qa_done = story.get("refinement_qa_done", False)
    ux_done = story.get("refinement_ux_done", False) or not has_ui

    if qa_done and ux_done:
        # Both sub-phases done but column not yet advanced → go through gate
        return [Send("refinement_gate", {**state, "active_story_key": sk})]

    sends: list[Send] = []
    if not qa_done:
        sends.append(Send("qa", {**state, "active_story_key": sk}))
    if not ux_done:
        sends.append(Send("ux", {**state, "active_story_key": sk}))
    return sends


def dispatch_stories(state: PipelineState) -> list[Send] | str:
    """Central dispatcher: scan columns RTL, emit Send() to agent nodes.

    Returns:
        A list of Send() objects (possibly empty if blocked).
        The string "finalize" when all stories are done.
    """
    from langgraph.types import interrupt

    stories = state.get("stories", {})
    wip_limits = state.get("wip_limits", {})

    if not stories:
        log("Dispatcher: no stories found — routing to finalize.")
        return "finalize"

    if all(s.get("column") == "done" for s in stories.values()):
        log("Dispatcher: all stories done — routing to finalize.")
        return "finalize"

    sends: list[Send] = []

    for column in COLUMNS_RTL:
        # Stories currently in this column that are not paused for HITL
        ready = [
            s for s in stories.values()
            if s.get("column") == column and not s.get("hitl_type")
        ]
        ready.sort(key=_story_dispatch_order)
        if not ready:
            continue

        if column == "refinement":
            next_col = "tech_design"
            wip_limit = wip_limits.get(next_col, 999)
            current_in_next = _wip_count(stories, next_col)

            gate_ready, ux_pending, qa_pending = [], [], []
            for story in ready:
                qa_done = story.get("refinement_qa_done", False)
                has_ui = story.get("has_ui", state.get("has_ui", False))
                ux_done = story.get("refinement_ux_done", False) or not has_ui
                if qa_done and ux_done:
                    gate_ready.append(story)
                elif qa_done:
                    ux_pending.append(story)
                else:
                    qa_pending.append(story)

            for story in gate_ready:
                sends.extend(_dispatch_refinement(stories, story, state))
            for story in ux_pending:
                sends.extend(_dispatch_refinement(stories, story, state))

            available_slots = max(0, wip_limit - current_in_next - len(gate_ready) - len(ux_pending))
            for story in qa_pending[:available_slots]:
                sends.extend(_dispatch_refinement(stories, story, state))
            continue

        if column == "tech_design":
            # TL is batch: one Send for ALL tech_design stories together.
            # tl_node processes only the subset included in active_story_keys so
            # development WIP limits are still respected.
            next_col = NEXT_COLUMN[column]
            wip_limit = wip_limits.get(next_col, 999)
            current_in_next = _wip_count(stories, next_col)
            available_slots = max(0, wip_limit - current_in_next)
            if available_slots > 0:
                batch = ready[:available_slots]
                log(f"Dispatcher: dispatching {len(batch)} story/stories to tl (batch)")
                sends.append(
                    Send(
                        "tl",
                        {**state, "active_story_keys": [story["story_key"] for story in batch]},
                    )
                )
            continue

        if column == "code_review":
            # Stories in code_review are split by sub-state: rework_needed → dev, otherwise → review.
            # Both sub-states still occupy the code_review WIP slot (no WIP gate here).
            rework = [s for s in ready if s.get("review_status") == "rework_needed"]
            pending = [s for s in ready if s.get("review_status") != "rework_needed"]
            for story in rework:
                sk = story["story_key"]
                log(f"Dispatcher: dispatching {sk} (code_review rework) → dev")
                sends.append(Send("dev", {**state, "active_story_key": sk}))
            for story in pending:
                sk = story["story_key"]
                log(f"Dispatcher: dispatching {sk} (code_review pending) → review")
                sends.append(Send("review", {**state, "active_story_key": sk}))
            continue

        next_col = NEXT_COLUMN[column]
        wip_limit = wip_limits.get(next_col, 999)
        current_in_next = _wip_count(stories, next_col)
        available_slots = max(0, wip_limit - current_in_next)

        agent = COLUMN_TO_AGENT[column]
        if agent is None:
            continue  # Should not happen for non-refinement or tech_design columns

        for story in ready[:available_slots]:
            sk = story["story_key"]
            log(f"Dispatcher: dispatching {sk} from {column} → {agent}")
            sends.append(Send(agent, {**state, "active_story_key": sk}))

    if sends:
        return sends

    # No work dispatched — check whether we're blocked waiting on HITL
    blocked = [s for s in stories.values() if s.get("hitl_type")]
    if blocked:
        log(f"Dispatcher: {len(blocked)} story/stories blocked on HITL. Interrupting for human.")
        interrupt({"type": "blocked", "blocked_stories": [s["story_key"] for s in blocked]})
        # After resume, we fall through to return an empty list — graph will re-call dispatcher
        return []

    # WIP-limited: stories exist but all slots are full — return empty (graph stays in dispatcher)
    log("Dispatcher: all ready stories are WIP-limited. Will retry after in-progress stories advance.")
    return []
