"""Tests for the kanban dispatcher — right-to-left priority, WIP limits, Send() output."""

from unittest.mock import MagicMock, patch

import pytest

from langgraph.types import Send

from agile_agent_factory.nodes.dispatcher import (
    COLUMNS_RTL,
    COLUMN_TO_AGENT,
    NEXT_COLUMN,
    dispatch_stories,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(stories: dict, wip_limits: dict | None = None) -> dict:
    return {
        "stories": stories,
        "wip_limits": wip_limits or {
            "refinement": 3,
            "tech_design": 2,
            "development": 2,
            "testing": 2,
            "code_review": 1,
        },
        "has_ui": False,
        "epic_keys": [],
    }


def _story(key: str, column: str, **extra) -> tuple[str, dict]:
    return key, {"story_key": key, "column": column, **extra}


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

def test_routes_to_finalize_when_all_done():
    state = _make_state({"F1-1": {"story_key": "F1-1", "column": "done"}})
    result = dispatch_stories(state)
    assert result == "finalize"


def test_routes_to_finalize_when_no_stories():
    state = _make_state({})
    result = dispatch_stories(state)
    assert result == "finalize"


def test_dispatches_code_review_story_to_review():
    state = _make_state({"F1-1": {"story_key": "F1-1", "column": "code_review"}})
    result = dispatch_stories(state)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Send)
    assert result[0].node == "review"


def test_dispatches_testing_story_to_test():
    state = _make_state({"F1-1": {"story_key": "F1-1", "column": "testing"}})
    result = dispatch_stories(state)
    assert isinstance(result, list)
    assert result[0].node == "test"


def test_dispatches_development_story_to_dev():
    state = _make_state({"F1-1": {"story_key": "F1-1", "column": "development"}})
    result = dispatch_stories(state)
    assert isinstance(result, list)
    assert result[0].node == "dev"


def test_dispatches_tech_design_story_to_tl():
    state = _make_state({"F1-1": {"story_key": "F1-1", "column": "tech_design"}})
    result = dispatch_stories(state)
    assert isinstance(result, list)
    assert result[0].node == "tl"


def test_tl_dispatched_once_for_multiple_tech_design_stories():
    """TL is batch — one Send regardless of how many stories are in tech_design."""
    state = _make_state({
        "F1-1": {"story_key": "F1-1", "column": "tech_design"},
        "F1-2": {"story_key": "F1-2", "column": "tech_design"},
        "F1-3": {"story_key": "F1-3", "column": "tech_design"},
    })
    result = dispatch_stories(state)
    tl_sends = [s for s in result if s.node == "tl"]
    assert len(tl_sends) == 1  # single batch Send, not one per story


# ---------------------------------------------------------------------------
# Right-to-left priority
# ---------------------------------------------------------------------------

def test_right_to_left_prefers_code_review_over_testing():
    """code_review story is dispatched first; testing story is blocked (next col is full)."""
    state = _make_state({
        "F1-1": {"story_key": "F1-1", "column": "code_review"},
        "F1-2": {"story_key": "F1-2", "column": "testing"},
    })
    result = dispatch_stories(state)
    # code_review WIP=1: F1-1 occupies it → dispatched to "review"
    # testing → next col is code_review, which is full (F1-1 is there) → 0 slots → F1-2 blocked
    nodes = {s.node for s in result}
    assert "review" in nodes
    assert "test" not in nodes  # F1-2 is WIP-blocked, not dispatched


def test_right_to_left_dispatches_rightmost_when_wip_limits_slot():
    """Two stories at different stages with explicit WIP limits — rightmost dispatched."""
    state = _make_state(
        {
            "F1-1": {"story_key": "F1-1", "column": "code_review"},
            "F1-2": {"story_key": "F1-2", "column": "development"},
        },
        wip_limits={"code_review": 1, "testing": 2, "development": 2, "tech_design": 2, "refinement": 3},
    )
    result = dispatch_stories(state)
    # F1-1 in code_review → dispatched to "review"
    # F1-2 in development → next col is testing (WIP=2, 0 occupied) → dispatched to "dev"
    nodes = {s.node for s in result}
    assert "review" in nodes
    assert "dev" in nodes


# ---------------------------------------------------------------------------
# WIP limits
# ---------------------------------------------------------------------------

def test_wip_limit_blocks_advancement_to_full_column():
    """Stories in testing are blocked (not dispatched) when code_review WIP is full."""
    state = _make_state(
        {
            "F1-1": {"story_key": "F1-1", "column": "testing"},
            "F1-2": {"story_key": "F1-2", "column": "testing"},
            "F1-3": {"story_key": "F1-3", "column": "code_review"},  # takes the 1 slot
        },
        wip_limits={"code_review": 1, "testing": 3, "development": 2, "tech_design": 2, "refinement": 3},
    )
    result = dispatch_stories(state)
    # F1-3 in code_review → dispatched to "review"
    # F1-1 and F1-2 in testing → next col is code_review (WIP=1, occupied by F1-3) → 0 slots → blocked
    assert isinstance(result, list)
    review_sends = [s for s in result if s.node == "review"]
    test_sends = [s for s in result if s.node == "test"]
    assert len(review_sends) == 1   # only F1-3 dispatched
    assert len(test_sends) == 0     # F1-1 and F1-2 are WIP-blocked


def test_wip_limit_caps_concurrent_dispatches():
    # development WIP=1, two stories ready in tech_design
    state = _make_state(
        {
            "F1-1": {"story_key": "F1-1", "column": "tech_design"},
            "F1-2": {"story_key": "F1-2", "column": "tech_design"},
        },
        wip_limits={"development": 1, "testing": 2, "code_review": 1, "tech_design": 2, "refinement": 3},
    )
    result = dispatch_stories(state)
    # Only 1 slot in development, so only 1 story dispatched to tl (which moves to development)
    tl_sends = [s for s in result if s.node == "tl"]
    assert len(tl_sends) == 1


# ---------------------------------------------------------------------------
# Refinement: QA + UX sub-phases
# ---------------------------------------------------------------------------

def test_refinement_dispatches_qa_when_not_done():
    state = _make_state({
        "F1-1": {
            "story_key": "F1-1",
            "column": "refinement",
            "has_ui": False,
            "refinement_qa_done": False,
            "refinement_ux_done": True,
        }
    })
    result = dispatch_stories(state)
    assert isinstance(result, list)
    assert any(s.node == "qa" for s in result)


def test_refinement_dispatches_ux_when_has_ui_and_not_done():
    state = _make_state({
        "F1-1": {
            "story_key": "F1-1",
            "column": "refinement",
            "has_ui": True,
            "refinement_qa_done": True,
            "refinement_ux_done": False,
        }
    })
    result = dispatch_stories(state)
    assert isinstance(result, list)
    assert any(s.node == "ux" for s in result)


def test_refinement_dispatches_both_qa_and_ux_when_has_ui():
    state = _make_state({
        "F1-1": {
            "story_key": "F1-1",
            "column": "refinement",
            "has_ui": True,
            "refinement_qa_done": False,
            "refinement_ux_done": False,
        }
    })
    result = dispatch_stories(state)
    nodes = {s.node for s in result}
    assert "qa" in nodes
    assert "ux" in nodes


def test_refinement_dispatches_gate_when_both_done_but_column_not_advanced():
    """When both QA and UX flags are set (post parallel fan-out), route to refinement_gate."""
    state = _make_state({
        "F1-1": {
            "story_key": "F1-1",
            "column": "refinement",
            "has_ui": True,
            "refinement_qa_done": True,
            "refinement_ux_done": True,
        }
    })
    result = dispatch_stories(state)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].node == "refinement_gate"
    assert result[0].arg.get("active_story_key") == "F1-1"


# ---------------------------------------------------------------------------
# HITL-blocked stories
# ---------------------------------------------------------------------------

def test_hitl_blocked_story_is_skipped_by_dispatcher():
    state = _make_state({
        "F1-1": {"story_key": "F1-1", "column": "testing", "hitl_type": "intervention"},
        "F1-2": {"story_key": "F1-2", "column": "development"},
    })
    result = dispatch_stories(state)
    # F1-1 is HITL-blocked → skipped; F1-2 in development → dispatched to "dev"
    assert isinstance(result, list)
    assert all(s.node != "test" for s in result)  # F1-1 should NOT be dispatched
    assert any(s.node == "dev" for s in result)   # F1-2 should be dispatched


# ---------------------------------------------------------------------------
# active_story_key in sent state
# ---------------------------------------------------------------------------

def test_send_includes_active_story_key():
    state = _make_state({"F1-1": {"story_key": "F1-1", "column": "development"}})
    result = dispatch_stories(state)
    assert isinstance(result, list)
    assert len(result) == 1
    sent_state = result[0].arg
    assert sent_state.get("active_story_key") == "F1-1"


def test_code_review_rework_needed_routes_to_dev():
    """A code_review story with review_status=rework_needed is dispatched to dev, not review."""
    state = _make_state({
        "F1-1": {"story_key": "F1-1", "column": "code_review", "review_status": "rework_needed"},
    })
    result = dispatch_stories(state)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].node == "dev"
    assert result[0].arg.get("active_story_key") == "F1-1"
