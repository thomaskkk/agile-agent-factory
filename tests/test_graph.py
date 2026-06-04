"""Integration tests for the LangGraph StateGraph pipeline.

Uses InMemorySaver so no SQLite files are written. All Jira and LLM calls
are mocked. Tests verify graph topology, routing, and state threading.
"""

from unittest.mock import MagicMock, patch

import pytest

from langgraph.checkpoint.memory import MemorySaver

from agile_agent_factory.graph import build_graph
from agile_agent_factory.state import PipelineState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THREAD = {"configurable": {"thread_id": "test"}}


def _build(checkpointer=None):
    return build_graph(checkpointer=checkpointer or MemorySaver())


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def test_build_graph_returns_compiled_graph():
    g = _build()
    assert g is not None


def test_graph_has_expected_nodes():
    g = _build()
    node_names = set(g.nodes.keys())
    for expected in ["init", "po", "dispatcher", "qa", "ux", "tl", "dev", "test", "review", "finalize"]:
        assert expected in node_names, f"Missing node: {expected}"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_route_dispatcher_returns_finalize_when_all_stories_done():
    from agile_agent_factory.graph import _route_dispatcher

    state: PipelineState = {
        "stories": {"F1-1": {"story_key": "F1-1", "column": "done"}},
        "wip_limits": {},
    }
    result = _route_dispatcher(state)
    assert result == "finalize"


def test_route_dispatcher_returns_sends_for_active_stories():
    from agile_agent_factory.graph import _route_dispatcher
    from langgraph.types import Send

    state: PipelineState = {
        "stories": {"F1-1": {"story_key": "F1-1", "column": "development"}},
        "wip_limits": {"testing": 2},
    }
    result = _route_dispatcher(state)
    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)


# ---------------------------------------------------------------------------
# Init node
# ---------------------------------------------------------------------------

def test_init_node_initializes_counters(tmp_path):
    from agile_agent_factory.nodes import init_node

    # PRODUCT_ROOT is patched to tmp_path; business_idea.md lives at PRODUCT_ROOT/business_idea.md
    business_idea = tmp_path / "business_idea.md"
    business_idea.write_text("Build a thing.")

    with patch("agile_agent_factory.nodes.pipeline.PRODUCT_ROOT", tmp_path):
        result = init_node({})

    assert result["review_retries"] == 0
    assert result["done_count"] == 0
    assert result["business_idea"] == "Build a thing."


def test_init_node_raises_if_no_business_idea(tmp_path):
    from agile_agent_factory.nodes import init_node

    with patch("agile_agent_factory.nodes.pipeline.PRODUCT_ROOT", tmp_path):
        with pytest.raises(FileNotFoundError):
            init_node({})


def test_init_node_preserves_existing_wip_limits(tmp_path):
    from agile_agent_factory.nodes import init_node

    (tmp_path / "business_idea.md").write_text("x")
    custom_wip = {"refinement": 5, "code_review": 2}

    with patch("agile_agent_factory.nodes.pipeline.PRODUCT_ROOT", tmp_path):
        result = init_node({"wip_limits": custom_wip})

    assert result["wip_limits"] == custom_wip


# ---------------------------------------------------------------------------
# PO node
# ---------------------------------------------------------------------------

def test_po_node_is_idempotent_when_stories_exist():
    from agile_agent_factory.nodes import po_node

    state = {"stories": {"F1-1": {"story_key": "F1-1", "column": "refinement"}}}
    result = po_node(state)
    assert result == {}  # idempotency guard — no Jira calls


def test_po_node_creates_stories_from_agent_result():
    from agile_agent_factory.nodes import po_node

    mock_result = {
        "epic_keys": ["F1-E1"],
        "story_keys": ["F1-1", "F1-2"],
        "story_to_epic": {"F1-1": "F1-E1", "F1-2": "F1-E1"},
        "has_ui": False,
    }

    jira = MagicMock()
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira), \
         patch("agile_agent_factory.agents.po_agent.analyze_and_provision", return_value=mock_result):
        # nodes.po_node imports analyze_and_provision lazily from po_agent
        from agile_agent_factory.agents import po_agent
        original = po_agent.analyze_and_provision
        po_agent.analyze_and_provision = MagicMock(return_value=mock_result)
        try:
            result = po_node({})
        finally:
            po_agent.analyze_and_provision = original

    assert len(result["stories"]) == 2
    assert all(s["column"] == "refinement" for s in result["stories"].values())
    assert result["total_count"] == 2
    assert result["has_ui"] is False


def test_po_node_sets_refinement_ux_done_true_when_no_ui():
    from agile_agent_factory.nodes import po_node
    from agile_agent_factory.agents import po_agent

    mock_result = {
        "epic_keys": ["F1-E1"],
        "story_keys": ["F1-1"],
        "story_to_epic": {"F1-1": "F1-E1"},
        "has_ui": False,
    }

    jira = MagicMock()
    original = po_agent.analyze_and_provision
    po_agent.analyze_and_provision = MagicMock(return_value=mock_result)
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = po_node({})
    finally:
        po_agent.analyze_and_provision = original

    story = result["stories"]["F1-1"]
    assert story["refinement_ux_done"] is True   # UX skipped for non-UI stories
    assert story["refinement_qa_done"] is False


def test_po_node_sets_refinement_ux_done_false_when_has_ui():
    from agile_agent_factory.nodes import po_node
    from agile_agent_factory.agents import po_agent

    mock_result = {
        "epic_keys": ["F1-E1"],
        "story_keys": ["F1-1"],
        "story_to_epic": {"F1-1": "F1-E1"},
        "has_ui": True,
    }

    jira = MagicMock()
    original = po_agent.analyze_and_provision
    po_agent.analyze_and_provision = MagicMock(return_value=mock_result)
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = po_node({})
    finally:
        po_agent.analyze_and_provision = original

    story = result["stories"]["F1-1"]
    assert story["refinement_ux_done"] is False   # UX must be run for UI stories
    assert story["refinement_qa_done"] is False


# ---------------------------------------------------------------------------
# QA node
# ---------------------------------------------------------------------------

def test_qa_node_sets_refinement_qa_done_flag():
    from agile_agent_factory.nodes import qa_node
    from agile_agent_factory.agents import qa_agent

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "refinement", "has_ui": False, "refinement_ux_done": True}},
        "gherkin_criteria": {},
    }

    jira = MagicMock()
    original = qa_agent.inject_gherkin_criteria
    qa_agent.inject_gherkin_criteria = MagicMock(return_value={"F1-1": ["Scenario: do thing"]})
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = qa_node(state)
    finally:
        qa_agent.inject_gherkin_criteria = original

    assert result["stories"]["F1-1"]["refinement_qa_done"] is True
    # Column NOT advanced by qa_node — dispatcher/gate handles it
    assert "column" not in result["stories"]["F1-1"]


def test_qa_node_does_not_advance_column():
    """qa_node only sets the flag; dispatcher routes through refinement_gate for column change."""
    from agile_agent_factory.nodes import qa_node
    from agile_agent_factory.agents import qa_agent

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "refinement", "has_ui": False, "refinement_ux_done": True}},
        "gherkin_criteria": {},
    }

    jira = MagicMock()
    original = qa_agent.inject_gherkin_criteria
    qa_agent.inject_gherkin_criteria = MagicMock(return_value={"F1-1": []})
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = qa_node(state)
    finally:
        qa_agent.inject_gherkin_criteria = original

    story_update = result.get("stories", {}).get("F1-1", {})
    assert "column" not in story_update


# ---------------------------------------------------------------------------
# UX node
# ---------------------------------------------------------------------------

def test_ux_node_sets_refinement_ux_done_flag():
    from agile_agent_factory.nodes import ux_node
    from agile_agent_factory.agents import ux_agent

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "refinement", "has_ui": True, "refinement_qa_done": True}},
        "ux_spec": {},
    }

    jira = MagicMock()
    mock_spec = {"screens": [], "flows": [], "ui_type": "web", "technology": "React"}
    original = ux_agent.design_user_experience
    ux_agent.design_user_experience = MagicMock(return_value=mock_spec)
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = ux_node(state)
    finally:
        ux_agent.design_user_experience = original

    assert result["stories"]["F1-1"]["refinement_ux_done"] is True
    assert "column" not in result["stories"]["F1-1"]


# ---------------------------------------------------------------------------
# Refinement gate node
# ---------------------------------------------------------------------------

def test_refinement_gate_advances_column_to_tech_design():
    from agile_agent_factory.nodes import refinement_gate_node

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "refinement",
                              "refinement_qa_done": True, "refinement_ux_done": True}},
    }
    result = refinement_gate_node(state)
    assert result["stories"]["F1-1"]["column"] == "tech_design"


# ---------------------------------------------------------------------------
# TL node
# ---------------------------------------------------------------------------

def test_tl_node_skips_when_no_tech_design_stories():
    from agile_agent_factory.nodes import tl_node

    state = {
        "stories": {"F1-1": {"story_key": "F1-1", "column": "development"}},
        "epic_keys": [],
    }

    jira = MagicMock()
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
        result = tl_node(state)

    assert result == {}


def test_tl_node_advances_all_tech_design_stories_to_development():
    from agile_agent_factory.nodes import tl_node
    from agile_agent_factory.agents import tl_agent

    state = {
        "stories": {
            "F1-1": {"story_key": "F1-1", "column": "tech_design"},
            "F1-2": {"story_key": "F1-2", "column": "tech_design"},
            "F1-3": {"story_key": "F1-3", "column": "code_review"},  # not in tech_design
        },
        "epic_keys": [],
        "gherkin_criteria": {},
        "ux_spec": {},
        "subtasks": {},
        "has_ui": False,
        "dependencies": [],
    }

    arch = {"files": [], "subtasks": [], "import_rules": "", "test_command": "", "dependencies": []}
    mock_result = {"architecture": arch, "subtasks": {}, "dependencies": []}

    jira = MagicMock()
    original = tl_agent.design_architecture
    tl_agent.design_architecture = MagicMock(return_value=mock_result)
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = tl_node(state)
    finally:
        tl_agent.design_architecture = original

    assert result["stories"]["F1-1"]["column"] == "development"
    assert result["stories"]["F1-2"]["column"] == "development"
    assert "F1-3" not in result["stories"]  # only tech_design stories advanced


# ---------------------------------------------------------------------------
# Review node — routing
# ---------------------------------------------------------------------------

def test_review_node_approved_marks_story_done():
    from agile_agent_factory.nodes import review_node
    from agile_agent_factory.agents import reviewer_agent

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "code_review", "review_retries": 0}},
        "epic_keys": [],
        "review_retries": 0,
    }

    jira = MagicMock()
    original = reviewer_agent.review_patch
    reviewer_agent.review_patch = MagicMock(return_value={"approved": True, "reason": ""})
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = review_node(state)
    finally:
        reviewer_agent.review_patch = original

    assert result["review_approved"] is True
    assert result["stories"]["F1-1"]["column"] == "done"


def test_review_node_rejected_keeps_story_in_code_review():
    from agile_agent_factory.nodes import review_node
    from agile_agent_factory.agents import reviewer_agent

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "code_review", "review_retries": 0}},
        "epic_keys": [],
        "review_retries": 0,
    }

    jira = MagicMock()
    original = reviewer_agent.review_patch
    reviewer_agent.review_patch = MagicMock(return_value={"approved": False, "reason": "Missing tests"})
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = review_node(state)
    finally:
        reviewer_agent.review_patch = original

    assert result["review_approved"] is False
    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert result["stories"]["F1-1"]["review_status"] == "rework_needed"
    assert result["review_retries"] == 1
    assert result["stories"]["F1-1"]["review_rejection_reason"] == "Missing tests"


def test_review_node_max_retries_exhausted_triggers_hitl():
    from agile_agent_factory.config import MAX_REVIEW_RETRIES
    from agile_agent_factory.nodes import review_node
    from agile_agent_factory.agents import reviewer_agent

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "code_review", "review_retries": MAX_REVIEW_RETRIES}},
        "epic_keys": [],
        "review_retries": MAX_REVIEW_RETRIES,
    }

    jira = MagicMock()
    original = reviewer_agent.review_patch
    reviewer_agent.review_patch = MagicMock(return_value={"approved": False, "reason": "Still failing"})
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira), \
             patch("langgraph.types.interrupt") as mock_interrupt:
            mock_interrupt.return_value = None
            result = review_node(state)
    finally:
        reviewer_agent.review_patch = original

    mock_interrupt.assert_called_once()
    call_arg = mock_interrupt.call_args[0][0]
    assert call_arg["type"] == "intervention"
    assert call_arg["blocking_key"] == "F1-1"
    assert result["review_retries"] == 0
    assert result["stories"]["F1-1"]["review_status"] == "pending_review"


def test_dev_node_rework_path_stays_in_code_review():
    from agile_agent_factory.nodes import dev_node

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {
            "story_key": "F1-1",
            "column": "code_review",
            "review_status": "rework_needed",
            "review_rejection_reason": "Missing tests",
        }},
        "epic_keys": [],
        "subtasks": {},
    }

    jira = MagicMock()
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira), \
         patch("agile_agent_factory.nodes.pipeline._generate_code_with_llm") as mock_gen, \
         patch("agile_agent_factory.nodes.pipeline.BLUEPRINT_PATH") as mock_bp:
        mock_bp.exists.return_value = False
        result = dev_node(state)

    story_update = result["stories"]["F1-1"]
    assert story_update["review_status"] == "pending_review"
    assert story_update["review_rejection_reason"] == ""
    assert "column" not in story_update
    # No Jira transition to Development on rework path
    transition_calls = [str(c) for c in jira.transition_issue.call_args_list]
    assert not any("Development" in c for c in transition_calls)


# ---------------------------------------------------------------------------
# Finalize node
# ---------------------------------------------------------------------------

def test_finalize_node_marks_all_stories_done():
    from agile_agent_factory.nodes import finalize_node
    from agile_agent_factory.agents import readme_agent, sre_agent

    state = {
        "stories": {
            "F1-1": {"story_key": "F1-1", "column": "code_review"},
            "F1-2": {"story_key": "F1-2", "column": "code_review"},
        },
        "epic_keys": [],
        "subtasks": {},
        "has_ui": False,
        "dependencies": [],
        "gherkin_criteria": {},
        "ux_spec": {},
        "architecture": {},
        "review_retries": 0,
    }

    jira = MagicMock()
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira), \
         patch.object(readme_agent, "generate_readme"), \
         patch.object(sre_agent, "emulate_deployment"):
        result = finalize_node(state)

    for sk in ["F1-1", "F1-2"]:
        assert result["stories"][sk]["column"] == "done"
    assert result["done_count"] == 2


# ---------------------------------------------------------------------------
# merge_stories reducer
# ---------------------------------------------------------------------------

def test_merge_stories_reducer_merges_without_overwriting():
    from agile_agent_factory.state import merge_stories

    current = {
        "F1-1": {"column": "refinement", "retries": 0},
        "F1-2": {"column": "development", "retries": 2},
    }
    update = {
        "F1-1": {"column": "tech_design"},  # only column changes
    }
    merged = merge_stories(current, update)

    assert merged["F1-1"]["column"] == "tech_design"
    assert merged["F1-1"]["retries"] == 0  # preserved
    assert merged["F1-2"]["column"] == "development"  # untouched


def test_merge_stories_reducer_adds_new_stories():
    from agile_agent_factory.state import merge_stories

    current = {"F1-1": {"column": "refinement"}}
    update = {"F1-2": {"column": "refinement"}}
    merged = merge_stories(current, update)

    assert "F1-1" in merged
    assert "F1-2" in merged


def test_merge_stories_parallel_qa_ux_race_condition_resolved():
    """Simulates parallel qa+ux both returning column='refinement' — merged result stays refinement."""
    from agile_agent_factory.state import merge_stories

    # Both QA and UX read the same initial state (both flags False) and return:
    qa_update = {"F1-1": {"refinement_qa_done": True}}   # no column key
    ux_update = {"F1-1": {"refinement_ux_done": True}}   # no column key

    initial = {"F1-1": {"column": "refinement", "refinement_qa_done": False, "refinement_ux_done": False}}

    after_qa = merge_stories(initial, qa_update)
    after_both = merge_stories(after_qa, ux_update)

    # Both flags set, column stays refinement — gate node handles the advance
    assert after_both["F1-1"]["refinement_qa_done"] is True
    assert after_both["F1-1"]["refinement_ux_done"] is True
    assert after_both["F1-1"]["column"] == "refinement"  # gate, not qa/ux, advances this
