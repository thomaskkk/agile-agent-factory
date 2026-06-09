"""Integration tests for the LangGraph StateGraph pipeline.

Uses InMemorySaver so no SQLite files are written. All Jira and LLM calls
are mocked. Tests verify graph topology, routing, and state threading.
"""

from unittest.mock import MagicMock, patch

import pytest

from langgraph.checkpoint.memory import MemorySaver

from agile_agent_factory.agents.contract import AgentResult
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
         patch("agile_agent_factory.agents.po_agent.analyze_and_provision", return_value=AgentResult(payload=mock_result)):
        # nodes.po_node imports analyze_and_provision lazily from po_agent
        from agile_agent_factory.agents import po_agent
        original = po_agent.analyze_and_provision
        po_agent.analyze_and_provision = MagicMock(return_value=AgentResult(payload=mock_result))
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
    po_agent.analyze_and_provision = MagicMock(return_value=AgentResult(payload=mock_result))
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
    po_agent.analyze_and_provision = MagicMock(return_value=AgentResult(payload=mock_result))
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
    qa_agent.inject_gherkin_criteria = MagicMock(return_value=AgentResult(payload={"gherkin_criteria": {"F1-1": ["Scenario: do thing"]}, "test_contracts": {"F1-1": {}}}))
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
    qa_agent.inject_gherkin_criteria = MagicMock(return_value=AgentResult(payload={"gherkin_criteria": {"F1-1": []}, "test_contracts": {"F1-1": {}}}))
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = qa_node(state)
    finally:
        qa_agent.inject_gherkin_criteria = original

    story_update = result.get("stories", {}).get("F1-1", {})
    assert "column" not in story_update


def test_qa_node_stores_test_contract_in_story_state():
    """qa_node must store test_contract returned by inject_gherkin_criteria."""
    from agile_agent_factory.nodes import qa_node
    from agile_agent_factory.agents import qa_agent

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "refinement", "has_ui": False, "refinement_ux_done": True}},
        "gherkin_criteria": {},
    }
    jira = MagicMock()
    mock_tc = {"test_file": "tests/test_feature.py", "test_functions": ["test_do_thing"], "target_imports": ["from app.feature import do_thing"], "fixtures": [], "sample_data": [], "edge_cases": []}
    original = qa_agent.inject_gherkin_criteria
    qa_agent.inject_gherkin_criteria = MagicMock(return_value=AgentResult(payload={"gherkin_criteria": {"F1-1": ["Scenario: do thing"]}, "test_contracts": {"F1-1": mock_tc}}))
    try:
        with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
            result = qa_node(state)
    finally:
        qa_agent.inject_gherkin_criteria = original

    assert result["stories"]["F1-1"]["test_contract"] == mock_tc
    assert result["stories"]["F1-1"]["refinement_qa_done"] is True


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
    ux_agent.design_user_experience = MagicMock(return_value=AgentResult(payload={"ux_spec": mock_spec}))
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
                              "has_ui": False, "refinement_qa_done": True, "refinement_ux_done": True,
                              "gherkin_criteria": ["Scenario: Do thing\n  Given X\n  When Y\n  Then Z"]}},
        "business_idea": "Build a thing.",
    }
    jira = MagicMock()
    jira._request.return_value = {"fields": {"summary": "Do thing"}}
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
        result = refinement_gate_node(state)
    assert result["stories"]["F1-1"]["column"] == "tech_design"
    assert result["stories"]["F1-1"]["ready_validated"] is True


def test_refinement_gate_invalid_contract_keeps_story_in_refinement():
    from agile_agent_factory.nodes import refinement_gate_node

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "refinement",
                              "has_ui": False, "refinement_qa_done": True, "refinement_ux_done": True,
                              "gherkin_criteria": []}},
        "business_idea": "Build a thing.",
    }
    jira = MagicMock()
    jira._request.return_value = {"fields": {"summary": "Do thing"}}
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
        result = refinement_gate_node(state)

    update = result["stories"]["F1-1"]
    assert "column" not in update
    assert update["ready_validated"] is False
    assert update["refinement_qa_done"] is False


def test_refinement_gate_valid_ui_contract_advances_to_tech_design():
    from agile_agent_factory.nodes import refinement_gate_node

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"story_key": "F1-1", "column": "refinement",
                              "has_ui": True, "refinement_qa_done": True, "refinement_ux_done": True,
                              "gherkin_criteria": ["Scenario: Submit form\n  Given a form\n  When I submit it\n  Then I see confirmation"],
                              "ux_spec": {"ui_type": "web", "technology": "Flask", "screens_or_flows": [
                                  {"story_key": "F1-1", "name": "Submit form", "purpose": "Collect input", "key_elements": ["submit"]}
                              ]}}},
        "business_idea": "Build a web form.",
    }
    jira = MagicMock()
    jira._request.return_value = {"fields": {"summary": "Submit form"}}
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira):
        result = refinement_gate_node(state)

    assert result["stories"]["F1-1"]["column"] == "tech_design"
    assert result["stories"]["F1-1"]["ready_contract"]["ui_flow_reference"]["name"] == "Submit form"


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
    tl_agent.design_architecture = MagicMock(return_value=AgentResult(payload=mock_result))
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
    reviewer_agent.review_patch = MagicMock(return_value=AgentResult(success=True, payload={"approved": True, "reason": ""}))
    try:
        with patch("agile_agent_factory.nodes.review_node.JiraClient", return_value=jira):
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
    reviewer_agent.review_patch = MagicMock(return_value=AgentResult(success=False, payload={"approved": False, "reason": "Missing tests"}))
    try:
        with patch("agile_agent_factory.nodes.review_node.JiraClient", return_value=jira):
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
    reviewer_agent.review_patch = MagicMock(return_value=AgentResult(success=False, payload={"approved": False, "reason": "Still failing"}))
    try:
        with patch("agile_agent_factory.nodes.review_node.JiraClient", return_value=jira), \
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
    with patch("agile_agent_factory.nodes.dev_node.JiraClient", return_value=jira), \
         patch("agile_agent_factory.nodes.dev_node._generate_code_with_llm_guarded", return_value=["app/auth.py"]):
        result = dev_node(state)

    story_update = result["stories"]["F1-1"]
    assert story_update["review_status"] == "pending_review"
    assert story_update["review_rejection_reason"] == ""
    assert "column" not in story_update
    # No Jira transition to Development on rework path
    transition_calls = [str(c) for c in jira.transition_to.call_args_list]
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


def test_finalize_node_no_regression_blockers_behaves_as_before():
    """finalize_node with no regression_blockers: identical behaviour, no Jira comment posted."""
    from agile_agent_factory.nodes import finalize_node
    from agile_agent_factory.agents import readme_agent, sre_agent

    state = {
        "stories": {
            "F1-1": {"story_key": "F1-1", "column": "code_review"},
            "F1-2": {"story_key": "F1-2", "column": "code_review"},
        },
        "epic_keys": [],
        "subtasks": {},
    }

    jira = MagicMock()
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira), \
         patch.object(readme_agent, "generate_readme"), \
         patch.object(sre_agent, "emulate_deployment"):
        result = finalize_node(state)

    # Stories marked done, done_count correct
    assert result["done_count"] == 2
    assert all(s["column"] == "done" for s in result["stories"].values())
    # No Jira regression comment posted
    jira.add_comment_adf.assert_not_called()


def test_finalize_node_requeues_done_owner_for_regression_blocker():
    """finalize_node must reopen the owning done story instead of only warning."""
    from agile_agent_factory.nodes import finalize_node
    from agile_agent_factory.agents import readme_agent, sre_agent

    # F1-2 owns app/feature.py via its test_contract; F1-1 quarantined a regression in that file.
    state = {
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "done",
                "test_contract": {
                    "test_file": "tests/test_main.py",
                    "target_imports": ["from app.main import run"],
                },
                "regression_blockers": ["app/feature.py"],
            },
            "F1-2": {
                "story_key": "F1-2",
                "column": "done",
                "test_contract": {
                    "test_file": "tests/test_feature.py",
                    "target_imports": ["from app.feature import do_thing"],
                },
                "regression_blockers": [],
            },
        },
        "epic_keys": [],
        "subtasks": {},
    }

    jira = MagicMock()
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira), \
         patch.object(readme_agent, "generate_readme") as readme_mock, \
         patch.object(sre_agent, "emulate_deployment") as deploy_mock:
        result = finalize_node(state)

    assert "done_count" not in result
    assert result["stories"]["F1-2"]["column"] == "testing"
    assert "app/feature.py" in result["stories"]["F1-2"]["incoming_regression_files"]
    assert result["stories"]["F1-1"]["regression_blockers"] == []
    readme_mock.assert_not_called()
    deploy_mock.assert_not_called()
    jira.add_comment_adf.assert_called_once()
    call_args = jira.add_comment_adf.call_args
    assert call_args[0][0] == "F1-2"
    comment_text = call_args[0][1]["content"][0]["content"][0]["text"]
    assert "app/feature.py" in comment_text
    assert "reopened" in comment_text.lower() or "requeued" in comment_text.lower()


def test_finalize_node_skips_jira_warning_when_owning_story_not_done():
    """finalize_node: if the owning story is not in 'done', skip the Jira warning (will resolve naturally)."""
    from agile_agent_factory.nodes import finalize_node
    from agile_agent_factory.agents import readme_agent, sre_agent

    # F1-2 owns app/feature.py but it is still in 'testing'; F1-1 has the blocker.
    state = {
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "done",
                "test_contract": {
                    "test_file": "tests/test_main.py",
                    "target_imports": ["from app.main import run"],
                },
                "regression_blockers": ["app/feature.py"],
            },
            "F1-2": {
                "story_key": "F1-2",
                "column": "testing",  # not done yet
                "test_contract": {
                    "test_file": "tests/test_feature.py",
                    "target_imports": ["from app.feature import do_thing"],
                },
                "regression_blockers": [],
            },
        },
        "epic_keys": [],
        "subtasks": {},
    }

    jira = MagicMock()
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira), \
         patch.object(readme_agent, "generate_readme"), \
         patch.object(sre_agent, "emulate_deployment"):
        result = finalize_node(state)

    # Both stories still get marked done by finalize
    assert result["done_count"] == 2
    # No Jira comment — owning story hasn't finished, blocker will resolve naturally
    jira.add_comment_adf.assert_not_called()


def test_finalize_node_no_owner_for_regression_blocker():
    """finalize_node: if no story owns the regression blocker file, warn on the quarantined story."""
    from agile_agent_factory.nodes import finalize_node
    from agile_agent_factory.agents import readme_agent, sre_agent

    # F1-1 quarantined a regression in app/orphan.py, but no story's test_contract covers it.
    state = {
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "done",
                "test_contract": {
                    "test_file": "tests/test_main.py",
                    "target_imports": ["from app.main import run"],
                },
                "regression_blockers": ["app/orphan.py"],
            },
        },
        "epic_keys": [],
        "subtasks": {},
    }

    jira = MagicMock()
    with patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira), \
         patch.object(readme_agent, "generate_readme"), \
         patch.object(sre_agent, "emulate_deployment"):
        result = finalize_node(state)

    # Story marked done
    assert result["done_count"] == 1
    assert result["stories"]["F1-1"]["column"] == "done"
    jira.add_comment_adf.assert_called_once()
    call_args = jira.add_comment_adf.call_args
    assert call_args[0][0] == "F1-1"
    comment_text = call_args[0][1]["content"][0]["content"][0]["text"]
    assert "app/orphan.py" in comment_text


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


# ---------------------------------------------------------------------------
# Quota autonomous resume — raise_quota_interrupt
# ---------------------------------------------------------------------------

def test_raise_quota_interrupt_returns_patch_on_first_autonomous_retry():
    """When autonomous_retries < max, raise_quota_interrupt must return a state patch
    with quota_retry_after set and must NOT call interrupt()."""
    from agile_agent_factory.nodes.helpers import raise_quota_interrupt
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded

    jira = MagicMock()
    exc = LLMQuotaExceeded("anthropic", "rate limit")

    with patch("agile_agent_factory.nodes.helpers.interrupt") as mock_interrupt, \
         patch("agile_agent_factory.nodes.helpers.time") as mock_time:
        mock_time.time.return_value = 1000.0
        state = {"quota_autonomous_retries": 0}
        patch_dict = raise_quota_interrupt(jira, "F1-1", exc, state=state, max_autonomous_retries=3)

    mock_interrupt.assert_not_called()
    assert "quota_retry_after" in patch_dict
    assert patch_dict["quota_retry_after"] == 1000.0 + 30  # 30s * 2^0
    assert patch_dict["quota_autonomous_retries"] == 1


def test_raise_quota_interrupt_backoff_doubles_each_retry():
    """Second retry uses 60s backoff (30 * 2^1), third uses 120s (30 * 2^2)."""
    from agile_agent_factory.nodes.helpers import raise_quota_interrupt
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded

    jira = MagicMock()
    exc = LLMQuotaExceeded("anthropic", "rate limit")

    with patch("agile_agent_factory.nodes.helpers.interrupt"), \
         patch("agile_agent_factory.nodes.helpers.time") as mock_time:
        mock_time.time.return_value = 1000.0

        patch1 = raise_quota_interrupt(jira, "F1-1", exc, state={"quota_autonomous_retries": 1}, max_autonomous_retries=3)
        assert patch1["quota_retry_after"] == 1000.0 + 60   # 30 * 2^1

        patch2 = raise_quota_interrupt(jira, "F1-1", exc, state={"quota_autonomous_retries": 2}, max_autonomous_retries=3)
        assert patch2["quota_retry_after"] == 1000.0 + 120  # 30 * 2^2


def test_raise_quota_interrupt_calls_interrupt_when_budget_exhausted():
    """When autonomous_retries >= max, raise_quota_interrupt must call interrupt()."""
    from agile_agent_factory.nodes.helpers import raise_quota_interrupt
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded

    jira = MagicMock()
    exc = LLMQuotaExceeded("anthropic", "rate limit")

    with patch("agile_agent_factory.nodes.helpers.interrupt") as mock_interrupt:
        mock_interrupt.return_value = None  # simulate non-raising interrupt (test mode)
        state = {"quota_autonomous_retries": 3}
        raise_quota_interrupt(jira, "F1-1", exc, state=state, max_autonomous_retries=3)

    mock_interrupt.assert_called_once()
    call_arg = mock_interrupt.call_args[0][0]
    assert call_arg["type"] == "quota"
    assert call_arg["blocking_key"] == "F1-1"


def test_raise_quota_interrupt_always_notifies_jira():
    """_notify_quota must fire on every quota error — both autonomous and escalation paths."""
    from agile_agent_factory.nodes.helpers import raise_quota_interrupt
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded

    jira = MagicMock()
    exc = LLMQuotaExceeded("anthropic", "rate limit")

    # Autonomous path
    with patch("agile_agent_factory.nodes.helpers.interrupt"), \
         patch("agile_agent_factory.nodes.helpers.time"):
        raise_quota_interrupt(jira, "F1-1", exc, state={"quota_autonomous_retries": 0})

    jira.set_flag.assert_called_once_with("F1-1")
    jira.add_comment_adf.assert_called_once()

    jira.reset_mock()

    # Escalation path (budget exhausted)
    with patch("agile_agent_factory.nodes.helpers.interrupt") as mock_interrupt:
        mock_interrupt.return_value = None
        raise_quota_interrupt(jira, "F1-1", exc, state={"quota_autonomous_retries": 3}, max_autonomous_retries=3)

    jira.set_flag.assert_called_once_with("F1-1")
    jira.add_comment_adf.assert_called_once()


def test_raise_quota_interrupt_default_state_none():
    """raise_quota_interrupt must work when state=None (backwards-compatible default)."""
    from agile_agent_factory.nodes.helpers import raise_quota_interrupt
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded

    jira = MagicMock()
    exc = LLMQuotaExceeded("anthropic", "rate limit")

    with patch("agile_agent_factory.nodes.helpers.interrupt"), \
         patch("agile_agent_factory.nodes.helpers.time") as mock_time:
        mock_time.time.return_value = 500.0
        # Calling with state=None (default) — should treat autonomous_retries as 0
        patch_dict = raise_quota_interrupt(jira, None, exc)

    assert "quota_retry_after" in patch_dict
    assert patch_dict["quota_autonomous_retries"] == 1


# ---------------------------------------------------------------------------
# Milestone 7 — hitl_type written before HITL interrupts
# ---------------------------------------------------------------------------

def test_test_node_intervention_sets_hitl_type():
    """When test_node fires the intervention interrupt (retries exhausted), the returned
    state patch must include hitl_type == 'intervention' in the story sub-dict."""
    import importlib
    tn_mod = importlib.import_module("agile_agent_factory.nodes.test_node")

    state = {
        "active_story_key": "F1-1",
        "stories": {"F1-1": {"column": "testing", "test_contract": {}}},
        "epic_keys": [],
        "gherkin_criteria": {},
    }

    interrupt_calls = []

    def fake_interrupt(payload):
        interrupt_calls.append(payload)
        return None  # simulate resume immediately

    jira = MagicMock()
    orig_run = tn_mod.run_pytest
    orig_load = tn_mod._load_dev_context
    orig_resolve = tn_mod.resolve_dependencies
    orig_jira = tn_mod.JiraClient
    orig_correct = tn_mod._correct_code
    try:
        tn_mod.run_pytest = lambda *a, **kw: (1, "FAILED tests/test_x.py::test_y\nAssertionError")
        tn_mod._load_dev_context = lambda sk: "bp"
        tn_mod.resolve_dependencies = lambda *a, **kw: []
        tn_mod.JiraClient = lambda: jira
        tn_mod._correct_code = lambda *a, **kw: ("ok", ["app/x.py"])
        with patch("langgraph.types.interrupt", side_effect=fake_interrupt):
            result = tn_mod.test_node(state)
    finally:
        tn_mod.run_pytest = orig_run
        tn_mod._load_dev_context = orig_load
        tn_mod.resolve_dependencies = orig_resolve
        tn_mod.JiraClient = orig_jira
        tn_mod._correct_code = orig_correct

    assert len(interrupt_calls) >= 1
    assert interrupt_calls[0]["type"] == "intervention"
    assert result["stories"]["F1-1"]["hitl_type"] == "intervention"


def test_review_node_exhaustion_sets_hitl_type():
    """When review_node exhausts review retries and fires the intervention interrupt,
    the returned state patch must include hitl_type == 'intervention'."""
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
    reviewer_agent.review_patch = MagicMock(
        return_value=AgentResult(success=False, payload={"approved": False, "reason": "Still failing tests"})
    )
    try:
        with patch("agile_agent_factory.nodes.review_node.JiraClient", return_value=jira), \
             patch("langgraph.types.interrupt") as mock_interrupt:
            mock_interrupt.return_value = None
            result = review_node(state)
    finally:
        reviewer_agent.review_patch = original

    mock_interrupt.assert_called_once()
    assert result["stories"]["F1-1"]["hitl_type"] == "intervention"


def test_raise_quota_interrupt_sets_hitl_type_for_known_story_key():
    """When quota budget exhausted and blocking_key is a known story key,
    the returned patch must include stories[key]['hitl_type'] == 'quota'."""
    from agile_agent_factory.nodes.helpers import raise_quota_interrupt
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded

    jira = MagicMock()
    exc = LLMQuotaExceeded("anthropic", "rate limit")
    state = {
        "quota_autonomous_retries": 3,
        "stories": {"F1-1": {"column": "development"}},
    }

    with patch("agile_agent_factory.nodes.helpers.interrupt") as mock_interrupt:
        mock_interrupt.return_value = None
        patch_dict = raise_quota_interrupt(jira, "F1-1", exc, state=state, max_autonomous_retries=3)

    mock_interrupt.assert_called_once()
    assert "stories" in patch_dict
    assert patch_dict["stories"]["F1-1"]["hitl_type"] == "quota"


def test_raise_quota_interrupt_no_hitl_type_for_non_story_key():
    """When blocking_key is not a known story key (e.g. epic or None),
    no 'stories' patch is emitted for hitl_type."""
    from agile_agent_factory.nodes.helpers import raise_quota_interrupt
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded

    jira = MagicMock()
    exc = LLMQuotaExceeded("anthropic", "rate limit")
    # State has F1-1 as a story, but we pass an epic key F1-0 as blocking_key
    state = {
        "quota_autonomous_retries": 3,
        "stories": {"F1-1": {"column": "development"}},
    }

    with patch("agile_agent_factory.nodes.helpers.interrupt") as mock_interrupt:
        mock_interrupt.return_value = None
        patch_dict = raise_quota_interrupt(jira, "F1-0", exc, state=state, max_autonomous_retries=3)

    # No stories entry for the unknown key
    assert "stories" not in patch_dict or "F1-0" not in patch_dict.get("stories", {})


# ---------------------------------------------------------------------------
# Fix #3: quota resume resets quota_autonomous_retries (main.py _handle_resume)
# ---------------------------------------------------------------------------

def test_handle_resume_quota_resets_autonomous_retries():
    """After quota HITL is resolved, _handle_resume must reset quota_autonomous_retries to 0."""
    import sys
    import types
    import importlib

    # Build a minimal fake snapshot with quota interrupt
    fake_task = MagicMock()
    fake_task.interrupts = [MagicMock(value={"type": "quota", "blocking_key": "F1-1"})]
    fake_snapshot = MagicMock()
    fake_snapshot.tasks = [fake_task]

    update_calls = []

    fake_graph = MagicMock()
    fake_graph.update_state.side_effect = lambda cfg, patch: update_calls.append(patch)

    fake_jira = MagicMock()
    fake_jira.is_flagged.return_value = False  # quota resolved

    with (
        patch("agile_agent_factory.tools.jira_client.JiraClient", return_value=fake_jira),
        patch("langgraph.types.Command"),
    ):
        # Import main and call _handle_resume directly
        import main as main_mod
        main_mod._handle_resume(fake_graph, {"configurable": {"thread_id": "test"}}, fake_snapshot)

    # update_state must have been called with quota_autonomous_retries=0
    assert update_calls, "_handle_resume must call graph.update_state before invoking"
    first_patch = update_calls[0]
    assert first_patch.get("quota_autonomous_retries") == 0, (
        "quota resume must reset quota_autonomous_retries to 0"
    )


def test_handle_quota_backoff_preserves_retry_counter_until_real_recovery():
    """Clearing an expired/pending backoff must not zero the autonomous retry budget."""
    import main as main_mod

    fake_snapshot = MagicMock()
    fake_snapshot.values = {"quota_retry_after": 1005.0, "quota_autonomous_retries": 2}
    fake_graph = MagicMock()

    with (
        patch("main.time.time", return_value=1000.0),
        patch("main.time.sleep"),
    ):
        main_mod._handle_quota_backoff(fake_graph, {"configurable": {"thread_id": "test"}}, fake_snapshot)

    fake_graph.update_state.assert_called_once_with(
        {"configurable": {"thread_id": "test"}},
        {"quota_retry_after": None},
    )


def test_handle_resume_intervention_clears_hitl_type():
    """After intervention HITL is resolved, _handle_resume must clear hitl_type to None."""
    import main as main_mod

    fake_task = MagicMock()
    fake_task.interrupts = [MagicMock(value={"type": "intervention", "blocking_key": "F1-2"})]
    fake_snapshot = MagicMock()
    fake_snapshot.tasks = [fake_task]

    update_calls = []
    fake_graph = MagicMock()
    fake_graph.update_state.side_effect = lambda cfg, patch: update_calls.append(patch)

    fake_jira = MagicMock()
    fake_jira.is_flagged.return_value = False
    fake_jira.get_last_comment_text.return_value = "please fix this"

    with (
        patch("agile_agent_factory.tools.jira_client.JiraClient", return_value=fake_jira),
        patch("langgraph.types.Command"),
    ):
        main_mod._handle_resume(fake_graph, {"configurable": {"thread_id": "test"}}, fake_snapshot)

    assert update_calls, "_handle_resume must call graph.update_state for intervention"
    stories_patch = update_calls[0].get("stories", {})
    assert stories_patch.get("F1-2", {}).get("hitl_type") is None, (
        "intervention resume must clear hitl_type to None"
    )


def test_handle_resume_refinement_gate_routes_back_to_upstream_repair():
    """Refinement-gate HITL resume must store feedback and reopen the owning upstream lane."""
    import main as main_mod

    fake_task = MagicMock()
    fake_task.interrupts = [MagicMock(value={
        "type": "intervention",
        "blocking_key": "F1-2",
        "source": "refinement_gate",
        "errors": ["acceptance_criteria must not be empty."],
    })]
    fake_snapshot = MagicMock()
    fake_snapshot.tasks = [fake_task]
    fake_snapshot.values = {"stories": {"F1-2": {"story_key": "F1-2"}}}

    update_calls = []
    fake_graph = MagicMock()
    fake_graph.update_state.side_effect = lambda cfg, patch: update_calls.append(patch)

    fake_jira = MagicMock()
    fake_jira.is_flagged.return_value = False
    fake_jira.get_last_comment_text.return_value = "Add explicit acceptance criteria"

    with (
        patch("agile_agent_factory.tools.jira_client.JiraClient", return_value=fake_jira),
        patch("langgraph.types.Command"),
    ):
        main_mod._handle_resume(fake_graph, {"configurable": {"thread_id": "test"}}, fake_snapshot)

    stories_patch = update_calls[0]["stories"]["F1-2"]
    assert stories_patch["hitl_type"] is None
    assert stories_patch["hitl_feedback"] == "Add explicit acceptance criteria"
    assert stories_patch["refinement_qa_done"] is False
