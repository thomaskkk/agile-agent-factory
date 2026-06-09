"""Tests for pipeline.py lifecycle nodes — focused on refinement_gate_node."""

from unittest.mock import MagicMock, patch


def _make_state(story_key: str = "F1-1", story_extra: dict | None = None) -> dict:
    story = {
        "story_key": story_key,
        "column": "refinement",
        "refinement_qa_done": True,
        "refinement_ux_done": True,
        "gherkin_criteria": ["Scenario: something"],
        "test_contract": {"test_file": "tests/test_x.py", "test_functions": ["test_x"]},
    }
    if story_extra:
        story.update(story_extra)
    return {
        "active_story_key": story_key,
        "stories": {story_key: story},
        "business_idea": "build a product",
        "epic_keys": [],
        "gherkin_criteria": {story_key: ["Scenario: something"]},
        "ux_spec": {},
    }


def _mock_jira():
    j = MagicMock()
    j.get_issue.return_value = {"fields": {"summary": "Do something"}}
    return j


# ---------------------------------------------------------------------------
# refinement_gate_node: retry counter + HITL cap (Fix #4)
# ---------------------------------------------------------------------------

def test_refinement_gate_increments_retry_on_failure():
    """Gate failure must increment refinement_retries and keep story in refinement."""
    from agile_agent_factory.nodes.pipeline import refinement_gate_node

    state = _make_state(story_extra={"refinement_retries": 0})

    with (
        patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=_mock_jira()),
        patch("agile_agent_factory.nodes.pipeline._story_summary", return_value="Do something"),
        patch("agile_agent_factory.agents.ready_contract.build_ready_contract", return_value={}),
        patch("agile_agent_factory.agents.ready_contract.validate_ready_contract", return_value=["missing field"]),
        patch("agile_agent_factory.agents.ready_contract.readiness_repair_update", return_value={}),
    ):
        result = refinement_gate_node(state)

    story_patch = result["stories"]["F1-1"]
    assert story_patch["refinement_retries"] == 1, "retry counter must increment on gate failure"
    # column should NOT have been set to tech_design
    assert story_patch.get("column") != "tech_design"


def test_refinement_gate_triggers_hitl_at_max_retries():
    """Gate must fire interrupt when refinement_retries reaches MAX_REFINEMENT_RETRIES."""
    from agile_agent_factory.nodes.pipeline import refinement_gate_node
    from agile_agent_factory.config import MAX_REFINEMENT_RETRIES

    # Start at max - 1 so next failure hits the cap
    state = _make_state(story_extra={"refinement_retries": MAX_REFINEMENT_RETRIES - 1})

    with (
        patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=_mock_jira()),
        patch("agile_agent_factory.nodes.pipeline._story_summary", return_value="Do something"),
        patch("agile_agent_factory.agents.ready_contract.build_ready_contract", return_value={}),
        patch("agile_agent_factory.agents.ready_contract.validate_ready_contract", return_value=["missing field"]),
        patch("agile_agent_factory.agents.ready_contract.readiness_repair_update", return_value={}),
        patch("langgraph.types.interrupt") as mock_interrupt,
    ):
        mock_interrupt.return_value = None
        result = refinement_gate_node(state)

    mock_interrupt.assert_called_once()
    interrupt_payload = mock_interrupt.call_args[0][0]
    assert interrupt_payload["type"] == "intervention"
    assert interrupt_payload["blocking_key"] == "F1-1"


def test_refinement_gate_hitl_does_not_clear_flag():
    """refinement_gate HITL must leave the Jira flag raised for human visibility and resume guards."""
    from agile_agent_factory.nodes.pipeline import refinement_gate_node
    from agile_agent_factory.config import MAX_REFINEMENT_RETRIES

    state = _make_state(story_extra={"refinement_retries": MAX_REFINEMENT_RETRIES - 1})
    jira = _mock_jira()

    with (
        patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=jira),
        patch("agile_agent_factory.nodes.pipeline._story_summary", return_value="Do something"),
        patch("agile_agent_factory.agents.ready_contract.build_ready_contract", return_value={}),
        patch("agile_agent_factory.agents.ready_contract.validate_ready_contract", return_value=["missing field"]),
        patch("agile_agent_factory.agents.ready_contract.readiness_repair_update", return_value={}),
        patch("langgraph.types.interrupt"),
    ):
        refinement_gate_node(state)

    jira.set_flag.assert_called_once()
    jira.clear_flag.assert_not_called()


def test_refinement_gate_resets_retries_on_success():
    """Successful gate validation must set refinement_retries=0 in the returned state."""
    from agile_agent_factory.nodes.pipeline import refinement_gate_node

    state = _make_state(story_extra={"refinement_retries": 2})

    with (
        patch("agile_agent_factory.nodes.pipeline.JiraClient", return_value=_mock_jira()),
        patch("agile_agent_factory.nodes.pipeline._story_summary", return_value="Do something"),
        patch("agile_agent_factory.agents.ready_contract.build_ready_contract", return_value={"expected_tests": ["test_x"]}),
        patch("agile_agent_factory.agents.ready_contract.validate_ready_contract", return_value=[]),
        patch("agile_agent_factory.agents.ready_contract.readiness_repair_update", return_value={}),
    ):
        result = refinement_gate_node(state)

    story_patch = result["stories"]["F1-1"]
    assert story_patch["column"] == "tech_design", "successful gate must advance to tech_design"
    assert story_patch["refinement_retries"] == 0, "success must reset retry counter to 0"
