from agile_agent_factory.agents.ready_contract import readiness_repair_update, validate_ready_contract


def _valid_contract(**overrides):
    contract = {
        "story_key": "F1-1",
        "story_summary": "Create tasks",
        "full_user_intent": "Users can create tasks.",
        "in_scope_behavior": ["Scenario: Create a task"],
        "out_of_scope_behavior": ["Editing tasks is out of scope."],
        "acceptance_criteria": ["Scenario: Create a task\n  Given an empty list\n  When I add Buy milk\n  Then Buy milk is listed"],
        "examples": [{"input": "Buy milk", "expected_output": "Buy milk is listed"}],
        "edge_cases": ["Empty title is rejected."],
        "expected_tests": ["test_f1_1_create_task"],
        "target_interfaces": {"paths": ["app/tasks.py"], "imports": ["app.tasks"]},
        "has_ui": False,
        "ui_flow_reference": None,
        "open_questions": [],
    }
    contract.update(overrides)
    return contract


def test_valid_contract_passes_validation():
    assert validate_ready_contract(_valid_contract()) == []


def test_missing_acceptance_criteria_fails_validation():
    errors = validate_ready_contract(_valid_contract(acceptance_criteria=[], in_scope_behavior=[]))
    assert any("acceptance_criteria" in error for error in errors)


def test_unresolved_open_questions_fail_validation():
    errors = validate_ready_contract(_valid_contract(open_questions=["What is the persistence model?"]))
    assert any("open_questions" in error for error in errors)


def test_ui_story_without_mapped_ux_flow_fails_validation():
    errors = validate_ready_contract(_valid_contract(has_ui=True, ui_flow_reference=None))
    assert any("ui_flow_reference" in error for error in errors)


def test_non_ui_story_does_not_require_ux_fields():
    contract = _valid_contract(has_ui=False, ui_flow_reference=None)
    assert validate_ready_contract(contract) == []


def test_path_unsafe_target_interfaces_fail_validation():
    errors = validate_ready_contract(_valid_contract(target_interfaces={"paths": ["../app/tasks.py"]}))
    assert any("path-unsafe" in error for error in errors)


def test_repairable_missing_acceptance_does_not_trigger_hitl():
    update = readiness_repair_update([
        "acceptance_criteria must not be empty.",
        "open_questions must be empty before tech_design.",
    ])
    assert update["refinement_qa_done"] is False
    assert "hitl_type" not in update


def test_missing_story_summary_routes_to_qa():
    update = readiness_repair_update(["story_summary is required."])
    assert update.get("refinement_qa_done") is False
    assert "hitl_type" not in update


def test_missing_user_intent_routes_to_qa():
    update = readiness_repair_update(["full_user_intent is required."])
    assert update.get("refinement_qa_done") is False
    assert "hitl_type" not in update


def test_missing_in_scope_behavior_routes_to_qa():
    update = readiness_repair_update(["in_scope_behavior must include at least one explicit behavior."])
    assert update.get("refinement_qa_done") is False
    assert "hitl_type" not in update


def test_missing_ui_flow_routes_to_ux():
    update = readiness_repair_update(["UI stories require a mapped ui_flow_reference."])
    assert update.get("refinement_ux_done") is False
    assert "hitl_type" not in update


def test_unsafe_target_interfaces_triggers_hitl():
    update = readiness_repair_update([
        "target_interfaces.paths contains path-unsafe value: ../app/tasks.py",
    ])
    assert update.get("hitl_type") == "refinement"


def test_open_questions_routes_to_qa_when_not_ui():
    update = readiness_repair_update(["open_questions must be empty before tech_design."])
    # open_questions without UI/flow content → QA can fix by regenerating
    assert update.get("refinement_qa_done") is False
    assert "hitl_type" not in update


from agile_agent_factory.agents.ready_contract import build_ready_contract


def _sample_tc():
    return {
        "test_file": "tests/test_auth.py",
        "test_functions": ["test_login_valid", "test_login_invalid_password"],
        "target_imports": ["from app.auth import login_user"],
        "fixtures": [{"name": "registered_user", "description": "A user already in the system"}],
        "sample_data": [{"username": "alice", "password": "correct_horse"}],
        "edge_cases": ["empty password string", "non-existent username"],
    }


def test_test_contract_overrides_expected_tests():
    """When test_contract is provided, expected_tests must come from test_functions, not Gherkin slugs."""
    contract = build_ready_contract(
        story_key="F1-1",
        story={},
        summary="User can log in",
        business_idea="Auth system",
        acceptance_criteria=["Scenario: Login success\n  Given valid creds\n  When login\n  Then success"],
        test_contract=_sample_tc(),
    )
    assert contract["expected_tests"] == ["test_login_valid", "test_login_invalid_password"]
    assert not any("f1_1" in t for t in contract["expected_tests"]), "Slug-based names must not appear when test_contract is provided"


def test_test_contract_overrides_target_interfaces():
    """When test_contract is provided, target_interfaces must come from target_imports + test_file."""
    contract = build_ready_contract(
        story_key="F1-1",
        story={},
        summary="User can log in",
        business_idea="Auth system",
        acceptance_criteria=["Scenario: Login\n  Given x\n  When y\n  Then z"],
        test_contract=_sample_tc(),
    )
    assert "from app.auth import login_user" in contract["target_interfaces"]["imports"]
    assert "tests/test_auth.py" in contract["target_interfaces"]["paths"]


def test_test_contract_overrides_edge_cases():
    """When test_contract.edge_cases is non-empty, those replace the Gherkin-extracted edge cases."""
    contract = build_ready_contract(
        story_key="F1-1",
        story={},
        summary="User can log in",
        business_idea="Auth system",
        acceptance_criteria=["Scenario: Login\n  Given x\n  When y\n  Then z"],
        test_contract=_sample_tc(),
    )
    assert "empty password string" in contract["edge_cases"]
    assert "non-existent username" in contract["edge_cases"]


def test_test_contract_stored_verbatim_in_contract():
    """The raw test_contract dict must be stored on the ready_contract for downstream consumers."""
    tc = _sample_tc()
    contract = build_ready_contract(
        story_key="F1-1",
        story={},
        summary="User can log in",
        business_idea="Auth system",
        acceptance_criteria=["Scenario: Login\n  Given x\n  When y\n  Then z"],
        test_contract=tc,
    )
    assert contract["test_contract"] == tc


def test_build_ready_contract_without_test_contract_uses_slug_expected_tests():
    """Absent test_contract, expected_tests must still be generated from Gherkin slugs (backward compat)."""
    contract = build_ready_contract(
        story_key="F1-1",
        story={},
        summary="User can log in",
        business_idea="Auth system",
        acceptance_criteria=["Scenario: Login success\n  Given valid creds\n  When login\n  Then success"],
    )
    assert any("f1_1" in t for t in contract["expected_tests"])
    assert contract["test_contract"] == {}
