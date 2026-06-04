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
