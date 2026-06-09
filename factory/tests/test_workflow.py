from agile_agent_factory.tools.workflow import WorkflowState


def test_workflow_states_have_expected_values():
    assert WorkflowState.BUSINESS_REFINEMENT.value == "Business Refinement"
    assert WorkflowState.TECH_REFINEMENT.value == "Tech Refinement"
    assert WorkflowState.IN_CODE_REVIEW.value == "In Code Review"
    assert WorkflowState.QA.value == "QA"
    assert WorkflowState.DONE.value == "Done"


def test_all_states_are_unique():
    values = [s.value for s in WorkflowState]
    assert len(values) == len(set(values))
