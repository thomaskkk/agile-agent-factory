from unittest.mock import MagicMock

from agile_agent_factory.tools.jira_facade import JiraFacade
from agile_agent_factory.tools.workflow import WorkflowState


def test_advance_delegates_to_transition_to():
    jira = MagicMock()
    JiraFacade(jira).advance("PROJ-1", WorkflowState.QA)
    jira.transition_to.assert_called_once_with("PROJ-1", WorkflowState.QA)


def test_move_all_transitions_each_key():
    jira = MagicMock()
    JiraFacade(jira).move_all(["PROJ-1", "PROJ-2"], WorkflowState.DONE)
    assert jira.transition_to.call_count == 2


def test_move_all_swallows_transition_errors():
    jira = MagicMock()
    jira.transition_to.side_effect = ValueError("no transition")
    # Must not raise — mirrors the old _transition_all contract.
    JiraFacade(jira).move_all(["PROJ-1"], WorkflowState.DONE)


def test_flag_for_human_comments_then_flags():
    jira = MagicMock()
    adf = {"type": "doc"}
    JiraFacade(jira).flag_for_human("PROJ-1", adf)
    jira.add_comment_adf.assert_called_once_with("PROJ-1", adf)
    jira.set_flag.assert_called_once_with("PROJ-1")


def test_post_section_builds_heading_and_bullets():
    jira = MagicMock()
    JiraFacade(jira).post_section("PROJ-1", "Notes", ["a", "b"])
    jira.add_comment_adf.assert_called_once()
    posted_key = jira.add_comment_adf.call_args.args[0]
    assert posted_key == "PROJ-1"
