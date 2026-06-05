from unittest.mock import patch

import pytest


def test_analyze_and_provision_skips_when_story_keys_already_exist(monkeypatch, tmp_path, mock_jira):
    """Idempotency: if story_keys are already in state, no Jira issues should be created."""
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    existing_state = {
        "status": "READY",
        "current_phase": "upstream_po_done",
        "story_keys": ["TEST-10", "TEST-11"],
        "epic_keys": ["TEST-9"],
        "blocking_issue_key": None,
        "subtasks": {},
        "review_retries": 0,
    }

    jira = mock_jira

    result = analyze_and_provision(jira, existing_state)

    jira.create_issue.assert_not_called()
    assert result["story_keys"] == ["TEST-10", "TEST-11"]
    assert result["epic_keys"] == ["TEST-9"]


def test_analyze_and_provision_creates_issues_when_story_keys_empty(tmp_path, monkeypatch, mock_jira):
    """When story_keys is empty, issues should be created as normal."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a todo app.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)

    llm_response = {
        "has_ambiguity": False,
        "ambiguity_description": "",
        "has_ui": True,
        "epics": [
            {
                "title": "Core App",
                "description": "The main epic",
                "stories": [
                    {
                        "title": "User can add tasks",
                        "description": "As a user, I want to add tasks.",
                        "definition_of_done": ["Task is saved"],
                    }
                ],
            }
        ],
    }

    jira = mock_jira
    jira.create_issue.side_effect = [
        {"key": "TEST-1"},  # epic
        {"key": "TEST-2"},  # story
    ]

    empty_state = {
        "status": "READY",
        "current_phase": None,
        "story_keys": [],
        "epic_keys": [],
        "blocking_issue_key": None,
        "subtasks": {},
        "review_retries": 0,
    }

    with patch("agile_agent_factory.agents.po_agent.call_llm_json", return_value=llm_response):
        result = analyze_and_provision(jira, empty_state)

    assert jira.create_issue.call_count == 2
    assert result["story_keys"] == ["TEST-2"]


def test_analyze_and_provision_includes_hitl_feedback_in_prompt(tmp_path, monkeypatch, mock_jira):
    """When hitl_feedback is in state, it must appear in the LLM prompt."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a todo app.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)

    captured_prompt = {}

    def fake_llm_json(prompt, **kwargs):
        captured_prompt["value"] = prompt
        return {"has_ambiguity": False, "ambiguity_description": "", "epics": []}

    jira = mock_jira
    state_with_feedback = {
        "status": "READY",
        "current_phase": None,
        "story_keys": [],
        "epic_keys": [],
        "blocking_issue_key": None,
        "hitl_feedback": "The app only needs to run on Linux. No Windows support required.",
        "subtasks": {},
        "review_retries": 0,
    }

    with patch("agile_agent_factory.agents.po_agent.call_llm_json", side_effect=fake_llm_json):
        analyze_and_provision(jira, state_with_feedback)

    assert "only needs to run on Linux" in captured_prompt["value"]


def test_writes_business_intent(tmp_path, monkeypatch, mock_jira):
    """After analyze_and_provision, BP_BUSINESS_INTENT must exist and contain idea text."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a kanban board app.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)
    monkeypatch.setattr("agile_agent_factory.agents.po_agent.BP_BUSINESS_INTENT", tmp_path / "business_intent.md")

    llm_response = {
        "has_ambiguity": False,
        "ambiguity_description": "",
        "has_ui": True,
        "epics": [
            {
                "title": "Board",
                "description": "Main epic",
                "stories": [{"title": "Add card", "description": "As a user...", "definition_of_done": ["card saved"]}],
            }
        ],
    }

    jira = mock_jira
    jira.create_issue.side_effect = [{"key": "KB-1"}, {"key": "KB-2"}]

    with patch("agile_agent_factory.agents.po_agent.call_llm_json", return_value=llm_response):
        analyze_and_provision(jira, {"status": "READY", "current_phase": None, "story_keys": [], "epic_keys": [], "blocking_issue_key": None, "subtasks": {}, "review_retries": 0})

    intent_file = tmp_path / "business_intent.md"
    assert intent_file.exists(), "BP_BUSINESS_INTENT must be written after provisioning"
    content = intent_file.read_text()
    assert "kanban board app" in content
