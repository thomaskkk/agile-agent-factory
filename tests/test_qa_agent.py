from unittest.mock import MagicMock, patch


def test_writes_per_story_criteria_file(tmp_path, monkeypatch):
    """inject_gherkin_criteria must write one QA criteria file per story."""
    import agile_agent_factory.agents.qa_agent as qa_agent
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    criteria_path = tmp_path / "F1-1.md"
    monkeypatch.setattr(
        "agile_agent_factory.agents.qa_agent.bp_qa_criteria_path",
        lambda sk: tmp_path / f"{sk}.md",
    )

    jira = MagicMock()
    jira._request.return_value = {
        "fields": {
            "summary": "User can log in",
            "description": {"content": []},
        }
    }

    llm_response = {
        "acceptance_criteria": [
            "Scenario: Login success\n  Given valid credentials\n  When user logs in\n  Then access granted"
        ]
    }

    with patch("agile_agent_factory.agents.qa_agent.call_llm_json", return_value=llm_response):
        inject_gherkin_criteria(jira, ["F1-1"])

    assert criteria_path.exists(), "QA criteria file must be written for story F1-1"
    content = criteria_path.read_text()
    assert "F1-1" in content
    assert "Login success" in content


def test_writes_separate_files_for_multiple_stories(tmp_path, monkeypatch):
    """Two stories must produce two separate QA criteria files."""
    import agile_agent_factory.agents.qa_agent as qa_agent
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    monkeypatch.setattr(
        "agile_agent_factory.agents.qa_agent.bp_qa_criteria_path",
        lambda sk: tmp_path / f"{sk}.md",
    )

    jira = MagicMock()
    jira._request.return_value = {
        "fields": {
            "summary": "A story",
            "description": {"content": []},
        }
    }

    llm_response = {"acceptance_criteria": ["Scenario: X\n  Given x\n  When y\n  Then z"]}

    with patch("agile_agent_factory.agents.qa_agent.call_llm_json", return_value=llm_response):
        inject_gherkin_criteria(jira, ["F1-1", "F1-2"])

    assert (tmp_path / "F1-1.md").exists()
    assert (tmp_path / "F1-2.md").exists()
