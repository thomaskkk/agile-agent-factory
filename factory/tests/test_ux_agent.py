from unittest.mock import patch

import pytest


def test_design_user_experience_returns_validated_spec(tmp_path, monkeypatch, mock_jira):
    """UX agent returns a validated spec dict with bounded ui_type and technology."""
    import agile_agent_factory.agents.ux_agent as ux_agent

    monkeypatch.setattr(ux_agent, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    (tmp_path / "business_idea.md").write_text("Build a CLI expense tracker.")

    jira = mock_jira
    jira._request.return_value = {"fields": {"summary": "Log expenses", "description": {}}}

    llm_response = {
        "ui_type": "cli",
        "technology": "argparse",
        "description": "Interactive CLI for expense logging",
        "screens_or_flows": [{"name": "log", "purpose": "Add expense", "key_elements": ["amount"], "story_key": "TEST-1"}],
        "state_management": "In-memory dict",
        "design_decisions": ["Use subcommands", "Tabular output"],
    }

    with patch("agile_agent_factory.tools.llm_adapters.ux.call_llm_json", return_value=llm_response):
        result = ux_agent.design_user_experience(jira, ["TEST-1"], {}, {})

    spec = result.payload["ux_spec"]
    assert spec["ui_type"] == "cli"
    assert spec["technology"] == "argparse"
    assert len(spec["screens_or_flows"]) == 1


def test_design_user_experience_appends_to_jira_description(tmp_path, monkeypatch, mock_jira):
    """UX agent must call update_issue_description for each story when ui_type != none."""
    import agile_agent_factory.agents.ux_agent as ux_agent

    monkeypatch.setattr(ux_agent, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    (tmp_path / "business_idea.md").write_text("Build a web app.")

    jira = mock_jira
    jira._request.return_value = {"fields": {"summary": "View dashboard", "description": {}}}

    llm_response = {
        "ui_type": "web",
        "technology": "Flask",
        "description": "Web dashboard",
        "screens_or_flows": [],
        "state_management": "Session-based",
        "design_decisions": ["Use Jinja2 templates"],
    }

    with patch("agile_agent_factory.tools.llm_adapters.ux.call_llm_json", return_value=llm_response):
        ux_agent.design_user_experience(jira, ["TEST-2", "TEST-3"], {}, {})

    assert jira.update_issue_description.call_count == 2


def test_design_user_experience_includes_hitl_feedback_in_story_context(tmp_path, monkeypatch, mock_jira):
    """Human clarification should be visible to the UX generator on regeneration."""
    import agile_agent_factory.agents.ux_agent as ux_agent

    monkeypatch.setattr(ux_agent, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    (tmp_path / "business_idea.md").write_text("Build a web app.")

    jira = mock_jira
    jira._request.return_value = {"fields": {"summary": "Landing page", "description": {}}}
    captured = {}

    def fake_generate(business_idea, stories_block, fallback):
        captured["stories_block"] = stories_block
        return {
            "ui_type": "web",
            "technology": "Flask",
            "description": "desc",
            "screens_or_flows": [],
            "state_management": "",
            "design_decisions": [],
        }

    with patch("agile_agent_factory.agents.ux_agent.generate_ux_spec", side_effect=fake_generate):
        ux_agent.design_user_experience(
            jira,
            ["TEST-1"],
            {},
            {},
            story_feedback={"TEST-1": "Use a single-step sign-in flow."},
        )

    assert "single-step sign-in flow" in captured["stories_block"]


def test_design_user_experience_quota_propagates(tmp_path, monkeypatch, mock_jira):
    """LLMQuotaExceeded raised inside design_user_experience must not be caught."""
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded
    import agile_agent_factory.agents.ux_agent as ux_agent

    monkeypatch.setattr(ux_agent, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    (tmp_path / "business_idea.md").write_text("Build something.")

    jira = mock_jira
    jira._request.return_value = {"fields": {"summary": "Story", "description": {}}}

    with patch("agile_agent_factory.tools.llm_adapters.ux.call_llm_json", side_effect=LLMQuotaExceeded("anthropic", "quota")):
        with pytest.raises(LLMQuotaExceeded):
            ux_agent.design_user_experience(jira, ["TEST-1"], {}, {})


def test_validate_ux_spec_rejects_unknown_ui_type():
    """Unknown ui_type must return the fallback spec."""
    from agile_agent_factory.agents.ux_agent import _validate_ux_spec, _UX_FALLBACK

    result = _validate_ux_spec({"ui_type": "hologram", "technology": "argparse"})
    assert result == _UX_FALLBACK


def test_validate_ux_spec_rejects_unknown_technology():
    """Unknown technology must return the fallback spec."""
    from agile_agent_factory.agents.ux_agent import _validate_ux_spec, _UX_FALLBACK

    result = _validate_ux_spec({"ui_type": "cli", "technology": "pyside6"})
    assert result == _UX_FALLBACK


def test_writes_ux_decisions_file(tmp_path, monkeypatch, mock_jira):
    """design_user_experience must write BP_UX_DECISIONS with technology in content."""
    import agile_agent_factory.agents.ux_agent as ux_agent

    monkeypatch.setattr(ux_agent, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    (tmp_path / "business_idea.md").write_text("Build a CLI expense tracker.")
    monkeypatch.setattr("agile_agent_factory.agents.ux_agent.BP_UX_DECISIONS", tmp_path / "ux_decisions.md")

    jira = mock_jira
    jira._request.return_value = {"fields": {"summary": "Log expenses", "description": {}}}

    llm_response = {
        "ui_type": "cli",
        "technology": "argparse",
        "description": "Interactive CLI for expense logging",
        "screens_or_flows": [{"name": "log", "purpose": "Add expense", "key_elements": ["amount"], "story_key": "TEST-1"}],
        "state_management": "In-memory dict",
        "design_decisions": ["Use subcommands"],
    }

    with patch("agile_agent_factory.tools.llm_adapters.ux.call_llm_json", return_value=llm_response):
        ux_agent.design_user_experience(jira, ["TEST-1"], {}, {})

    ux_file = tmp_path / "ux_decisions.md"
    assert ux_file.exists(), "BP_UX_DECISIONS must be written"
    content = ux_file.read_text()
    assert "argparse" in content


def test_writes_ux_file_when_no_ui(tmp_path, monkeypatch, mock_jira):
    """A minimal BP_UX_DECISIONS file must be written even when ui_type == 'none'."""
    import agile_agent_factory.agents.ux_agent as ux_agent

    monkeypatch.setattr(ux_agent, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    (tmp_path / "business_idea.md").write_text("Build a pure library.")
    monkeypatch.setattr("agile_agent_factory.agents.ux_agent.BP_UX_DECISIONS", tmp_path / "ux_decisions.md")

    jira = mock_jira
    jira._request.return_value = {"fields": {"summary": "Library core", "description": {}}}

    llm_response = {
        "ui_type": "none",
        "technology": "none",
        "description": "",
        "screens_or_flows": [],
        "state_management": "",
        "design_decisions": [],
    }

    with patch("agile_agent_factory.tools.llm_adapters.ux.call_llm_json", return_value=llm_response):
        ux_agent.design_user_experience(jira, ["TEST-1"], {}, {})

    ux_file = tmp_path / "ux_decisions.md"
    assert ux_file.exists(), "BP_UX_DECISIONS must be written even when ui_type is none"
    assert "No user interface required" in ux_file.read_text()
