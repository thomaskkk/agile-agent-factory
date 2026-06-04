from unittest.mock import MagicMock, patch

import pytest


def _state(phase, **extra):
    base = {
        "status": "READY",
        "current_phase": phase,
        "story_keys": ["F1-1"],
        "epic_keys": ["F1-2"],
        "blocking_issue_key": None,
        "hitl_feedback": "",
        "subtasks": {},
        "review_retries": 0,
        "has_ui": False,
        "dependencies": [],
        "gherkin_criteria": {},
        "ux_spec": {},
        "architecture": {},
    }
    base.update(extra)
    return base


def test_reuses_stored_architecture_on_resume(tmp_path, monkeypatch):
    """When phase is upstream_arch_done with a stored architecture, no LLM call is made."""
    import agile_agent_factory.agents.tl_agent as tl
    monkeypatch.setattr(tl, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    monkeypatch.setattr(tl, "BLUEPRINT_PATH", tmp_path / "handoff_blueprint.md")
    (tmp_path / "business_idea.md").write_text("Build something.")

    stored_arch = {
        "files": [{"path": "app/x.py", "purpose": "p", "functions": []}],
        "subtasks": [{"title": "Task A", "story_key": "F1-1", "description": "d"}],
        "import_rules": "...",
        "test_command": "uv run pytest ../tests/ -v",
        "dependencies": [],
    }
    state = _state("upstream_arch_done", architecture=stored_arch)

    jira = MagicMock()
    jira.get_subtask_issue_type.return_value = "Subtask"
    jira.create_issue.return_value = {"key": "F1-3"}

    with patch.object(tl, "call_llm_json") as mock_llm:
        tl.design_architecture(jira, ["F1-1"], {}, {}, state)

    mock_llm.assert_not_called()
    jira.create_issue.assert_called_once()  # the one subtask from stored arch


def test_skips_already_created_subtasks(tmp_path, monkeypatch):
    """Subtasks already present in state['subtasks'] must not be recreated."""
    import agile_agent_factory.agents.tl_agent as tl
    monkeypatch.setattr(tl, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    monkeypatch.setattr(tl, "BLUEPRINT_PATH", tmp_path / "handoff_blueprint.md")
    (tmp_path / "business_idea.md").write_text("Build something.")

    stored_arch = {
        "files": [],
        "subtasks": [
            {"title": "Task A", "story_key": "F1-1", "description": "d"},
            {"title": "Task B", "story_key": "F1-1", "description": "d"},
        ],
        "import_rules": "...",
        "test_command": "uv run pytest ../tests/ -v",
        "dependencies": [],
    }
    # Task A already created in a prior (interrupted) run
    state = _state("upstream_arch_done", architecture=stored_arch, subtasks={"Task A": "F1-9"})

    jira = MagicMock()
    jira.get_subtask_issue_type.return_value = "Subtask"
    jira.create_issue.return_value = {"key": "F1-10"}

    with patch.object(tl, "call_llm_json"):
        result = tl.design_architecture(jira, ["F1-1"], {}, {}, state)

    # Only Task B is created; Task A is skipped
    jira.create_issue.assert_called_once()
    created_title = jira.create_issue.call_args.args[0]
    assert created_title == "Task B"
    assert result["subtasks"] == {"Task A": "F1-9", "Task B": "F1-10"}


def test_fresh_run_calls_llm_and_persists_architecture(tmp_path, monkeypatch):
    """A fresh TL run calls the LLM and returns architecture in the result dict."""
    import agile_agent_factory.agents.tl_agent as tl
    monkeypatch.setattr(tl, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    monkeypatch.setattr(tl, "BLUEPRINT_PATH", tmp_path / "handoff_blueprint.md")
    (tmp_path / "business_idea.md").write_text("Build something.")

    arch = {
        "files": [],
        "subtasks": [],
        "import_rules": "...",
        "test_command": "uv run pytest ../tests/ -v",
        "dependencies": ["flask"],
    }
    state = _state("upstream_ux_done")

    jira = MagicMock()
    jira.get_subtask_issue_type.return_value = "Subtask"

    with patch.object(tl, "call_llm_json", return_value=arch) as mock_llm:
        result = tl.design_architecture(jira, ["F1-1"], {}, {}, state)

    mock_llm.assert_called_once()
    assert result["current_phase"] == "upstream_tl_done"
    assert result["dependencies"] == ["flask"]
    # architecture is returned in the result dict (LangGraph checkpointer persists it)
    assert result["architecture"] == arch
