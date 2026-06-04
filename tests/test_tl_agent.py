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


def test_writes_architecture_files(tmp_path, monkeypatch):
    """design_architecture must write both decisions.md and constraints.md."""
    import agile_agent_factory.agents.tl_agent as tl
    monkeypatch.setattr(tl, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    monkeypatch.setattr(tl, "BLUEPRINT_PATH", tmp_path / "handoff_blueprint.md")
    monkeypatch.setattr("agile_agent_factory.agents.tl_agent.BP_ARCH_DECISIONS", tmp_path / "decisions.md")
    monkeypatch.setattr("agile_agent_factory.agents.tl_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md")
    monkeypatch.setattr("agile_agent_factory.agents.tl_agent.bp_task_path", lambda sk: tmp_path / f"{sk}.md")
    (tmp_path / "business_idea.md").write_text("Build a task manager.")

    arch = {
        "files": [{"path": "app/main.py", "purpose": "Entry point", "functions": ["main()"]}],
        "subtasks": [],
        "import_rules": "from app.main import main",
        "test_command": "uv run pytest ../tests/ -v",
        "dependencies": ["requests"],
    }
    state = _state("upstream_ux_done")

    jira = MagicMock()
    jira.get_subtask_issue_type.return_value = "Subtask"

    with patch.object(tl, "call_llm_json", return_value=arch):
        tl.design_architecture(jira, ["F1-1"], {}, {}, state)

    decisions = tmp_path / "decisions.md"
    constraints = tmp_path / "constraints.md"
    assert decisions.exists(), "decisions.md must be written"
    assert constraints.exists(), "constraints.md must be written"
    assert "app/main.py" in decisions.read_text()
    assert "requests" in decisions.read_text()
    assert "pytest" in constraints.read_text()


def test_writes_task_file_per_story(tmp_path, monkeypatch):
    """design_architecture must write one task file per story in BP_TASKS_DIR."""
    import agile_agent_factory.agents.tl_agent as tl
    monkeypatch.setattr(tl, "BUSINESS_IDEA_PATH", tmp_path / "business_idea.md")
    monkeypatch.setattr(tl, "BLUEPRINT_PATH", tmp_path / "handoff_blueprint.md")
    monkeypatch.setattr("agile_agent_factory.agents.tl_agent.BP_ARCH_DECISIONS", tmp_path / "decisions.md")
    monkeypatch.setattr("agile_agent_factory.agents.tl_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md")

    tasks_dir = tmp_path / "tasks"
    monkeypatch.setattr(
        "agile_agent_factory.agents.tl_agent.bp_task_path",
        lambda sk: tasks_dir / f"{sk}.md",
    )
    (tmp_path / "business_idea.md").write_text("Build a task manager.")

    arch = {
        "files": [{"path": "app/main.py", "purpose": "Entry point", "functions": []}],
        "subtasks": [],
        "import_rules": "from app.main import main",
        "test_command": "uv run pytest ../tests/ -v",
        "dependencies": [],
    }
    gherkin = {
        "F1-1": ["Scenario: Create task\n  Given empty list\n  When I add\n  Then count is 1"],
        "F1-2": ["Scenario: Delete task\n  Given one task\n  When I delete\n  Then count is 0"],
    }
    state = _state("upstream_ux_done", story_keys=["F1-1", "F1-2"])

    jira = MagicMock()
    jira.get_subtask_issue_type.return_value = "Subtask"

    with patch.object(tl, "call_llm_json", return_value=arch):
        tl.design_architecture(jira, ["F1-1", "F1-2"], gherkin, {}, state)

    assert (tasks_dir / "F1-1.md").exists(), "Task file must be written for F1-1"
    assert (tasks_dir / "F1-2.md").exists(), "Task file must be written for F1-2"
    assert "Create task" in (tasks_dir / "F1-1.md").read_text()
    assert "Delete task" in (tasks_dir / "F1-2.md").read_text()
