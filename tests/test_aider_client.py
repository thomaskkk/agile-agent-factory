import pytest


@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    import agile_agent_factory.tools.aider_client as aider_client
    monkeypatch.setattr(aider_client, "AIDER_ENABLED", False)
    monkeypatch.setattr(aider_client, "AIDER_MODEL", "anthropic/claude-sonnet-4-6")
    monkeypatch.setattr(aider_client, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(aider_client, "OPENAI_API_KEY", "")
    monkeypatch.setattr(aider_client, "LLM_TIMEOUT_SECONDS", 60)


def test_is_available_returns_false_when_disabled(mocker):
    import agile_agent_factory.tools.aider_client as aider_client
    mocker.patch.object(aider_client, "AIDER_ENABLED", False)
    mocker.patch("agile_agent_factory.tools.aider_client.shutil.which", return_value="/usr/bin/aider")
    assert aider_client.is_available() is False


def test_is_available_returns_false_when_binary_missing(mocker):
    import agile_agent_factory.tools.aider_client as aider_client
    mocker.patch.object(aider_client, "AIDER_ENABLED", True)
    mocker.patch("agile_agent_factory.tools.aider_client.shutil.which", return_value=None)
    assert aider_client.is_available() is False


def test_is_available_returns_true_when_enabled_and_binary_present(mocker):
    import agile_agent_factory.tools.aider_client as aider_client
    mocker.patch.object(aider_client, "AIDER_ENABLED", True)
    mocker.patch("agile_agent_factory.tools.aider_client.shutil.which", return_value="/usr/local/bin/aider")
    assert aider_client.is_available() is True


def test_run_task_calls_subprocess_with_correct_args(mocker):
    import agile_agent_factory.tools.aider_client as aider_client

    mocker.patch.object(aider_client, "AIDER_MODEL", "anthropic/claude-sonnet-4-6")
    mocker.patch.object(aider_client, "ANTHROPIC_API_KEY", "test-key")
    mocker.patch.object(aider_client, "LLM_TIMEOUT_SECONDS", 60)

    mock_run = mocker.patch("agile_agent_factory.tools.aider_client.subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "aider output"
    mock_run.return_value.stderr = ""

    result = aider_client.run_task("Implement the blueprint.", "# Blueprint content")

    assert result["success"] is True
    assert result["output"] == "aider output"

    call_args = mock_run.call_args
    cmd = call_args[0][0]
    assert cmd[0] == "aider"
    assert "--model" in cmd
    assert "anthropic/claude-sonnet-4-6" in cmd
    assert "--yes" in cmd
    assert "--no-git" in cmd
    assert "--no-auto-commits" in cmd
    assert call_args.kwargs["cwd"] == str(aider_client.PRODUCT_ROOT)


def test_run_task_returns_failure_on_nonzero_exit(mocker):
    import agile_agent_factory.tools.aider_client as aider_client

    mocker.patch.object(aider_client, "LLM_TIMEOUT_SECONDS", 60)

    mock_run = mocker.patch("agile_agent_factory.tools.aider_client.subprocess.run")
    mock_run.return_value.returncode = 1
    mock_run.return_value.stdout = ""
    mock_run.return_value.stderr = "Error: model not found"

    result = aider_client.run_task("Do something.", "")

    assert result["success"] is False
    assert "Error: model not found" in result["output"]
