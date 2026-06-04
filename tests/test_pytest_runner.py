import pytest


def test_run_pytest_injects_product_root_into_pythonpath(mocker):
    mock_run = mocker.patch("agile_agent_factory.tools.pytest_runner.subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "1 passed"
    mock_run.return_value.stderr = ""

    import agile_agent_factory.tools.pytest_runner as pytest_runner
    exit_code, output = pytest_runner.run_pytest()

    assert exit_code == 0
    call_env = mock_run.call_args.kwargs["env"]
    assert str(pytest_runner.PRODUCT_ROOT) in call_env["PYTHONPATH"]


def test_run_pytest_returns_failure_code_and_output(mocker):
    mock_run = mocker.patch("agile_agent_factory.tools.pytest_runner.subprocess.run")
    mock_run.return_value.returncode = 1
    mock_run.return_value.stdout = "FAILED test_x.py::test_y"
    mock_run.return_value.stderr = ""

    import agile_agent_factory.tools.pytest_runner as pytest_runner
    exit_code, output = pytest_runner.run_pytest()

    assert exit_code == 1
    assert "FAILED" in output


def test_run_pytest_preserves_existing_pythonpath(mocker):
    mock_run = mocker.patch("agile_agent_factory.tools.pytest_runner.subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = ""
    mock_run.return_value.stderr = ""

    import os
    import agile_agent_factory.tools.pytest_runner as pytest_runner
    original_env = os.environ.copy()
    original_env["PYTHONPATH"] = "/some/existing/path"
    mocker.patch("agile_agent_factory.tools.pytest_runner.os.environ", original_env)

    pytest_runner.run_pytest()

    call_env = mock_run.call_args.kwargs["env"]
    assert "/some/existing/path" in call_env["PYTHONPATH"]
    assert str(pytest_runner.PRODUCT_ROOT) in call_env["PYTHONPATH"]


def test_run_pytest_adds_extra_packages_as_with_flags(mocker):
    mock_run = mocker.patch("agile_agent_factory.tools.pytest_runner.subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = ""
    mock_run.return_value.stderr = ""

    import agile_agent_factory.tools.pytest_runner as pytest_runner
    pytest_runner.run_pytest(["flask", "requests"])

    cmd = mock_run.call_args.args[0]
    assert "--with=flask" in cmd
    assert "--with=requests" in cmd
    assert cmd.index("--with=flask") < cmd.index("pytest")


def test_run_pytest_no_extra_packages_has_no_with_flags(mocker):
    mock_run = mocker.patch("agile_agent_factory.tools.pytest_runner.subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = ""
    mock_run.return_value.stderr = ""

    import agile_agent_factory.tools.pytest_runner as pytest_runner
    pytest_runner.run_pytest()

    cmd = mock_run.call_args.args[0]
    assert not any(str(arg).startswith("--with") for arg in cmd)
