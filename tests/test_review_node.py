"""Tests for review_node — Milestone 5 verdict classification helpers."""
from unittest.mock import MagicMock, patch

import pytest


def _import_rn():
    import importlib
    return importlib.import_module("agile_agent_factory.nodes.review_node")


# ---------------------------------------------------------------------------
# Layer 1: out-of-scope rejection detection
# ---------------------------------------------------------------------------

def test_out_of_scope_rejection_when_only_out_of_scope_files_cited():
    rn = _import_rn()
    reason = "tests/test_other_story.py has a broken import"
    write_scope = ["tests/test_my_story.py", "app/my_module.py"]
    assert rn._is_out_of_scope_rejection(reason, write_scope) is True


def test_not_out_of_scope_when_in_scope_file_cited():
    rn = _import_rn()
    reason = "app/my_module.py is missing the required endpoint"
    write_scope = ["app/my_module.py"]
    assert rn._is_out_of_scope_rejection(reason, write_scope) is False


def test_not_out_of_scope_when_no_files_cited():
    rn = _import_rn()
    reason = "The acceptance criteria were not met"
    write_scope = ["app/my_module.py"]
    assert rn._is_out_of_scope_rejection(reason, write_scope) is False


def test_not_out_of_scope_when_write_scope_empty():
    rn = _import_rn()
    reason = "tests/other.py is broken"
    assert rn._is_out_of_scope_rejection(reason, []) is False


# ---------------------------------------------------------------------------
# Layer 2: vagueness detection
# ---------------------------------------------------------------------------

def test_vague_rejection_with_no_anchor():
    rn = _import_rn()
    reason = "The implementation does not satisfy the requirements."
    assert rn._is_vague_rejection(reason, []) is True


def test_not_vague_when_file_path_cited():
    rn = _import_rn()
    reason = "app/auth.py is missing the login function"
    assert rn._is_vague_rejection(reason, []) is False


def test_not_vague_when_test_name_cited():
    rn = _import_rn()
    reason = "test_login_valid is not passing"
    assert rn._is_vague_rejection(reason, []) is False


def test_not_vague_when_criterion_text_present():
    rn = _import_rn()
    criteria = ["Scenario: Login success\n  Given valid creds\n  When login\n  Then success"]
    reason = "Scenario: Login success is not implemented correctly"
    assert rn._is_vague_rejection(reason, criteria) is False


def test_empty_rejection_reason_is_vague():
    rn = _import_rn()
    assert rn._is_vague_rejection("", []) is True


# ---------------------------------------------------------------------------
# _filter_rejection combining both layers
# ---------------------------------------------------------------------------

def test_filter_rejects_out_of_scope_only_rejection():
    rn = _import_rn()
    reason = "tests/test_other.py has missing fixtures"
    write_scope = ["tests/test_mine.py"]
    should_count, _ = rn._filter_rejection(reason, write_scope, [], "F1-1")
    assert should_count is False


def test_filter_rejects_vague_rejection():
    rn = _import_rn()
    reason = "The code does not look complete."
    should_count, _ = rn._filter_rejection(reason, [], [], "F1-1")
    assert should_count is False


def test_filter_passes_concrete_in_scope_rejection():
    rn = _import_rn()
    reason = "app/auth.py is missing the login() function that test_login_valid requires"
    write_scope = ["app/auth.py", "tests/test_auth.py"]
    should_count, _ = rn._filter_rejection(reason, write_scope, [], "F1-1")
    assert should_count is True


def test_filter_passes_real_in_scope_defect():
    rn = _import_rn()
    reason = "tests/test_tasks.py::test_create_task fails: app/tasks.py raises KeyError"
    write_scope = ["tests/test_tasks.py", "app/tasks.py"]
    criteria = ["Scenario: Create task\n  Given empty list\n  When add task\n  Then listed"]
    should_count, _ = rn._filter_rejection(reason, write_scope, criteria, "F1-2")
    assert should_count is True


# ---------------------------------------------------------------------------
# Rework prompt contains write_scope + failing criteria (Milestone 5b)
# ---------------------------------------------------------------------------

def test_rework_prompt_includes_write_scope(tmp_path):
    """_generate_code_with_llm rework branch must mention write_scope files in the prompt."""
    import importlib
    from unittest.mock import patch
    dn = importlib.import_module("agile_agent_factory.nodes.dev_node")

    captured = {}

    def fake_call_llm_json(prompt, system="", fallback=None, model=None, prefill=""):
        captured["prompt"] = prompt
        return []

    write_scope = ["tests/test_auth.py", "app/auth.py"]
    with patch("agile_agent_factory.nodes.dev_node.call_llm_json", side_effect=fake_call_llm_json):
        dn._generate_code_with_llm(
            blueprint="Build auth module",
            review_feedback="login() function is missing",
            write_scope=write_scope,
            model=None,
        )

    assert "tests/test_auth.py" in captured.get("prompt", ""), (
        "write_scope files must appear in rework prompt"
    )
    assert "app/auth.py" in captured.get("prompt", ""), (
        "write_scope files must appear in rework prompt"
    )


# ---------------------------------------------------------------------------
# Milestone 3: _pre_review_gate
# ---------------------------------------------------------------------------

def _import_prg():
    import importlib
    rn = importlib.import_module("agile_agent_factory.nodes.review_node")
    return rn._pre_review_gate


def test_prg_check1_fails_when_file_missing(tmp_path):
    """Gate fails when a write_scope file is absent from disk."""
    prg = _import_prg()
    story = {}
    write_scope = ["app/missing.py"]
    passed, reason = prg(story, write_scope, tmp_path)
    assert passed is False
    assert "Missing required files" in reason
    assert "app/missing.py" in reason


def test_prg_check1_passes_when_all_files_exist(tmp_path):
    """Gate check 1 passes when all write_scope files are present on disk."""
    prg = _import_prg()
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "present.py").write_text("x = 1")
    story = {}
    write_scope = ["app/present.py"]
    passed, reason = prg(story, write_scope, tmp_path)
    # check 1 passes; no test_contract so checks 2–4 also trivially pass
    assert passed is True
    assert reason == ""


def test_prg_check2_fails_when_expected_test_missing(tmp_path):
    """Gate fails when an expected test function is absent from the test file.

    expected_tests comes from ready_contract (normalized), not test_contract (raw QA output).
    """
    prg = _import_prg()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_auth.py"
    test_file.write_text("def test_login_invalid(): pass\n")
    story = {
        "test_contract": {
            "test_file": "tests/test_auth.py",
            "target_imports": [],
        },
        "ready_contract": {
            "expected_tests": ["test_login_valid", "test_login_invalid"],
        },
    }
    passed, reason = prg(story, [], tmp_path)
    assert passed is False
    assert "Missing expected test functions" in reason
    assert "test_login_valid" in reason


def test_prg_check2_passes_when_all_expected_tests_present(tmp_path):
    """Gate check 2 passes when all expected test functions are defined."""
    prg = _import_prg()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_auth.py"
    test_file.write_text("def test_login_valid(): pass\ndef test_login_invalid(): pass\n")
    story = {
        "test_contract": {
            "test_file": "tests/test_auth.py",
            "target_imports": [],
        },
        "ready_contract": {
            "expected_tests": ["test_login_valid", "test_login_invalid"],
        },
    }
    passed, reason = prg(story, [], tmp_path)
    assert passed is True
    assert reason == ""


def test_prg_check2_uses_ready_contract_not_test_contract(tmp_path):
    """expected_tests in test_contract alone must NOT trigger the check.

    test_contract (raw QA) uses test_functions; ready_contract (normalized) uses expected_tests.
    The gate must look at ready_contract so it never silently gets [] from the wrong key.
    """
    prg = _import_prg()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_auth.py").write_text("def test_other(): pass\n")
    story = {
        "test_contract": {
            "test_file": "tests/test_auth.py",
            "test_functions": ["test_login_valid"],  # raw QA key — should NOT be read here
            "target_imports": [],
        },
        "ready_contract": {
            "expected_tests": ["test_login_valid"],  # normalized key — MUST be read here
        },
    }
    passed, reason = prg(story, [], tmp_path)
    assert passed is False, "Gate must detect missing test_login_valid via ready_contract"
    assert "test_login_valid" in reason


def test_prg_check3_fails_when_source_has_syntax_error(tmp_path):
    """Gate fails when a source file referenced by target_imports has a syntax error."""
    prg = _import_prg()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "auth.py").write_text("def broken(\n")  # syntax error
    story = {
        "test_contract": {
            "test_file": "",
            "expected_tests": [],
            "target_imports": ["from app.auth import login"],
        }
    }
    passed, reason = prg(story, [], tmp_path)
    assert passed is False
    assert "Syntax error in app/auth.py" in reason


def test_prg_check3_passes_when_source_is_valid(tmp_path):
    """Gate check 3 passes when all source files are syntactically valid."""
    prg = _import_prg()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "auth.py").write_text("def login(user, pwd): return True\n")
    story = {
        "test_contract": {
            "test_file": "",
            "expected_tests": [],
            "target_imports": ["from app.auth import login"],
        }
    }
    passed, reason = prg(story, [], tmp_path)
    assert passed is True
    assert reason == ""


def test_prg_check4_fails_when_pytest_fails(tmp_path):
    """Gate fails when targeted pytest run returns non-zero exit code."""
    prg = _import_prg()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_foo.py"
    test_file.write_text("def test_pass(): pass\n")
    story = {
        "test_contract": {
            "test_file": "tests/test_foo.py",
            "expected_tests": [],
            "target_imports": [],
        }
    }
    with patch(
        "agile_agent_factory.nodes.review_node._pre_review_gate.__wrapped__"
        if hasattr(prg, "__wrapped__") else
        "agile_agent_factory.tools.pytest_runner.run_pytest",
        return_value=(1, "FAILED tests/test_foo.py::test_pass - AssertionError"),
    ):
        passed, reason = prg(story, [], tmp_path)
    assert passed is False
    assert "Targeted tests failing" in reason


def test_prg_check4_passes_when_pytest_passes(tmp_path):
    """Gate check 4 passes when targeted pytest run returns exit 0."""
    prg = _import_prg()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_foo.py"
    test_file.write_text("def test_pass(): pass\n")
    story = {
        "test_contract": {
            "test_file": "tests/test_foo.py",
            "expected_tests": [],
            "target_imports": [],
        }
    }
    with patch(
        "agile_agent_factory.tools.pytest_runner.run_pytest",
        return_value=(0, "1 passed"),
    ):
        passed, reason = prg(story, [], tmp_path)
    assert passed is True
    assert reason == ""


def test_prg_check4_passes_extra_packages_to_run_pytest(tmp_path):
    """Gate check 4 forwards extra_packages to run_pytest so deps like flask are available."""
    prg = _import_prg()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_foo.py").write_text("def test_pass(): pass\n")
    story = {
        "test_contract": {
            "test_file": "tests/test_foo.py",
            "expected_tests": [],
            "target_imports": [],
        }
    }
    with patch(
        "agile_agent_factory.tools.pytest_runner.run_pytest",
        return_value=(0, "1 passed"),
    ) as mock_run:
        passed, reason = prg(story, [], tmp_path, extra_packages=["flask", "sqlalchemy"])
    assert passed is True
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == ["flask", "sqlalchemy"]


def test_prg_all_checks_pass(tmp_path):
    """Gate returns (True, '') when all 4 checks pass."""
    prg = _import_prg()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (app_dir / "mod.py").write_text("def do_thing(): pass\n")
    (tests_dir / "test_mod.py").write_text("def test_do_thing(): pass\n")
    story = {
        "test_contract": {
            "test_file": "tests/test_mod.py",
            "target_imports": ["from app.mod import do_thing"],
        },
        "ready_contract": {
            "expected_tests": ["test_do_thing"],
        },
    }
    write_scope = ["app/mod.py", "tests/test_mod.py"]
    with patch(
        "agile_agent_factory.tools.pytest_runner.run_pytest",
        return_value=(0, "1 passed"),
    ):
        passed, reason = prg(story, write_scope, tmp_path)
    assert passed is True
    assert reason == ""


# ---------------------------------------------------------------------------
# Milestone 3: review_node routes via pre-gate
# ---------------------------------------------------------------------------

def _make_review_state(story_key: str = "F1-1", extra_story: dict | None = None) -> dict:
    story = {
        "column": "code_review",
        "review_status": "pending_review",
        "review_retries": 0,
        "test_contract": {},
    }
    if extra_story:
        story.update(extra_story)
    return {
        "active_story_key": story_key,
        "stories": {story_key: story},
        "epic_keys": [],
        "gherkin_criteria": {},
        "review_retries": 0,
    }


def test_review_node_routes_rework_when_gate_fails():
    """review_node returns rework_needed without calling review_patch when gate fails."""
    import importlib
    rn = importlib.import_module("agile_agent_factory.nodes.review_node")

    state = _make_review_state()

    with (
        patch("agile_agent_factory.nodes.review_node._pre_review_gate", return_value=(False, "Missing required files: ['app/x.py']")),
        patch("agile_agent_factory.agents.reviewer_agent.review_patch") as mock_review,
        patch("agile_agent_factory.nodes.review_node.JiraClient") as MockJira,
    ):
        mock_jira_inst = MagicMock()
        MockJira.return_value = mock_jira_inst
        result = rn.review_node(state)

    mock_review.assert_not_called()
    assert result["stories"]["F1-1"]["review_status"] == "rework_needed"


def test_review_node_calls_review_patch_when_gate_passes():
    """review_node calls review_patch normally when the pre-gate passes."""
    import importlib
    rn = importlib.import_module("agile_agent_factory.nodes.review_node")

    state = _make_review_state()

    approved_result = MagicMock()
    approved_result.payload = {"approved": True, "reason": ""}

    with (
        patch("agile_agent_factory.nodes.review_node._pre_review_gate", return_value=(True, "")),
        patch("agile_agent_factory.agents.reviewer_agent.review_patch", return_value=approved_result) as mock_review,
        patch("agile_agent_factory.tools.jira_client.JiraClient") as MockJira,
    ):
        mock_jira_inst = MagicMock()
        MockJira.return_value = mock_jira_inst
        result = rn.review_node(state)

    mock_review.assert_called_once()
    assert result.get("review_approved") is True
