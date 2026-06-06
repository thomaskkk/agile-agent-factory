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
