"""Tests for test_node — Milestones 1b, 2, 8."""
import importlib
from unittest.mock import MagicMock, call, patch

import pytest


def _tn():
    return importlib.import_module("agile_agent_factory.nodes.test_node")


def _make_state(sk="F1-1", column="testing", test_contract=None):
    return {
        "active_story_key": sk,
        "stories": {
            sk: {
                "column": column,
                "test_contract": test_contract or {},
            }
        },
        "epic_keys": [],
        "gherkin_criteria": {},
    }


def _mock_jira():
    jira = MagicMock()
    jira.add_comment_adf.return_value = None
    jira.set_flag.return_value = None
    jira.clear_flag.return_value = None
    jira.transition_to.return_value = None
    return jira


# ---------------------------------------------------------------------------
# Mechanical recovery: missing_dependency does NOT consume retries
# ---------------------------------------------------------------------------

def test_missing_dependency_retried_without_consuming_retries(monkeypatch):
    """A missing-dependency failure should trigger dep re-resolve without incrementing retries."""
    tn = _tn()

    state = _make_state()
    run_calls = []

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls.append(len(run_calls))
        if len(run_calls) == 1:
            return 1, "ModuleNotFoundError: No module named 'flask'"
        return 0, "1 passed"

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    result = tn.test_node(state)

    assert result["stories"]["F1-1"]["column"] == "code_review"
    # 2 pytest calls: dep failure + success
    assert len(run_calls) == 2


# ---------------------------------------------------------------------------
# Mechanical recovery: missing_module scaffolding does NOT consume retries
# ---------------------------------------------------------------------------

def test_missing_module_scaffolded_without_consuming_retries(monkeypatch):
    tn = _tn()

    state = _make_state()
    run_calls = []

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls.append(1)
        if len(run_calls) == 1:
            return 1, "ModuleNotFoundError: No module named 'app.utils'"
        return 0, "1 passed"

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())
    monkeypatch.setattr(tn, "_scaffold_missing_module", lambda output: ["app/utils.py"])

    result = tn.test_node(state)

    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert len(run_calls) == 2  # scaffold attempt + success


# ---------------------------------------------------------------------------
# Targeted + full suite routing (Milestone 2)
# ---------------------------------------------------------------------------

def test_targeted_green_full_green_advances_to_code_review(monkeypatch):
    tn = _tn()

    tc = {"test_file": "tests/test_auth.py", "test_functions": ["test_login"]}
    state = _make_state(test_contract=tc)

    def fake_run_pytest(extra_packages=None, test_targets=None):
        return 0, "1 passed"

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    result = tn.test_node(state)
    assert result["stories"]["F1-1"]["column"] == "code_review"


def test_targeted_green_full_red_in_scope_keeps_iterating(monkeypatch):
    """Full suite regression in write-scope → keep iterating correction."""
    tn = _tn()

    tc = {
        "test_file": "tests/test_auth.py",
        "target_imports": ["from app.auth import login"],
    }
    state = _make_state(test_contract=tc)

    call_n = [0]

    def fake_run_pytest(extra_packages=None, test_targets=None):
        call_n[0] += 1
        if call_n[0] == 1:
            return 0, "1 passed"   # targeted
        if call_n[0] == 2:
            # full suite fails — in-scope file cited
            return 1, "FAILED tests/test_auth.py::test_login\nAssertionError"
        return 0, "1 passed"  # final success

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "blueprint")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())
    monkeypatch.setattr(
        tn, "_correct_code",
        lambda *a, **kw: ("ok", ["tests/test_auth.py"]),
    )

    result = tn.test_node(state)
    assert result["stories"]["F1-1"]["column"] == "code_review"


def test_targeted_green_full_red_out_of_scope_quarantines(monkeypatch):
    """Full suite regression outside write-scope → quarantine and advance."""
    tn = _tn()

    tc = {"test_file": "tests/test_auth.py", "target_imports": ["from app.auth import login"]}
    state = _make_state(test_contract=tc)

    call_n = [0]

    def fake_run_pytest(extra_packages=None, test_targets=None):
        call_n[0] += 1
        if call_n[0] == 1:
            return 0, "1 passed"  # targeted
        # full suite: cross-story file fails
        return 1, "FAILED tests/test_other_story.py::test_foo\nAssertionError"

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "blueprint")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    result = tn.test_node(state)
    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert "regression_blockers" in result["stories"]["F1-1"]
    blockers = result["stories"]["F1-1"]["regression_blockers"]
    assert any("test_other_story" in b for b in blockers)


# ---------------------------------------------------------------------------
# Strategy budget: truncated response does NOT consume retries
# ---------------------------------------------------------------------------

def test_truncated_correction_uses_strategy_budget_not_retry(monkeypatch):
    """A truncated correction response must NOT consume the reasoning retry budget."""
    tn = _tn()

    state = _make_state()
    run_calls = [0]

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls[0] += 1
        if run_calls[0] <= 2:
            return 1, "FAILED tests/test_x.py::test_y\nAssertionError: expected True got False"
        return 0, "1 passed"

    correction_calls = [0]

    def fake_correct(blueprint, traceback, model=None):
        correction_calls[0] += 1
        if correction_calls[0] == 1:
            return ("truncated", [])  # first call: truncated
        return ("ok", ["app/x.py"])   # second call: ok

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_correct_code", fake_correct)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "bp")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    result = tn.test_node(state)
    assert result["stories"]["F1-1"]["column"] == "code_review"


# ---------------------------------------------------------------------------
# Genuine reasoning exhaustion still fires HITL
# ---------------------------------------------------------------------------

def test_reasoning_exhaustion_fires_hitl(monkeypatch):
    """After MAX_RETRIES_DEV genuine LLM reasoning failures, intervention HITL fires."""
    tn = _tn()
    from agile_agent_factory.config import MAX_RETRIES_DEV

    state = _make_state()
    interrupt_calls = []

    def fake_interrupt(payload):
        interrupt_calls.append(payload)
        return None

    def fake_run_pytest(extra_packages=None, test_targets=None):
        if interrupt_calls:
            return 0, "1 passed"
        return 1, "FAILED tests/test_x.py::test_y\nAssertionError"

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_correct_code", lambda *a, **kw: ("ok", ["app/x.py"]))
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "bp")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    with patch("langgraph.types.interrupt", side_effect=fake_interrupt):
        result = tn.test_node(state)

    assert len(interrupt_calls) >= 1
    assert interrupt_calls[0]["type"] == "intervention"
