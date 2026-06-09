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


def test_targeted_green_full_red_out_of_scope_requeues_done_owner(monkeypatch):
    """Full suite regression outside write-scope → requeue the owning done story."""
    tn = _tn()

    state = {
        "active_story_key": "F1-1",
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "testing",
                "test_contract": {
                    "test_file": "tests/test_auth.py",
                    "target_imports": ["from app.auth import login"],
                },
            },
            "F1-2": {
                "story_key": "F1-2",
                "column": "done",
                "test_contract": {
                    "test_file": "tests/test_other_story.py",
                    "target_imports": ["from app.other_story import run"],
                },
            },
        },
        "epic_keys": [],
        "gherkin_criteria": {},
    }

    call_n = [0]
    jira = _mock_jira()

    def fake_run_pytest(extra_packages=None, test_targets=None):
        call_n[0] += 1
        if call_n[0] == 1:
            return 0, "1 passed"  # targeted
        # full suite: cross-story file fails
        return 1, "FAILED tests/test_other_story.py::test_foo\nAssertionError"

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "blueprint")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: jira)

    result = tn.test_node(state)
    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert result["stories"]["F1-1"]["regression_blockers"] == []
    assert result["stories"]["F1-2"]["column"] == "testing"
    assert "tests/test_other_story.py" in result["stories"]["F1-2"]["incoming_regression_files"]
    assert jira.add_comment_adf.call_count >= 2


def test_incoming_regression_runs_full_suite_before_targeted(monkeypatch):
    """Owner stories reopened for regression should skip the targeted-first fast path."""
    tn = _tn()

    state = {
        "active_story_key": "F1-2",
        "stories": {
            "F1-2": {
                "story_key": "F1-2",
                "column": "testing",
                "test_contract": {
                    "test_file": "tests/test_other_story.py",
                    "target_imports": ["from app.other_story import run"],
                },
                "incoming_regression_files": ["tests/test_other_story.py"],
                "incoming_regression_output": "FAILED tests/test_other_story.py::test_foo",
            }
        },
        "epic_keys": [],
        "gherkin_criteria": {},
    }

    seen_targets = []

    def fake_run_pytest(extra_packages=None, test_targets=None):
        seen_targets.append(test_targets)
        return 0, "2 passed"

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "blueprint")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    result = tn.test_node(state)

    assert result["stories"]["F1-2"]["column"] == "code_review"
    assert seen_targets == [None]


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

    def fake_correct(blueprint, traceback, model=None, write_scope=None, **kwargs):
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


def test_reasoning_exhaustion_does_not_clear_flag(monkeypatch):
    """Downstream HITL must leave the Jira flag set until main.py resumes it."""
    tn = _tn()
    jira = _mock_jira()

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
    monkeypatch.setattr(tn, "JiraClient", lambda: jira)

    with patch("langgraph.types.interrupt", side_effect=fake_interrupt):
        tn.test_node(state)

    jira.set_flag.assert_called_once()
    jira.clear_flag.assert_not_called()


def test_repeated_failure_narrows_correction_scope(monkeypatch):
    """When the same failure repeats, test_node should narrow the next correction scope."""
    tn = _tn()

    tc = {
        "test_file": "tests/test_auth.py",
        "target_imports": ["from app.auth import login", "from app.shared import helper"],
    }
    state = _make_state(test_contract=tc)
    jira = _mock_jira()
    scopes: list[list[str] | None] = []
    run_calls = {"n": 0}

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls["n"] += 1
        if run_calls["n"] < 3:
            return 1, "FAILED tests/test_auth.py::test_login\nAssertionError\napp/auth.py"
        return 0, "1 passed"

    def fake_correct(blueprint, traceback, model=None, write_scope=None, **kwargs):
        scopes.append(list(write_scope) if write_scope is not None else None)
        return ("ok", ["app/auth.py"])

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_correct_code", fake_correct)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "bp")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: jira)

    result = tn.test_node(state)

    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert scopes[0] == ["tests/test_auth.py", "app/auth.py", "app/shared.py"]
    assert scopes[1] == ["tests/test_auth.py", "app/auth.py"]


# ---------------------------------------------------------------------------
# New deterministic classes (Milestone 1)
# ---------------------------------------------------------------------------

def test_fixture_not_found_scaffolded_without_consuming_retries(monkeypatch):
    """fixture_not_found should scaffold conftest and retry without consuming retries."""
    tn = _tn()

    state = _make_state()
    run_calls = [0]

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls[0] += 1
        if run_calls[0] == 1:
            return 1, "fixture 'db_session' not found"
        return 0, "1 passed"

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())
    monkeypatch.setattr(tn, "_scaffold_fixture", lambda output: ["tests/conftest.py"])

    result = tn.test_node(state)

    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert run_calls[0] == 2  # scaffold + success


def test_missing_test_function_scaffolded_without_consuming_retries(monkeypatch):
    """missing_test_function should scaffold stub and retry without consuming retries."""
    tn = _tn()

    state = _make_state()
    run_calls = [0]

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls[0] += 1
        if run_calls[0] == 1:
            return 1, "ERRORS\n'test_login' not found in tests/test_auth.py"
        return 0, "1 passed"

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())
    monkeypatch.setattr(tn, "_scaffold_missing_test_function", lambda output, tf: ["tests/test_auth.py"])

    result = tn.test_node(state)

    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert run_calls[0] == 2


def test_namespace_collision_reruns_dep_resolution_without_consuming_retries(monkeypatch):
    """namespace_collision should re-resolve deps and retry without consuming retries."""
    tn = _tn()

    state = _make_state()
    run_calls = [0]
    dep_resolve_calls = [0]

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls[0] += 1
        if run_calls[0] == 1:
            return 1, "import app.models\nimport app.models\nduplicate: app.models referenced twice"
        return 0, "1 passed"

    def fake_resolve(legacy_state, root):
        dep_resolve_calls[0] += 1
        return []

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "")
    monkeypatch.setattr(tn, "resolve_dependencies", fake_resolve)
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    result = tn.test_node(state)

    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert run_calls[0] == 2
    # resolve_dependencies called at start + once for collision recovery
    assert dep_resolve_calls[0] >= 2


def test_syntax_error_uses_targeted_llm_hint_not_retries(monkeypatch):
    """syntax_error should send a targeted hint to _correct_code using strategy_retries."""
    tn = _tn()

    state = _make_state()
    run_calls = [0]
    correct_calls = []

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls[0] += 1
        if run_calls[0] == 1:
            return 1, "SyntaxError: invalid syntax\napp/models.py line 5"
        return 0, "1 passed"

    def fake_correct(blueprint, traceback, model=None, write_scope=None, **kwargs):
        correct_calls.append(traceback)
        return ("ok", ["app/models.py"])

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_correct_code", fake_correct)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "bp")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    result = tn.test_node(state)

    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert len(correct_calls) == 1
    # Hint must be prepended to traceback
    assert "TARGETED FIX REQUIRED" in correct_calls[0]
    assert "SyntaxError" in correct_calls[0]


def test_bad_import_signature_uses_targeted_llm_hint(monkeypatch):
    """bad_import_signature should send a targeted hint to _correct_code."""
    tn = _tn()

    state = _make_state()
    run_calls = [0]
    correct_calls = []

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls[0] += 1
        if run_calls[0] == 1:
            return 1, "ImportError: cannot import name 'create_app' from 'app.factory'"
        return 0, "1 passed"

    def fake_correct(blueprint, traceback, model=None, write_scope=None, **kwargs):
        correct_calls.append(traceback)
        return ("ok", ["app/factory.py"])

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_correct_code", fake_correct)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "bp")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    result = tn.test_node(state)

    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert "TARGETED FIX REQUIRED" in correct_calls[0]
    assert "cannot import name" in correct_calls[0]


def test_collection_error_generic_uses_targeted_llm_hint(monkeypatch):
    """collection_error_generic (exit 4/5 non-module) should send a targeted hint."""
    tn = _tn()

    state = _make_state()
    run_calls = [0]
    correct_calls = []

    def fake_run_pytest(extra_packages=None, test_targets=None):
        run_calls[0] += 1
        if run_calls[0] == 1:
            return 4, "collected 0 items / 1 error\nsome unexpected issue"
        return 0, "1 passed"

    def fake_correct(blueprint, traceback, model=None, write_scope=None, **kwargs):
        correct_calls.append(traceback)
        return ("ok", ["app/something.py"])

    monkeypatch.setattr(tn, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(tn, "_correct_code", fake_correct)
    monkeypatch.setattr(tn, "_load_dev_context", lambda sk: "bp")
    monkeypatch.setattr(tn, "resolve_dependencies", lambda *a, **kw: [])
    monkeypatch.setattr(tn, "JiraClient", lambda: _mock_jira())

    result = tn.test_node(state)

    assert result["stories"]["F1-1"]["column"] == "code_review"
    assert "TARGETED FIX REQUIRED" in correct_calls[0]
