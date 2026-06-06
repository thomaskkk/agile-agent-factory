"""Tests for failure_recovery — classify_failure, files_from_traceback, _scaffold_missing_module."""
import importlib

import pytest


def _mod():
    import agile_agent_factory.nodes.failure_recovery as m
    return m


# ---------------------------------------------------------------------------
# files_from_traceback
# ---------------------------------------------------------------------------

def test_files_from_traceback_extracts_app_paths():
    tb = (
        "FAILED app/auth/login.py::test_login\n"
        "  File 'app/auth/login.py', line 5\n"
        "  assert result == True\n"
        "  File 'tests/test_auth.py', line 22\n"
    )
    result = _mod().files_from_traceback(tb)
    assert "app/auth/login.py" in result
    assert "tests/test_auth.py" in result


def test_files_from_traceback_preserves_order():
    tb = "File 'app/b.py'\nFile 'app/a.py'\nFile 'tests/test_b.py'"
    result = _mod().files_from_traceback(tb)
    assert result[0] == "app/b.py"
    assert result[1] == "app/a.py"
    assert result[2] == "tests/test_b.py"


def test_files_from_traceback_deduplicates():
    tb = "app/utils.py\napp/utils.py\napp/utils.py"
    result = _mod().files_from_traceback(tb)
    assert result.count("app/utils.py") == 1


def test_files_from_traceback_ignores_stdlib():
    tb = "File '/usr/lib/python3.12/os.py'\napp/module.py"
    result = _mod().files_from_traceback(tb)
    assert all(p.startswith(("app/", "tests/")) for p in result)


def test_files_from_traceback_empty_output():
    assert _mod().files_from_traceback("") == []


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------

def test_classify_failure_exit_4_with_no_module_named_is_missing_module():
    output = "ModuleNotFoundError: No module named 'app.database'"
    assert _mod().classify_failure(4, output) == "missing_module"


def test_classify_failure_exit_5_with_no_module_named_is_missing_module():
    output = "ModuleNotFoundError: No module named 'app.utils'"
    assert _mod().classify_failure(5, output) == "missing_module"


def test_classify_failure_exit_4_import_name_error_is_other():
    output = "ImportError: cannot import name 'clear_db' from 'app.database'"
    assert _mod().classify_failure(4, output) == "other"


def test_classify_failure_exit_4_syntax_error_is_other():
    output = "SyntaxError: invalid syntax\ncollected 0 items / 1 error"
    assert _mod().classify_failure(4, output) == "other"


def test_classify_failure_exit_5_no_output_is_other():
    assert _mod().classify_failure(5, "no tests ran") == "other"


def test_classify_failure_missing_dependency():
    output = "ModuleNotFoundError: No module named 'flask'"
    assert _mod().classify_failure(1, output) == "missing_dependency"


def test_classify_failure_missing_app_module():
    output = "ModuleNotFoundError: No module named 'app.utils'"
    assert _mod().classify_failure(1, output) == "missing_module"


def test_classify_failure_missing_tests_module():
    output = "ModuleNotFoundError: No module named 'tests.helpers'"
    assert _mod().classify_failure(1, output) == "missing_module"


def test_classify_failure_import_error_from_app():
    output = "ImportError: cannot import name 'foo' from app.bar"
    assert _mod().classify_failure(1, output) == "missing_module"


def test_classify_failure_assertion():
    output = "FAILED tests/test_x.py::test_y\nAssertionError: expected 1 got 2"
    assert _mod().classify_failure(1, output) == "assertion"


def test_classify_failure_other():
    output = "Some unknown error occurred"
    assert _mod().classify_failure(1, output) == "other"


# ---------------------------------------------------------------------------
# _scaffold_missing_module
# ---------------------------------------------------------------------------

def test_scaffold_missing_module_creates_empty_file(tmp_path, monkeypatch):
    import agile_agent_factory.nodes.failure_recovery as fr
    from pathlib import Path

    def fake_norm(path: str):
        p = Path(path)
        if str(p).startswith("app/") or str(p).startswith("tests/"):
            return tmp_path / p
        raise ValueError(f"Bad path: {path}")

    monkeypatch.setattr(fr, "normalize_generated_path", fake_norm)

    output = "ModuleNotFoundError: No module named 'app.utils'"
    result = fr._scaffold_missing_module(output)

    assert any("app/utils.py" in r for r in result)
    assert (tmp_path / "app" / "utils.py").exists()
    assert (tmp_path / "app" / "utils.py").read_text() == ""


def test_scaffold_missing_module_skips_non_app_modules(monkeypatch):
    import agile_agent_factory.nodes.failure_recovery as fr

    called = []
    def fake_norm(path: str):
        called.append(path)
        raise ValueError("Should not be called for non-app modules")

    monkeypatch.setattr(fr, "normalize_generated_path", fake_norm)

    output = "ModuleNotFoundError: No module named 'flask'"
    result = fr._scaffold_missing_module(output)
    assert result == []
    assert not called, "normalize_generated_path should not be called for non-app/tests modules"


def test_scaffold_missing_module_empty_output():
    result = _mod()._scaffold_missing_module("no import errors here")
    assert result == []
