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


def test_classify_failure_exit_4_import_name_error_is_collection_error_generic():
    # exit-4 + "cannot import name" → collection_error_generic (not missing_module or bad_import_signature)
    output = "ImportError: cannot import name 'clear_db' from 'app.database'"
    assert _mod().classify_failure(4, output) == "collection_error_generic"


def test_classify_failure_exit_4_syntax_error_is_collection_error_generic():
    output = "SyntaxError: invalid syntax\ncollected 0 items / 1 error"
    assert _mod().classify_failure(4, output) == "collection_error_generic"


def test_classify_failure_exit_5_no_output_is_collection_error_generic():
    assert _mod().classify_failure(5, "no tests ran") == "collection_error_generic"


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
    # "cannot import name" takes priority → bad_import_signature
    output = "ImportError: cannot import name 'foo' from app.bar"
    assert _mod().classify_failure(1, output) == "bad_import_signature"


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


# ---------------------------------------------------------------------------
# scaffold_paths — path-list-driven structural stubs (I5)
# ---------------------------------------------------------------------------

def test_scaffold_paths_creates_package_and_leaf_stubs(tmp_path, monkeypatch):
    """scaffold_paths creates parent __init__.py packages and an empty leaf module."""
    import agile_agent_factory.nodes.failure_recovery as fr
    from pathlib import Path

    def fake_norm(path: str):
        p = Path(path)
        if str(p).startswith("app/") or str(p).startswith("tests/"):
            return tmp_path / p
        raise ValueError(f"Bad path: {path}")

    monkeypatch.setattr(fr, "normalize_generated_path", fake_norm)

    result = fr.scaffold_paths(["app/pkg/leaf.py"])

    assert "app/pkg/leaf.py" in result
    assert "app/pkg/__init__.py" in result
    assert (tmp_path / "app" / "pkg" / "leaf.py").read_text() == ""
    assert (tmp_path / "app" / "pkg" / "__init__.py").read_text() == ""


def test_scaffold_paths_skips_non_py_and_root_files(monkeypatch):
    """Non-.py paths, directory entries, and root-level files are never scaffolded."""
    import agile_agent_factory.nodes.failure_recovery as fr

    called = []

    def fake_norm(path: str):
        called.append(path)
        raise AssertionError("normalize must not be reached for filtered paths")

    monkeypatch.setattr(fr, "normalize_generated_path", fake_norm)

    result = fr.scaffold_paths(["README.md", "app/templates/", "requirements.txt", "main.py"])

    assert result == []
    assert called == []


def test_scaffold_paths_skips_existing_files(tmp_path, monkeypatch):
    """Existing files are left untouched (no overwrite, not reported as scaffolded)."""
    import agile_agent_factory.nodes.failure_recovery as fr
    from pathlib import Path

    def fake_norm(path: str):
        p = Path(path)
        if str(p).startswith("app/") or str(p).startswith("tests/"):
            return tmp_path / p
        raise ValueError(f"Bad path: {path}")

    monkeypatch.setattr(fr, "normalize_generated_path", fake_norm)

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "existing.py").write_text("x = 1\n")

    result = fr.scaffold_paths(["app/existing.py"])

    assert "app/existing.py" not in result
    assert (tmp_path / "app" / "existing.py").read_text() == "x = 1\n"


# ---------------------------------------------------------------------------
# classify_failure — new classes (Milestone 1)
# ---------------------------------------------------------------------------

def test_classify_failure_syntax_error():
    output = "SyntaxError: invalid syntax\napp/models.py, line 12"
    assert _mod().classify_failure(1, output) == "syntax_error"


def test_classify_failure_fixture_not_found():
    output = "fixture 'db_session' not found"
    assert _mod().classify_failure(1, output) == "fixture_not_found"


def test_classify_failure_bad_import_signature_exit_1():
    output = "ImportError: cannot import name 'create_app' from 'app.factory'"
    assert _mod().classify_failure(1, output) == "bad_import_signature"


def test_classify_failure_missing_test_function():
    output = "ERRORS\n'test_login' not found in tests/test_auth.py"
    assert _mod().classify_failure(1, output) == "missing_test_function"


def test_classify_failure_namespace_collision():
    output = (
        "import app.models\n"
        "import app.models\n"
        "duplicate: app.models referenced twice"
    )
    assert _mod().classify_failure(1, output) == "namespace_collision"


def test_classify_failure_collection_error_generic_exit_4():
    # exit-4 with no ModuleNotFoundError → collection_error_generic
    output = "collected 0 items / 1 error\nsome unknown collection issue"
    assert _mod().classify_failure(4, output) == "collection_error_generic"


def test_classify_failure_collection_error_generic_exit_5():
    output = "no tests collected"
    assert _mod().classify_failure(5, output) == "collection_error_generic"


def test_classify_failure_exit_4_no_module_named_still_missing_module():
    # exit-4 with explicit "No module named" → still missing_module (not collection_error_generic)
    output = "ModuleNotFoundError: No module named 'app.database'"
    assert _mod().classify_failure(4, output) == "missing_module"


# ---------------------------------------------------------------------------
# _scaffold_fixture
# ---------------------------------------------------------------------------

def test_scaffold_fixture_creates_conftest_stub(tmp_path, monkeypatch):
    import agile_agent_factory.nodes.failure_recovery as fr
    from pathlib import Path

    def fake_norm(path: str):
        p = Path(path)
        if str(p).startswith("tests/"):
            return tmp_path / p
        raise ValueError(f"Bad path: {path}")

    monkeypatch.setattr(fr, "normalize_generated_path", fake_norm)

    output = "fixture 'db_session' not found"
    result = fr._scaffold_fixture(output)

    assert result == ["tests/conftest.py"]
    conftest = (tmp_path / "tests" / "conftest.py").read_text()
    assert "def db_session" in conftest
    assert "@pytest.fixture" in conftest


def test_scaffold_fixture_skips_existing_fixture(tmp_path, monkeypatch):
    import agile_agent_factory.nodes.failure_recovery as fr
    from pathlib import Path

    conftest_path = tmp_path / "tests" / "conftest.py"
    conftest_path.parent.mkdir(parents=True, exist_ok=True)
    conftest_path.write_text("import pytest\n\n@pytest.fixture\ndef db_session():\n    return None\n")

    def fake_norm(path: str):
        p = Path(path)
        if str(p).startswith("tests/"):
            return tmp_path / p
        raise ValueError(f"Bad path: {path}")

    monkeypatch.setattr(fr, "normalize_generated_path", fake_norm)

    output = "fixture 'db_session' not found"
    result = fr._scaffold_fixture(output)

    # No new stub written — fixture already present
    assert result == []


def test_scaffold_fixture_empty_output():
    result = _mod()._scaffold_fixture("some unrelated error")
    assert result == []


# ---------------------------------------------------------------------------
# _scaffold_missing_test_function
# ---------------------------------------------------------------------------

def test_scaffold_missing_test_function_creates_stub(tmp_path, monkeypatch):
    import agile_agent_factory.nodes.failure_recovery as fr
    from pathlib import Path

    def fake_norm(path: str):
        p = Path(path)
        if str(p).startswith("tests/"):
            return tmp_path / p
        raise ValueError(f"Bad path: {path}")

    monkeypatch.setattr(fr, "normalize_generated_path", fake_norm)

    output = "ERRORS\n'test_login' not found in tests/test_auth.py"
    result = fr._scaffold_missing_test_function(output, "tests/test_auth.py")

    assert "tests/test_auth.py" in result
    content = (tmp_path / "tests" / "test_auth.py").read_text()
    assert "def test_login" in content


def test_scaffold_missing_test_function_skips_existing(tmp_path, monkeypatch):
    import agile_agent_factory.nodes.failure_recovery as fr
    from pathlib import Path

    test_path = tmp_path / "tests" / "test_auth.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("def test_login():\n    pass\n")

    def fake_norm(path: str):
        p = Path(path)
        if str(p).startswith("tests/"):
            return tmp_path / p
        raise ValueError(f"Bad path: {path}")

    monkeypatch.setattr(fr, "normalize_generated_path", fake_norm)

    output = "ERRORS\n'test_login' not found in tests/test_auth.py"
    result = fr._scaffold_missing_test_function(output, "tests/test_auth.py")

    assert result == []  # already present — no write


def test_scaffold_missing_test_function_empty_output():
    result = _mod()._scaffold_missing_test_function("no errors here", None)
    assert result == []


# ---------------------------------------------------------------------------
# _scaffold_missing_module — namespace collision guard
# ---------------------------------------------------------------------------

def test_scaffold_missing_module_skips_init_when_flat_module_exists(tmp_path, monkeypatch):
    """_scaffold_missing_module must NOT create app/models/__init__.py when app/models.py exists."""
    import agile_agent_factory.tools.path_utils as path_utils
    import agile_agent_factory.nodes.failure_recovery as fr
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    importlib.reload(fr)

    # Pre-existing flat module
    (tmp_path / "app").mkdir(parents=True)
    flat = tmp_path / "app" / "models.py"
    flat.write_text("# real Recipe module")

    output = "No module named 'app.models.recipe'"
    fr._scaffold_missing_module(output)

    # The flat module must still exist (not removed)
    assert flat.exists(), "app/models.py must not be deleted"
    # The package __init__ must NOT have been created (collision avoided)
    pkg_init = tmp_path / "app" / "models" / "__init__.py"
    assert not pkg_init.exists(), "app/models/__init__.py must not be created when app/models.py exists"


def test_scaffold_missing_module_removes_shadow_when_creating_package(tmp_path, monkeypatch):
    """When scaffolding __init__.py, remove the same-named .py shadow."""
    import agile_agent_factory.tools.path_utils as path_utils
    import agile_agent_factory.nodes.failure_recovery as fr
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    importlib.reload(fr)

    # Pre-existing flat module at app/services.py
    (tmp_path / "app").mkdir(parents=True)
    flat = tmp_path / "app" / "services.py"
    flat.write_text("")
    (tmp_path / "app" / "__init__.py").write_text("")

    # Scaffold a submodule: app.services.recipe_service
    output = "No module named 'app.services.recipe_service'"
    fr._scaffold_missing_module(output)

    # The flat services.py should be removed (package took over)
    assert not flat.exists(), "app/services.py should be removed when package app/services/ is created"
    assert (tmp_path / "app" / "services" / "__init__.py").exists()
