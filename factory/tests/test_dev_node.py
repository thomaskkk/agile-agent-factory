"""Tests for dev_node._write_generated_files namespace collision detection."""
import importlib

import pytest


def _mod():
    # nodes/__init__.py shadows 'dev_node' attribute with the function import,
    # so we bypass the package namespace via importlib.
    return importlib.import_module("agile_agent_factory.nodes.dev_node")


def test_write_init_removes_shadowed_py_file(tmp_path, monkeypatch):
    """Writing X/__init__.py must remove an existing X.py to prevent namespace shadow."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "models.py").write_text("# old flat module")

    _mod()._write_generated_files([
        {"path": "app/models/__init__.py", "content": "# package"},
    ])

    assert (tmp_path / "app" / "models" / "__init__.py").exists()
    assert not (tmp_path / "app" / "models.py").exists(), (
        "app/models.py must be removed — it is shadowed by the app/models/ package"
    )


def test_write_flat_py_skipped_when_package_owns_namespace(tmp_path, monkeypatch):
    """Writing X.py must be skipped when X/__init__.py already exists (package wins)."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)

    (tmp_path / "app" / "models").mkdir(parents=True)
    (tmp_path / "app" / "models" / "__init__.py").write_text("# package")

    written = _mod()._write_generated_files([
        {"path": "app/models.py", "content": "# flat module"},
    ])

    assert "app/models.py" not in written
    assert not (tmp_path / "app" / "models.py").exists(), (
        "app/models.py must not be created — app/models/ package already owns this namespace"
    )


def test_write_init_no_collision_when_no_shadow(tmp_path, monkeypatch):
    """Writing X/__init__.py with no existing X.py writes normally."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    written = _mod()._write_generated_files([
        {"path": "app/models/__init__.py", "content": "# package"},
    ])

    assert "app/models/__init__.py" in written
    assert (tmp_path / "app" / "models" / "__init__.py").exists()


def test_write_flat_py_no_collision_when_no_package(tmp_path, monkeypatch):
    """Writing X.py with no existing X/ package writes normally."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    written = _mod()._write_generated_files([
        {"path": "app/models.py", "content": "# flat module"},
    ])

    assert "app/models.py" in written
    assert (tmp_path / "app" / "models.py").exists()


# ---------------------------------------------------------------------------
# _correct_code: traceback-named files appear first and un-truncated (Milestone 1a)
# ---------------------------------------------------------------------------

def test_correct_code_puts_traceback_file_first(tmp_path, monkeypatch):
    """A file named in the traceback must appear BEFORE other files in the LLM prompt."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    # Create generated files — 'b.py' comes first alphabetically, but traceback names 'a.py'
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "a.py").write_text("# file a")
    (tmp_path / "app" / "b.py").write_text("# file b")

    captured_prompt = {}

    def fake_call_llm(prompt, system="", model=None, prefill=""):
        captured_prompt["val"] = prompt
        return "[]"

    traceback = "File 'app/a.py', line 5\nAssertionError"

    with patch("agile_agent_factory.nodes.dev_node.call_llm", side_effect=fake_call_llm), \
         patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path):
        _mod()._correct_code("blueprint", traceback)

    prompt = captured_prompt.get("val", "")
    assert "app/a.py" in prompt
    pos_a = prompt.index("### app/a.py")
    pos_b = prompt.index("### app/b.py")
    assert pos_a < pos_b, "Traceback-named file app/a.py must appear before app/b.py in prompt"


def test_correct_code_returns_tuple_status(tmp_path, monkeypatch):
    """_correct_code must return a (status, files) tuple."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("x = 1")

    def fake_call_llm(prompt, system="", model=None, prefill=""):
        return '[{"path": "app/x.py", "content": "x = 2"}]'

    with patch("agile_agent_factory.nodes.dev_node.call_llm", side_effect=fake_call_llm), \
         patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path):
        status, written = _mod()._correct_code("blueprint", "AssertionError")

    assert status == "ok"
    assert isinstance(written, list)


def test_correct_code_returns_empty_on_bad_json(tmp_path, monkeypatch):
    """Unparseable LLM response → ('empty', [])."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    with patch("agile_agent_factory.nodes.dev_node.call_llm", return_value="not json at all"), \
         patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path):
        status, written = _mod()._correct_code("blueprint", "AssertionError")

    assert status == "empty"
    assert written == []


def test_correct_code_includes_write_scope_in_system_prompt(tmp_path, monkeypatch):
    """write_scope must appear in the LLM system prompt so the model doesn't generate out-of-scope files."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "config.py").write_text("# config")

    captured_system = {}

    def fake_call_llm(prompt, system="", model=None, prefill=""):
        captured_system["val"] = system
        return "[]"

    scope = ["tests/test_app_skeleton.py", "app/config.py", "app/extensions.py"]

    with patch("agile_agent_factory.nodes.dev_node.call_llm", side_effect=fake_call_llm), \
         patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path):
        _mod()._correct_code("blueprint", "AssertionError", write_scope=scope)

    system = captured_system.get("val", "")
    assert "tests/test_app_skeleton.py" in system
    assert "app/config.py" in system
    assert "app/extensions.py" in system


# ---------------------------------------------------------------------------
# _write_generated_files: write_scope enforcement (Milestone 2)
# ---------------------------------------------------------------------------

def test_write_scope_drops_out_of_scope_path(tmp_path, monkeypatch):
    """Files whose path is NOT in write_scope must be silently skipped."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    written = _mod()._write_generated_files(
        [
            {"path": "app/allowed.py", "content": "# ok"},
            {"path": "app/forbidden.py", "content": "# nope"},
        ],
        write_scope=["app/allowed.py"],
    )

    assert "app/allowed.py" in written
    assert "app/forbidden.py" not in written
    assert (tmp_path / "app" / "allowed.py").exists()
    assert not (tmp_path / "app" / "forbidden.py").exists()


def test_write_scope_logs_warning_for_skipped_file(tmp_path, monkeypatch, capsys):
    """Skipping an out-of-scope write must emit a log warning."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    log_calls: list[str] = []
    with patch("agile_agent_factory.nodes.dev_node.log", side_effect=log_calls.append):
        _mod()._write_generated_files(
            [{"path": "app/other.py", "content": "x = 1"}],
            write_scope=["app/main.py"],
        )

    assert any("skipping out-of-scope write" in msg and "app/other.py" in msg for msg in log_calls)


def test_write_scope_writes_in_scope_paths_normally(tmp_path, monkeypatch):
    """Files whose path IS in write_scope must be written as normal."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    written = _mod()._write_generated_files(
        [{"path": "app/service.py", "content": "# service"}],
        write_scope=["app/service.py"],
    )

    assert "app/service.py" in written
    assert (tmp_path / "app" / "service.py").exists()
    assert (tmp_path / "app" / "service.py").read_text() == "# service"


def test_write_scope_directory_prefix_allows_nested_files(tmp_path, monkeypatch):
    """Directory scope entries should allow writes anywhere under that owned directory."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app" / "repository").mkdir(parents=True)

    written = _mod()._write_generated_files(
        [{"path": "app/repository/recipe_repository.py", "content": "# repo"}],
        write_scope=["app/repository/"],
    )

    assert written == ["app/repository/recipe_repository.py"]
    assert (tmp_path / "app" / "repository" / "recipe_repository.py").exists()


def test_write_scope_allows_explicit_root_files(tmp_path, monkeypatch):
    """Root files like README.md should be writable when explicitly owned by the story."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)

    written = _mod()._write_generated_files(
        [{"path": "README.md", "content": "# Product\n"}],
        write_scope=["README.md"],
    )

    assert written == ["README.md"]
    assert (tmp_path / "README.md").exists()


def test_write_scope_skips_unchanged_root_file(tmp_path, monkeypatch):
    """Rewriting identical content must be treated as a no-op, not a touched file."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "README.md").write_text("# Product\n")

    written = _mod()._write_generated_files(
        [{"path": "README.md", "content": "# Product\n"}],
        write_scope=["README.md"],
    )

    assert written == []
    assert (tmp_path / "README.md").read_text() == "# Product\n"


def test_write_scope_none_writes_all_paths(tmp_path, monkeypatch):
    """When write_scope is None, all valid paths are written (backward compat)."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    written = _mod()._write_generated_files(
        [
            {"path": "app/a.py", "content": "a = 1"},
            {"path": "app/b.py", "content": "b = 2"},
        ],
        write_scope=None,
    )

    assert "app/a.py" in written
    assert "app/b.py" in written


def test_write_scope_empty_list_writes_all_paths(tmp_path, monkeypatch):
    """When write_scope is an empty list, all valid paths are written (backward compat)."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    written = _mod()._write_generated_files(
        [
            {"path": "app/c.py", "content": "c = 3"},
            {"path": "app/d.py", "content": "d = 4"},
        ],
        write_scope=[],
    )

    assert "app/c.py" in written
    assert "app/d.py" in written


def test_write_scope_return_value_contains_only_written_paths(tmp_path, monkeypatch):
    """Return value must only include paths actually written, not skipped ones."""
    import agile_agent_factory.tools.path_utils as pu
    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    written = _mod()._write_generated_files(
        [
            {"path": "app/keep.py", "content": "# keep"},
            {"path": "app/skip.py", "content": "# skip"},
            {"path": "app/also_keep.py", "content": "# also keep"},
        ],
        write_scope=["app/keep.py", "app/also_keep.py"],
    )

    assert written == ["app/keep.py", "app/also_keep.py"]


# ---------------------------------------------------------------------------
# write_scope derivation from target_imports (Issue 1 regression guard)
# ---------------------------------------------------------------------------

def test_write_scope_derivation_produces_module_path_not_symbol():
    """write_scope derivation from target_imports must extract the MODULE path, not the symbol.

    Given "from app.auth import login_user", write_scope should contain "app/auth.py",
    NOT "login_user.py" (which was the bug caused by splitting on 'import').
    """
    import re

    def derive_write_scope(target_imports):
        write_scope = []
        for imp in target_imports:
            if isinstance(imp, str) and imp.strip():
                m = re.match(r"from (app(?:\.\w+)+) import", imp)
                if m:
                    path_str = m.group(1).replace(".", "/") + ".py"
                    if path_str not in write_scope:
                        write_scope.append(path_str)
        return write_scope

    result = derive_write_scope(["from app.auth import login_user"])

    assert "app/auth.py" in result, (
        "Expected 'app/auth.py' in write_scope, not the symbol name"
    )
    assert "login_user.py" not in result, (
        "Symbol name 'login_user.py' must NOT appear in write_scope"
    )


def test_write_scope_derivation_nested_module():
    """Nested module paths like 'from app.services.auth import authenticate' → 'app/services/auth.py'."""
    import re

    def derive_write_scope(target_imports):
        write_scope = []
        for imp in target_imports:
            if isinstance(imp, str) and imp.strip():
                m = re.match(r"from (app(?:\.\w+)+) import", imp)
                if m:
                    path_str = m.group(1).replace(".", "/") + ".py"
                    if path_str not in write_scope:
                        write_scope.append(path_str)
        return write_scope

    result = derive_write_scope(["from app.services.auth import authenticate"])

    assert "app/services/auth.py" in result
    assert "authenticate.py" not in result


def test_write_scope_derivation_deduplicates():
    """Multiple imports from the same module must not produce duplicate entries."""
    import re

    def derive_write_scope(target_imports):
        write_scope = []
        for imp in target_imports:
            if isinstance(imp, str) and imp.strip():
                m = re.match(r"from (app(?:\.\w+)+) import", imp)
                if m:
                    path_str = m.group(1).replace(".", "/") + ".py"
                    if path_str not in write_scope:
                        write_scope.append(path_str)
        return write_scope

    result = derive_write_scope([
        "from app.auth import login_user",
        "from app.auth import logout_user",
    ])

    assert result.count("app/auth.py") == 1


def test_write_scope_derivation_ignores_non_app_imports():
    """Imports not matching 'from app...' (e.g. stdlib, third-party) are silently skipped."""
    import re

    def derive_write_scope(target_imports):
        write_scope = []
        for imp in target_imports:
            if isinstance(imp, str) and imp.strip():
                m = re.match(r"from (app(?:\.\w+)+) import", imp)
                if m:
                    path_str = m.group(1).replace(".", "/") + ".py"
                    if path_str not in write_scope:
                        write_scope.append(path_str)
        return write_scope

    result = derive_write_scope([
        "from flask import Flask",
        "from os.path import join",
        "import json",
    ])

    assert result == []


# ---------------------------------------------------------------------------
# Fix #7: write_scope derived on non-rework (initial generation) path
# ---------------------------------------------------------------------------

def test_write_scope_derived_on_initial_gen_not_only_rework(tmp_path, monkeypatch):
    """write_scope must be derived from test_contract even when is_rework=False."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch, MagicMock

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()

    captured_scope = {}

    def fake_generate(blueprint, review_feedback="", write_scope=None, model=None):
        captured_scope["write_scope"] = write_scope

    dn = _mod()

    state = {
        "active_story_key": "F1-1",
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "development",  # initial gen, not rework
                "test_contract": {
                    "test_file": "tests/test_auth.py",
                    "target_imports": ["from app.auth import login"],
                },
            }
        },
        "epic_keys": [],
    }

    with (
        patch.object(dn, "_load_dev_context", return_value="some blueprint"),
        patch.object(dn, "_generate_code_with_llm", side_effect=fake_generate),
        patch("agile_agent_factory.nodes.dev_node.JiraClient", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node._safe_transition", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node.raise_quota_interrupt", return_value=None),
        patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path),
    ):
        dn.dev_node(state)

    scope = captured_scope.get("write_scope") or []
    assert "tests/test_auth.py" in scope, "test_file must be in write_scope for initial gen"
    assert "app/auth.py" in scope, "module from target_imports must be in write_scope for initial gen"


def test_derive_story_write_scope_expands_scaffold_contract_files(tmp_path):
    """Scaffold stories must inherit architecture-backed directory/root hints from their test contract."""
    import importlib
    from unittest.mock import patch

    helpers = importlib.import_module("agile_agent_factory.nodes.helpers")

    product_root = tmp_path / "product"
    (product_root / "tests").mkdir(parents=True)
    (product_root / "tests" / "test_scaffold_setup.py").write_text(
        """
readme_candidates = ["README.md", "README.rst"]
required_dirs = [
    "app/routes",
    "app/models",
    "app/repository",
    "app/templates",
]
req_file = "requirements.txt"
pyproject = "pyproject.toml"
"""
    )

    task_path = tmp_path / "F3-730.md"
    task_path.write_text(
        """
Scenario: Web server starts on configured port
  Given the application is properly initialized
  When a developer executes 'flask run' or 'uvicorn main:app' from the project root
  Then a web server starts and listens on http://localhost:5000

### File Contracts
- `app/__init__.py`: Package init
- `app/main.py`: Flask app entry point
- `app/routes/__init__.py`: Routes package init
- `app/routes/health.py`: Health check route
- `app/models/__init__.py`: Models package init
- `app/repository/__init__.py`: Repository package init
- `app/repository/recipe_repository.py`: Repository implementation
- `app/templates/base.html`: Base template
- `app/templates/home.html`: Home template
- `tests/test_scaffold_setup.py`: Scaffold tests
- `requirements.txt`: Dependencies
- `README.md`: Documentation
"""
    )

    story = {
        "story_key": "F3-730",
        "test_contract": {
            "test_file": "tests/test_scaffold_setup.py",
            "target_imports": [
                "from app.main import app",
                "from app.routes.health import health_check",
            ],
            "sample_data": [
                {"directory": "app/routes", "must_exist": True},
                {"directory": "app/models", "must_exist": True},
            ],
        },
    }

    with patch("agile_agent_factory.nodes.helpers.bp_task_path", return_value=task_path):
        scope = helpers.derive_story_write_scope("F3-730", story, product_root=product_root)

    assert "tests/test_scaffold_setup.py" in scope
    assert "app/main.py" in scope
    assert "app/routes/health.py" in scope
    assert "app/routes/__init__.py" in scope
    assert "app/repository/" in scope
    assert "app/templates/" in scope
    assert "app/models/" in scope
    assert "main.py" in scope
    assert "README.md" in scope
    assert "requirements.txt" in scope


def test_derive_story_write_scope_prefers_package_init_for_package_import(tmp_path):
    """Package imports should resolve to __init__.py when the story architecture owns a package."""
    import importlib
    from unittest.mock import patch

    helpers = importlib.import_module("agile_agent_factory.nodes.helpers")

    product_root = tmp_path / "product"
    task_path = tmp_path / "F3-755.md"
    task_path.write_text(
        """
### File Contracts
- `app/models/__init__.py`: Re-export Recipe
- `app/models/recipe.py`: Recipe model
- `app/repository.py`: Repository
- `tests/test_recipe_repository.py`: Repository tests
"""
    )

    story = {
        "story_key": "F3-755",
        "test_contract": {
            "test_file": "tests/test_recipe_repository.py",
            "target_imports": [
                "from app.repository import RecipeRepository",
                "from app.models import Recipe",
            ],
        },
    }

    with patch("agile_agent_factory.nodes.helpers.bp_task_path", return_value=task_path):
        scope = helpers.derive_story_write_scope("F3-755", story, product_root=product_root)

    assert "tests/test_recipe_repository.py" in scope
    assert "app/repository.py" in scope
    assert "app/models/__init__.py" in scope
    assert "app/models.py" not in scope


def test_derive_story_write_scope_ignores_readme_mentions_in_test_docstrings(tmp_path):
    """Test prose mentioning README must not widen write_scope to README.md."""
    import importlib
    from unittest.mock import patch

    helpers = importlib.import_module("agile_agent_factory.nodes.helpers")

    product_root = tmp_path / "product"
    (product_root / "tests").mkdir(parents=True)
    (product_root / "tests" / "test_web_server.py").write_text(
        '''
"""Tests for the Flask web server."""

from app.application import create_app


def test_readme_documents_setup():
    """README or application module documents setup instructions."""
    assert callable(create_app)
'''
    )

    task_path = tmp_path / "F3-757.md"
    task_path.write_text(
        """
### File Contracts
- `app/application.py`: Flask application factory
- `app.py`: Entry point
- `tests/test_web_server.py`: Web server tests
- `README.md`: Project documentation
"""
    )

    story = {
        "story_key": "F3-757",
        "test_contract": {
            "test_file": "tests/test_web_server.py",
            "target_imports": ["from app.application import create_app"],
        },
    }

    with patch("agile_agent_factory.nodes.helpers.bp_task_path", return_value=task_path):
        scope = helpers.derive_story_write_scope("F3-757", story, product_root=product_root)

    assert "tests/test_web_server.py" in scope
    assert "app/application.py" in scope
    assert "README.md" not in scope


def test_dev_node_expands_scaffold_write_scope_from_architecture(tmp_path, monkeypatch):
    """dev_node should pass scaffold-owned architecture files into the guarded generation scope."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import MagicMock, patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    dn = _mod()

    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_scaffold_setup.py").write_text(
        'required_dirs = ["app/repository", "app/templates"]\nreadme = "README.md"\n'
    )
    task_path = tmp_path / "F3-730.md"
    task_path.write_text(
        """
Scenario: Web server starts on configured port
  Given the application is properly initialized
  When a developer executes 'uvicorn main:app' from the project root
  Then a web server starts and listens on http://localhost:5000

### File Contracts
- `app/main.py`: Flask app entry point
- `app/routes/health.py`: Health check route
- `app/repository/__init__.py`: Repository package init
- `app/templates/base.html`: Base template
- `README.md`: Documentation
"""
    )

    captured_scope = {}

    def fake_generate(blueprint, review_feedback="", write_scope=None, model=None):
        captured_scope["write_scope"] = write_scope

    state = {
        "active_story_key": "F3-730",
        "stories": {
            "F3-730": {
                "story_key": "F3-730",
                "column": "development",
                "test_contract": {
                    "test_file": "tests/test_scaffold_setup.py",
                    "target_imports": [
                        "from app.main import app",
                        "from app.routes.health import health_check",
                    ],
                },
            }
        },
        "epic_keys": [],
    }

    with (
        patch.object(dn, "_load_dev_context", return_value="some blueprint"),
        patch.object(dn, "_generate_code_with_llm", side_effect=fake_generate),
        patch("agile_agent_factory.nodes.dev_node.JiraClient", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node._safe_transition", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node.raise_quota_interrupt", return_value=None),
        patch("agile_agent_factory.tools.aider_client.is_available", return_value=False),
        patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path),
        patch("agile_agent_factory.nodes.helpers.PRODUCT_ROOT", tmp_path),
        patch("agile_agent_factory.nodes.helpers.bp_task_path", return_value=task_path),
    ):
        dn.dev_node(state)

    scope = captured_scope.get("write_scope") or []
    assert "app/repository/" in scope
    assert "app/templates/" in scope
    assert "main.py" in scope
    assert "README.md" in scope


def test_aider_failure_falls_back_to_llm(tmp_path):
    """If aider is available but fails, dev_node must fall back to LLM-direct generation."""
    from unittest.mock import MagicMock, patch

    dn = _mod()
    state = {
        "active_story_key": "F1-1",
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "development",
                "test_contract": {},
            }
        },
        "epic_keys": [],
    }

    with (
        patch.object(dn, "_load_dev_context", return_value="some blueprint"),
        patch.object(dn, "_generate_code_with_llm") as mock_llm,
        patch("agile_agent_factory.nodes.dev_node.JiraClient", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node._safe_transition", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node.raise_quota_interrupt", return_value=None),
        patch("agile_agent_factory.tools.aider_client.is_available", return_value=True),
        patch("agile_agent_factory.tools.aider_client.run_task", return_value={"success": False, "output": "boom"}),
    ):
        dn.dev_node(state)

    mock_llm.assert_called_once()


def test_llm_direct_scope_guard_rolls_back_out_of_scope_writes(tmp_path, monkeypatch):
    """If a generator writes outside write_scope, dev_node must roll it back without crashing."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import MagicMock, patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    dn = _mod()

    state = {
        "active_story_key": "F1-1",
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "development",
                "test_contract": {
                    "test_file": "tests/test_auth.py",
                    "target_imports": ["from app.auth import login"],
                },
            }
        },
        "epic_keys": [],
    }

    def fake_generate(*args, **kwargs):
        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "forbidden.py").write_text("x = 1")

    with (
        patch.object(dn, "_load_dev_context", return_value="some blueprint"),
        patch.object(dn, "_generate_code_with_llm", side_effect=fake_generate),
        patch("agile_agent_factory.nodes.dev_node.JiraClient", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node._safe_transition", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node.raise_quota_interrupt", return_value=None),
        patch("agile_agent_factory.tools.aider_client.is_available", return_value=False),
        patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path),
        patch("agile_agent_factory.nodes.helpers.PRODUCT_ROOT", tmp_path),
    ):
        result = dn.dev_node(state)

    assert result["stories"]["F1-1"]["column"] == "testing"
    assert result["stories"]["F1-1"]["last_changed_files"] == []
    assert not (tmp_path / "app" / "forbidden.py").exists()


def test_llm_direct_scope_guard_keeps_in_scope_writes(tmp_path, monkeypatch):
    """Out-of-scope side effects must be rolled back while valid in-scope writes survive."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import MagicMock, patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    dn = _mod()

    state = {
        "active_story_key": "F1-1",
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "development",
                "test_contract": {
                    "test_file": "tests/test_auth.py",
                    "target_imports": ["from app.auth import login"],
                },
            }
        },
        "epic_keys": [],
    }

    def fake_generate(*args, **kwargs):
        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "tests").mkdir(exist_ok=True)
        (tmp_path / "app" / "auth.py").write_text("def login():\n    return True\n")
        (tmp_path / "tests" / "test_auth.py").write_text("def test_login():\n    assert True\n")
        (tmp_path / "app" / "forbidden.py").write_text("x = 1")

    with (
        patch.object(dn, "_load_dev_context", return_value="some blueprint"),
        patch.object(dn, "_generate_code_with_llm", side_effect=fake_generate),
        patch("agile_agent_factory.nodes.dev_node.JiraClient", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node._safe_transition", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node.raise_quota_interrupt", return_value=None),
        patch("agile_agent_factory.tools.aider_client.is_available", return_value=False),
        patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path),
        patch("agile_agent_factory.nodes.helpers.PRODUCT_ROOT", tmp_path),
    ):
        result = dn.dev_node(state)

    assert result["stories"]["F1-1"]["column"] == "testing"
    assert result["stories"]["F1-1"]["last_changed_files"] == ["app/auth.py", "tests/test_auth.py"]
    assert (tmp_path / "app" / "auth.py").exists()
    assert (tmp_path / "tests" / "test_auth.py").exists()
    assert not (tmp_path / "app" / "forbidden.py").exists()


def test_initial_gen_prompt_includes_write_scope_paths():
    """_generate_code_with_llm initial gen branch must include write_scope paths in the prompt."""
    from unittest.mock import patch

    dn = _mod()
    captured = {}

    def fake_call_llm_json(prompt, system="", fallback=None, model=None, prefill=""):
        captured["prompt"] = prompt
        return []

    write_scope = ["tests/test_auth.py", "app/auth.py"]
    with patch("agile_agent_factory.nodes.dev_node.call_llm_json", side_effect=fake_call_llm_json):
        dn._generate_code_with_llm(
            blueprint="Build auth module",
            review_feedback="",  # initial gen path
            write_scope=write_scope,
            model=None,
        )

    prompt = captured.get("prompt", "")
    assert "tests/test_auth.py" in prompt, "write_scope paths must appear in initial gen prompt"
    assert "app/auth.py" in prompt, "write_scope paths must appear in initial gen prompt"


def test_generate_code_with_llm_skips_string_entries_without_crashing(tmp_path, monkeypatch):
    """Malformed list entries from the LLM should be ignored instead of crashing generation."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    with patch(
        "agile_agent_factory.nodes.dev_node.call_llm_json",
        return_value=["README.md", {"path": "app/main.py", "content": "app = None\n"}],
    ):
        _mod()._generate_code_with_llm(
            blueprint="Build scaffold",
            review_feedback="",
            write_scope=["app/main.py"],
            model=None,
        )

    assert (tmp_path / "app" / "main.py").exists()
    assert not (tmp_path / "README.md").exists()


def test_generate_code_with_llm_accepts_dict_wrapped_files_payload(tmp_path, monkeypatch):
    """Object-shaped generation payloads with a files array should still be accepted."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()

    with patch(
        "agile_agent_factory.nodes.dev_node.call_llm_json",
        return_value={"files": [{"path": "app/main.py", "content": "app = None\n"}]},
    ):
        _mod()._generate_code_with_llm(
            blueprint="Build scaffold",
            review_feedback="",
            write_scope=["app/main.py"],
            model=None,
        )

    assert (tmp_path / "app" / "main.py").exists()


def test_dev_rework_noop_readme_uses_readme_fallback():
    """README review loops should retry through the README-specific fallback before giving up."""
    from unittest.mock import MagicMock, patch

    dn = _mod()
    state = {
        "active_story_key": "F1-1",
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "code_review",
                "review_status": "rework_needed",
                "review_rejection_reason": "README must contain server startup instructions",
                "test_contract": {"test_file": "tests/test_scaffold_setup.py"},
            }
        },
        "epic_keys": [],
    }

    with (
        patch.object(dn, "_load_dev_context", return_value="blueprint"),
        patch.object(dn, "_generate_code_with_llm_guarded", return_value=[]),
        patch.object(dn, "_generate_readme_rework_guarded", return_value=["README.md"]) as readme_fallback,
        patch.object(dn, "derive_story_write_scope", return_value=["README.md"]),
        patch("agile_agent_factory.nodes.dev_node.JiraClient", MagicMock()),
        patch("agile_agent_factory.nodes.dev_node.raise_quota_interrupt", return_value=None),
        patch("agile_agent_factory.tools.aider_client.is_available", return_value=False),
    ):
        result = dn.dev_node(state)

    readme_fallback.assert_called_once_with(["README.md"])
    assert result["stories"]["F1-1"]["review_status"] == "pending_review"
    assert result["stories"]["F1-1"]["last_changed_files"] == ["README.md"]


def test_dev_rework_noop_escalates_after_retries():
    """Repeated no-op rework attempts should interrupt instead of looping back to review unchanged."""
    from unittest.mock import MagicMock, patch

    dn = _mod()
    state = {
        "active_story_key": "F1-1",
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "code_review",
                "review_status": "rework_needed",
                "review_rejection_reason": "Missing startup instructions",
                "test_contract": {
                    "test_file": "tests/test_auth.py",
                    "target_imports": ["from app.auth import login"],
                },
            }
        },
        "epic_keys": [],
    }

    jira = MagicMock()
    interrupt_calls = []

    def fake_interrupt(payload):
        interrupt_calls.append(payload)
        return None

    with (
        patch.object(dn, "_load_dev_context", return_value="blueprint"),
        patch.object(dn, "_generate_code_with_llm_guarded", side_effect=[[], []]),
        patch.object(dn, "_generate_readme_rework_guarded", return_value=[]),
        patch("agile_agent_factory.nodes.dev_node.JiraClient", return_value=jira),
        patch("agile_agent_factory.nodes.dev_node.raise_quota_interrupt", return_value=None),
        patch("agile_agent_factory.tools.aider_client.is_available", return_value=False),
        patch("langgraph.types.interrupt", side_effect=fake_interrupt),
    ):
        result = dn.dev_node(state)

    assert interrupt_calls == [{"type": "intervention", "blocking_key": "F1-1", "source": "dev_noop_rework"}]
    jira.set_flag.assert_called_once_with("F1-1")
    assert result["stories"]["F1-1"]["hitl_type"] == "intervention"
    assert result["stories"]["F1-1"]["review_status"] == "rework_needed"


def test_dev_rework_noop_after_hitl_resume_routes_back_to_review():
    """Manual HITL fixes should not re-escalate just because the dev agent has no further diff."""
    from unittest.mock import MagicMock, patch

    dn = _mod()
    state = {
        "active_story_key": "F1-1",
        "stories": {
            "F1-1": {
                "story_key": "F1-1",
                "column": "code_review",
                "review_status": "rework_needed",
                "review_rejection_reason": "Connect home page to the shared recipe store.",
                "hitl_feedback": "Manually fixed on disk. Re-run review.",
                "test_contract": {
                    "test_file": "tests/test_home_page.py",
                    "target_imports": ["from app.routes import home"],
                },
            }
        },
        "epic_keys": [],
    }

    jira = MagicMock()

    with (
        patch.object(dn, "_load_dev_context", return_value="blueprint"),
        patch.object(dn, "_generate_code_with_llm_guarded", side_effect=[[], []]),
        patch.object(dn, "_generate_readme_rework_guarded", return_value=[]),
        patch("agile_agent_factory.nodes.dev_node.JiraClient", return_value=jira),
        patch("agile_agent_factory.nodes.dev_node.raise_quota_interrupt", return_value=None),
        patch("agile_agent_factory.tools.aider_client.is_available", return_value=False),
        patch("langgraph.types.interrupt") as mock_interrupt,
    ):
        result = dn.dev_node(state)

    mock_interrupt.assert_not_called()
    jira.set_flag.assert_not_called()
    assert result["stories"]["F1-1"]["review_status"] == "pending_review"
    assert result["stories"]["F1-1"]["review_rejection_reason"] == ""
    assert result["stories"]["F1-1"]["hitl_type"] is None


# ---------------------------------------------------------------------------
# Fix #5: correction prompt includes spec block
# ---------------------------------------------------------------------------

def test_correct_code_prompt_includes_test_contract_functions(tmp_path, monkeypatch):
    """Correction prompt must include expected test function names from test_contract."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "auth.py").write_text("x = 1")

    captured_prompt = {}

    def fake_call_llm(prompt, system="", model=None, prefill=""):
        captured_prompt["val"] = prompt
        return "[]"

    test_contract = {
        "test_functions": ["test_login_valid", "test_login_invalid"],
        "target_imports": ["from app.auth import login"],
    }

    with patch("agile_agent_factory.nodes.dev_node.call_llm", side_effect=fake_call_llm), \
         patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path):
        _mod()._correct_code(
            "blueprint",
            "AssertionError: test_login_valid failed",
            test_contract=test_contract,
        )

    prompt = captured_prompt.get("val", "")
    assert "test_login_valid" in prompt, "correction prompt must include expected test functions"
    assert "test_login_invalid" in prompt, "correction prompt must include expected test functions"


def test_correct_code_reduced_scope_retry_propagates_quota(tmp_path, monkeypatch):
    """B1: a quota error raised inside the reduced-scope truncation retry must propagate.

    Previously `except (LLMQuotaExceeded, Exception)` swallowed the quota event into a
    ('truncated', []) strategy retry, so the pipeline never reached quota backoff/HITL.
    """
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "a.py").write_text("# file a")

    dn = _mod()
    calls = {"n": 0}

    def fake_call_llm(prompt, system="", model=None, prefill=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return '[{"path":'  # truncated first response
        raise LLMQuotaExceeded("anthropic", "rate limited")  # reduced-scope retry

    with (
        patch("agile_agent_factory.nodes.dev_node.call_llm", side_effect=fake_call_llm),
        patch("agile_agent_factory.nodes.dev_node._parse_correction_response", return_value=None),
        patch("agile_agent_factory.nodes.dev_node._is_truncated_json", return_value=True),
        patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path),
    ):
        with pytest.raises(LLMQuotaExceeded):
            dn._correct_code("blueprint", "File 'app/a.py', line 5\nAssertionError")

    assert calls["n"] == 2, "reduced-scope retry must have been attempted"


def test_correct_code_prompt_valid_without_test_contract(tmp_path, monkeypatch):
    """Correction call with no test_contract must still produce a valid prompt (spec block omitted)."""
    import agile_agent_factory.tools.path_utils as pu
    from unittest.mock import patch

    monkeypatch.setattr(pu, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("x = 1")

    def fake_call_llm(prompt, system="", model=None, prefill=""):
        return '[{"path": "app/x.py", "content": "x = 2"}]'

    with patch("agile_agent_factory.nodes.dev_node.call_llm", side_effect=fake_call_llm), \
         patch("agile_agent_factory.nodes.dev_node.PRODUCT_ROOT", tmp_path):
        status, written = _mod()._correct_code("blueprint", "AssertionError", test_contract=None)

    assert status == "ok"
    assert written  # files were written normally
