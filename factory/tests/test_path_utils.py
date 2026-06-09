import pytest


def test_normalize_app_path(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    assert path_utils.normalize_generated_path("app/foo.py") == tmp_path / "app/foo.py"


def test_normalize_duplicate_app_prefix(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    assert path_utils.normalize_generated_path("app/app/foo.py") == tmp_path / "app/foo.py"


def test_normalize_tests_path(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    assert path_utils.normalize_generated_path("tests/test_x.py") == tmp_path / "tests/test_x.py"


def test_normalize_strips_leading_dotdot(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    assert path_utils.normalize_generated_path("../app/foo.py") == tmp_path / "app/foo.py"


def test_normalize_duplicate_tests_prefix(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    assert path_utils.normalize_generated_path("tests/tests/test_x.py") == tmp_path / "tests/test_x.py"


def test_reject_absolute_path(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Absolute paths are forbidden"):
        path_utils.normalize_generated_path("/etc/passwd")


def test_reject_path_outside_app_or_tests(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside allowed story paths"):
        path_utils.normalize_generated_path("config.py")


def test_reject_embedded_dotdot_traversal(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside allowed story paths"):
        path_utils.normalize_generated_path("app/../../../etc/passwd")


def test_normalize_strips_hallucinated_project_prefix_app(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    # LLM emits the repo-parent dir name as a leading component
    result = path_utils.normalize_generated_path("agile-agent-factory/app/models/__init__.py")
    assert result == tmp_path / "app/models/__init__.py"


def test_normalize_strips_hallucinated_project_prefix_tests(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    result = path_utils.normalize_generated_path("agile-agent-factory/tests/test_recipes_list.py")
    assert result == tmp_path / "tests/test_recipes_list.py"


def test_normalize_root_readme_when_explicitly_allowed(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    assert path_utils.normalize_generated_path(
        "README.md",
        allowed_root_paths=["README.md"],
    ) == tmp_path / "README.md"


def test_normalize_root_main_with_hallucinated_prefix_when_allowed(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    result = path_utils.normalize_generated_path(
        "agile-agent-factory/main.py",
        allowed_root_paths=["main.py"],
    )
    assert result == tmp_path / "main.py"


def test_reject_root_file_when_not_explicitly_allowed(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PRODUCT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside allowed story paths"):
        path_utils.normalize_generated_path("README.md")


def test_collision_guard_package_removes_shadow(tmp_path):
    """Writing __init__.py removes the same-named .py shadow."""
    import agile_agent_factory.tools.path_utils as path_utils
    shadow = tmp_path / "app" / "models.py"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("# shadow")
    init = tmp_path / "app" / "models" / "__init__.py"
    init.parent.mkdir(parents=True)
    result = path_utils.resolve_namespace_collision(init)
    assert result is True
    assert not shadow.exists(), "shadow .py should be removed when package __init__.py is written"


def test_collision_guard_flat_file_skipped_if_package_exists(tmp_path):
    """Writing a .py file is skipped when the same-named package already exists."""
    import agile_agent_factory.tools.path_utils as path_utils
    init = tmp_path / "app" / "models" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text("")
    target = tmp_path / "app" / "models.py"
    result = path_utils.resolve_namespace_collision(target)
    assert result is False, "should return False (skip) when package already claims this namespace"


def test_collision_guard_no_collision_returns_true(tmp_path):
    """No collision → returns True (safe to write)."""
    import agile_agent_factory.tools.path_utils as path_utils
    target = tmp_path / "app" / "utils.py"
    target.parent.mkdir(parents=True)
    result = path_utils.resolve_namespace_collision(target)
    assert result is True
