import pytest


def test_normalize_app_path(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    assert path_utils.normalize_generated_path("app/foo.py") == tmp_path / "app/foo.py"


def test_normalize_duplicate_app_prefix(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    assert path_utils.normalize_generated_path("app/app/foo.py") == tmp_path / "app/foo.py"


def test_normalize_tests_path(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    assert path_utils.normalize_generated_path("tests/test_x.py") == tmp_path / "tests/test_x.py"


def test_normalize_strips_leading_dotdot(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    assert path_utils.normalize_generated_path("../app/foo.py") == tmp_path / "app/foo.py"


def test_normalize_duplicate_tests_prefix(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    assert path_utils.normalize_generated_path("tests/tests/test_x.py") == tmp_path / "tests/test_x.py"


def test_reject_absolute_path(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Absolute paths are forbidden"):
        path_utils.normalize_generated_path("/etc/passwd")


def test_reject_path_outside_app_or_tests(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside app/ or tests/"):
        path_utils.normalize_generated_path("config.py")


def test_reject_embedded_dotdot_traversal(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside app/ or tests/"):
        path_utils.normalize_generated_path("app/../../../etc/passwd")


def test_normalize_strips_hallucinated_project_prefix_app(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    # LLM emits the repo-parent dir name as a leading component
    result = path_utils.normalize_generated_path("Factory_project_claude/app/models/__init__.py")
    assert result == tmp_path / "app/models/__init__.py"


def test_normalize_strips_hallucinated_project_prefix_tests(monkeypatch, tmp_path):
    import agile_agent_factory.tools.path_utils as path_utils
    monkeypatch.setattr(path_utils, "PARENT_ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    result = path_utils.normalize_generated_path("Factory_project_claude/tests/test_recipes_list.py")
    assert result == tmp_path / "tests/test_recipes_list.py"
