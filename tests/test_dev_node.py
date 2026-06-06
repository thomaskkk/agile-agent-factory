"""Tests for dev_node._write_generated_files namespace collision detection."""
import importlib


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
