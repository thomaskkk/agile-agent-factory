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
