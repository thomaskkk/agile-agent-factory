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
