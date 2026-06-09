def test_scan_finds_third_party_import(tmp_path):
    from agile_agent_factory.tools.dependencies import scan_third_party_imports

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "factory.py").write_text("from flask import Flask\nimport os\n")

    found = scan_third_party_imports(tmp_path)
    assert "flask" in found
    assert "os" not in found  # stdlib excluded


def test_scan_excludes_local_and_pytest(tmp_path):
    from agile_agent_factory.tools.dependencies import scan_third_party_imports

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "import pytest\nfrom app.factory import create_app\nimport requests\n"
    )

    found = scan_third_party_imports(tmp_path)
    assert "requests" in found
    assert "app" not in found
    assert "pytest" not in found


def test_scan_applies_package_alias(tmp_path):
    from agile_agent_factory.tools.dependencies import scan_third_party_imports

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "cfg.py").write_text("import yaml\n")

    found = scan_third_party_imports(tmp_path)
    assert "pyyaml" in found
    assert "yaml" not in found


def test_scan_skips_unparseable_file(tmp_path):
    from agile_agent_factory.tools.dependencies import scan_third_party_imports

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "broken.py").write_text("def f(:\n")  # syntax error
    (tmp_path / "app" / "ok.py").write_text("import flask\n")

    found = scan_third_party_imports(tmp_path)
    assert "flask" in found  # broken file skipped, others still scanned


def test_resolve_unions_all_signals(tmp_path):
    from agile_agent_factory.tools.dependencies import resolve_dependencies

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("import requests\n")

    state = {
        "dependencies": ["sqlalchemy"],
        "ux_spec": {"technology": "Flask"},
    }
    result = resolve_dependencies(state, tmp_path)
    assert set(result) == {"sqlalchemy", "flask", "requests"}


def test_resolve_recovers_flask_when_state_deps_empty(tmp_path):
    """The reported bug: dependencies empty in state but the code clearly imports flask."""
    from agile_agent_factory.tools.dependencies import resolve_dependencies

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "factory.py").write_text("from flask import Flask, render_template\n")

    state = {"dependencies": [], "ux_spec": {"technology": "Flask"}}
    result = resolve_dependencies(state, tmp_path)
    assert "flask" in result
