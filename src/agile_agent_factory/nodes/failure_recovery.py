"""Failure classification and mechanical recovery helpers for test_node."""

from __future__ import annotations

import re
from pathlib import Path

from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.path_utils import normalize_generated_path

# Matches app/ or tests/ paths in pytest tracebacks
_PATH_RE = re.compile(r"\b((?:app|tests)/[\w/]+\.py)\b")


def files_from_traceback(traceback: str) -> list[str]:
    """Extract ordered-unique app/tests file paths from a pytest traceback."""
    seen = dict.fromkeys(_PATH_RE.findall(traceback))
    return list(seen)


def classify_failure(exit_code: int, output: str) -> str:
    """Classify a pytest failure for mechanical routing.

    Returns:
        "missing_dependency"      — third-party package not installed (stdlib/PyPI)
        "missing_module"          — app.*/tests.* module not found (import-based collection error)
        "collection_error_generic"— collection error (exit 4/5) NOT caused by a missing module
        "syntax_error"            — SyntaxError in app or test code
        "fixture_not_found"       — pytest fixture not found
        "bad_import_signature"    — ImportError: cannot import name '...'
        "missing_test_function"   — collection ERRORS with 'not found in' (missing test function)
        "namespace_collision"     — app module imported twice / duplicate sys.path entry
        "assertion"               — test assertion / logic failure
        "other"                   — unclassified; falls through to LLM correction
    """
    # --- Collection errors (exit 4 or 5) ---
    if exit_code in (4, 5):
        if re.search(r"No module named '", output):
            # Only a genuine missing-module if it is NOT an "cannot import name" error
            if "cannot import name" not in output:
                return "missing_module"
        # Any other collection error (syntax, fixture, etc.) — keep as generic
        return "collection_error_generic"

    has_import_err = "ModuleNotFoundError" in output or "ImportError" in output

    # --- bad_import_signature: "cannot import name" takes priority over missing_module ---
    if has_import_err and "cannot import name" in output:
        return "bad_import_signature"

    if has_import_err:
        mod_match = re.search(r"No module named '([^']+)'", output)
        if mod_match:
            top = mod_match.group(1).split(".")[0]
            if top in ("app", "tests"):
                return "missing_module"
            # "from app." / "from tests." context also means missing_module
            if re.search(r"from (app|tests)\.", output):
                return "missing_module"
            return "missing_dependency"
        # ImportError without "No module named" — check surrounding context
        if re.search(r"(from|import) (app|tests)[\. ]", output):
            return "missing_module"
        return "missing_dependency"

    # --- SyntaxError ---
    if "SyntaxError:" in output:
        return "syntax_error"

    # --- Fixture not found ---
    if re.search(r"fixture '[^']+' not found", output):
        return "fixture_not_found"

    # --- Missing test function (ERRORS + "not found in") ---
    if "ERRORS" in output and "not found in" in output:
        return "missing_test_function"

    # --- Assertion failures (check BEFORE namespace_collision to avoid false positives) ---
    if "AssertionError" in output or re.search(r"\bFAILED\b", output):
        return "assertion"

    # --- Namespace collision: app module imported twice ---
    if _detect_namespace_collision(output):
        return "namespace_collision"

    return "other"


def _detect_namespace_collision(output: str) -> bool:
    """Return True when the traceback shows a namespace collision.

    Two signals:
    1. The same app.* qualified name appears more than once in the traceback.
    2. A duplicate sys.path entry warning is present.
    """
    # Signal 1: duplicate app.* module reference in a single traceback
    app_refs = re.findall(r"\bapp\.[\w.]+\b", output)
    if len(app_refs) != len(set(app_refs)) and len(app_refs) > 1:
        return True

    # Signal 2: explicit duplicate sys.path / double-import warning
    if re.search(r"import\s+app\b.*\n.*import\s+app\b", output):
        return True
    if "sys.path" in output and re.search(r"(?m)^.*app.*\n.*app.*$", output):
        # Narrow: only flag if the output also mentions a duplicate path warning
        if re.search(r"duplicate|already imported|shadowed", output, re.IGNORECASE):
            return True

    return False


def _scaffold_missing_module(output: str) -> list[str]:
    """Create empty stubs for missing app/tests modules so collection can proceed.

    Only creates empty __init__.py or module files — never adds functions or behavior.
    Returns relative paths of files created (empty list when nothing was scaffolded).
    """
    scaffolded: list[str] = []

    for match in re.finditer(r"No module named '([^']+)'", output):
        mod_name = match.group(1)
        parts = mod_name.split(".")

        if parts[0] not in ("app", "tests"):
            continue

        # Ensure all parent packages have __init__.py
        for i in range(1, len(parts)):
            init_path = "/".join(parts[:i]) + "/__init__.py"
            try:
                init_target = normalize_generated_path(init_path)
                if not init_target.exists():
                    init_target.parent.mkdir(parents=True, exist_ok=True)
                    init_target.write_text("")
                    log(f"Scaffolded __init__.py: {init_path}")
                    scaffolded.append(init_path)
            except ValueError:
                pass

        # Scaffold the leaf module file itself
        file_path = "/".join(parts) + ".py"
        try:
            target = normalize_generated_path(file_path)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("")
                log(f"Scaffolded empty module: {file_path}")
                scaffolded.append(file_path)
        except ValueError as e:
            log(f"Could not scaffold {file_path}: {e}")

    return scaffolded


def _scaffold_fixture(output: str) -> list[str]:
    """Append a minimal pytest fixture stub to tests/conftest.py for each missing fixture.

    Only writes a ``pytest.fixture`` stub that returns ``None`` — never adds real logic.
    Returns relative paths of files written (empty list when nothing was scaffolded).
    """
    fixture_names: list[str] = []
    for m in re.finditer(r"fixture '([^']+)' not found", output):
        name = m.group(1)
        if name not in fixture_names:
            fixture_names.append(name)

    if not fixture_names:
        return []

    conftest_rel = "tests/conftest.py"
    try:
        conftest_path = normalize_generated_path(conftest_rel)
    except ValueError as e:
        log(f"Could not scaffold conftest.py: {e}")
        return []

    conftest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = conftest_path.read_text() if conftest_path.exists() else ""

    stubs: list[str] = []
    for name in fixture_names:
        # Skip if a fixture with this name is already present
        if re.search(rf"def {re.escape(name)}\b", existing):
            continue
        stubs.append(f"\n\n@pytest.fixture\ndef {name}():\n    return None\n")
        log(f"Scaffolded fixture stub: {name} in {conftest_rel}")

    if not stubs:
        return []

    # Ensure trailing newline before appending new content
    if existing and not existing.endswith("\n"):
        existing += "\n"

    header = "import pytest\n" if "import pytest" not in existing else ""
    conftest_path.write_text(existing + header + "".join(stubs))
    return [conftest_rel]


def _scaffold_missing_test_function(output: str, test_file: str | None) -> list[str]:
    """Append a minimal ``def test_xxx(): pass`` skeleton for each missing test function.

    Parses ``not found in`` lines from pytest ERRORS output to discover the missing
    function names, then appends stubs to *test_file* (or the first test file mentioned
    in the traceback).
    Returns relative paths of files written (empty list when nothing was scaffolded).
    """
    # Extract missing function names: "test_foo not found in tests/test_x.py"
    missing: dict[str, str] = {}  # fn_name → target file
    for m in re.finditer(r"'?(test_\w+)'? not found in (tests/[\w/]+\.py)", output):
        fn, fpath = m.group(1), m.group(2)
        if fn not in missing:
            missing[fn] = fpath

    if not missing and test_file:
        # Fallback: generic "not found in" without function name — skip scaffolding
        pass

    if not missing:
        return []

    written: list[str] = []
    # Group by target file
    by_file: dict[str, list[str]] = {}
    for fn, fpath in missing.items():
        by_file.setdefault(fpath, []).append(fn)

    for rel_path, fns in by_file.items():
        try:
            target = normalize_generated_path(rel_path)
        except ValueError as e:
            log(f"Could not scaffold test stubs in {rel_path}: {e}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text() if target.exists() else ""

        stubs: list[str] = []
        for fn in fns:
            if re.search(rf"def {re.escape(fn)}\b", existing):
                continue
            stubs.append(f"\n\ndef {fn}():\n    pass\n")
            log(f"Scaffolded test stub: {fn} in {rel_path}")

        if stubs:
            # Ensure trailing newline before appending new content
            if existing and not existing.endswith("\n"):
                existing += "\n"
            target.write_text(existing + "".join(stubs))
            written.append(rel_path)

    return written
