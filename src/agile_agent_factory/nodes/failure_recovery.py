"""Failure classification and mechanical recovery helpers for test_node."""

from __future__ import annotations

import re

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
        "missing_dependency"  — third-party package not installed (stdlib/PyPI)
        "missing_module"      — app.*/tests.* module not found, or collection error (exit 4/5)
        "assertion"           — test assertion / logic failure
        "other"               — unclassified; falls through to LLM correction
    """
    if exit_code in (4, 5):
        return "missing_module"

    has_import_err = "ModuleNotFoundError" in output or "ImportError" in output
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

    if "AssertionError" in output or re.search(r"\bFAILED\b", output):
        return "assertion"

    return "other"


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
