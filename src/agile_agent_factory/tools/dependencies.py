"""Resolve the third-party packages the generated product needs at test time.

The Tech Lead is asked to declare `dependencies`, but the LLM sometimes omits the
field entirely. Relying on it alone leaves `state["dependencies"]` empty even for an
obvious Flask app. So we derive the effective set from three signals and union them:

1. Whatever the TL declared in state (when present).
2. The UX spec's chosen technology (e.g. Flask).
3. A static scan of the actual imports in the generated app/ and tests/ code.

Signal 3 is the most reliable because it reflects what the code really imports.
"""

import ast
import sys
import warnings
from pathlib import Path

# import-name -> pip package name (only where they differ)
_PACKAGE_ALIASES = {
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
}

# UX-spec technology -> pip package (stdlib/none technologies omitted)
UX_TECH_TO_PACKAGE = {
    "flask": "flask",
    "fastapi": "fastapi",
    "django": "django",
    "rich": "rich",
    "click": "click",
}

# never pass these to `uv run --with` (local packages or already present)
_EXCLUDED = {"app", "tests", "pytest"}


def scan_third_party_imports(product_root: Path) -> set[str]:
    """Parse generated app/ and tests/ files and return third-party top-level packages."""
    found: set[str] = set()
    for target in ("app", "tests"):
        d = product_root / target
        if not d.exists():
            continue
        for f in d.rglob("*.py"):
            try:
                with warnings.catch_warnings():
                    # generated product code may contain invalid escape sequences etc.;
                    # those warnings are about the product, not our concern while scanning
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(f.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue  # a broken file can't be scanned; other files still inform us
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        found.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:  # absolute imports only
                        found.add(node.module.split(".")[0])

    third_party = {
        m for m in found
        if m
        and m not in sys.stdlib_module_names
        and m not in _EXCLUDED
        and not m.startswith("_")
    }
    return {_PACKAGE_ALIASES.get(m, m) for m in third_party}


def resolve_dependencies(state: dict, product_root: Path) -> list[str]:
    """Union of declared deps, UX technology package, and scanned code imports."""
    deps: set[str] = set(state.get("dependencies", []) or [])

    tech = ((state.get("ux_spec") or {}).get("technology") or "").lower()
    if tech in UX_TECH_TO_PACKAGE:
        deps.add(UX_TECH_TO_PACKAGE[tech])

    deps |= scan_third_party_imports(product_root)
    deps -= _EXCLUDED
    return sorted(deps)
