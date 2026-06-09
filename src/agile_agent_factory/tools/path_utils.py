import re
from pathlib import Path

from agile_agent_factory.config import PRODUCT_ROOT as PARENT_ROOT

ROOT_WRITE_ALLOWLIST = {
    "README",
    "README.md",
    "README.rst",
    "README.txt",
    "requirements.txt",
    "pyproject.toml",
    "main.py",
    "app.py",
    "wsgi.py",
}


def _allowed_root_targets(allowed_root_paths: list[str] | None) -> set[Path]:
    targets: set[Path] = set()
    for entry in allowed_root_paths or []:
        if not isinstance(entry, str):
            continue
        cleaned = entry.strip().lstrip("./")
        if not cleaned or cleaned.endswith("/"):
            continue
        if "/" in cleaned:
            continue
        if cleaned not in ROOT_WRITE_ALLOWLIST:
            continue
        targets.add((PARENT_ROOT / cleaned).resolve())
    return targets


def normalize_generated_path(raw_path: str, allowed_root_paths: list[str] | None = None) -> Path:
    """
    Converts an AI-generated path to an absolute path under PARENT_ROOT/app/, PARENT_ROOT/tests/,
    or an explicitly allowed product-root file.
    """
    p = raw_path.strip()

    if p.startswith("/"):
        raise ValueError(f"Absolute paths are forbidden: {raw_path}")

    # Strip one or more leading ../
    p = re.sub(r"^(\.\./)+", "", p)

    # Collapse duplicate directory prefixes: app/app/ → app/, tests/tests/ → tests/
    for prefix in ("app", "tests"):
        p = re.sub(rf"^{prefix}/{prefix}/", f"{prefix}/", p)

    # Strip any leading path components that precede app/ or tests/.
    # Handles LLM-hallucinated prefixes like Factory_project_claude/app/...
    parts = Path(p).parts
    for i, part in enumerate(parts):
        if part in ("app", "tests"):
            p = str(Path(*parts[i:]))
            break
    else:
        if parts:
            tail = parts[-1]
            if tail in ROOT_WRITE_ALLOWLIST:
                p = tail

    result = (PARENT_ROOT / p).resolve()

    allowed = [(PARENT_ROOT / "app").resolve(), (PARENT_ROOT / "tests").resolve()]
    allowed_roots = _allowed_root_targets(allowed_root_paths)
    if not any(str(result).startswith(str(d) + "/") or result == d for d in allowed) and result not in allowed_roots:
        raise ValueError(
            f"Path '{raw_path}' resolves outside allowed story paths: {result}"
        )

    return result


def resolve_namespace_collision(target: Path) -> bool:
    """Prevent Python namespace collisions between same-named package dirs and .py files.

    Writing X/__init__.py: removes X.py if it exists (package shadows the file).
    Writing X.py: skips the write if X/__init__.py already exists (package wins).
    Returns True to proceed, False to skip.
    """
    from agile_agent_factory.tools.logger import log

    if target.name == "__init__.py":
        shadow = target.parent.parent / (target.parent.name + ".py")
        if shadow.exists():
            log(f"Namespace collision: removing {shadow} (shadowed by package {target.parent}/)")
            shadow.unlink()
    else:
        pkg_init = target.parent / target.stem / "__init__.py"
        if pkg_init.exists():
            log(f"Namespace collision: skipping {target} — {target.parent / target.stem}/ package already owns this namespace")
            return False
    return True
