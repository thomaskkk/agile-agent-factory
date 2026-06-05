import re
from pathlib import Path

from agile_agent_factory.config import PRODUCT_ROOT as PARENT_ROOT


def normalize_generated_path(raw_path: str) -> Path:
    """
    Converts an AI-generated path to an absolute path under PARENT_ROOT/app/ or PARENT_ROOT/tests/.
    Rejects absolute paths and anything that resolves outside the two allowed directories.
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

    result = (PARENT_ROOT / p).resolve()

    allowed = [(PARENT_ROOT / "app").resolve(), (PARENT_ROOT / "tests").resolve()]
    if not any(str(result).startswith(str(d) + "/") or result == d for d in allowed):
        raise ValueError(
            f"Path '{raw_path}' resolves outside app/ or tests/: {result}"
        )

    return result
