from pathlib import Path

from agile_agent_factory.config import PRODUCT_ROOT, BLUEPRINT_PATH, REVIEWER_MODEL
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_doc
from agile_agent_factory.tools.llm_client import call_llm_json
from agile_agent_factory.tools.logger import log

_REVIEW_FALLBACK = {"approved": False, "rejection_reason": "LLM did not return valid JSON — manual review required."}

MAX_REVIEW_FILES = 30
MAX_REVIEW_FILE_CHARS = 8000      # generous per-file cap; small products are well under this
MAX_REVIEW_TOTAL_CHARS = 50000    # overall prompt budget for the files block


def review_patch(jira: JiraClient, story_keys: list[str]) -> dict:
    for key in story_keys:
        try:
            jira.transition_issue(key, "In Code Review")
        except ValueError as e:
            log(str(e))

    blueprint = BLUEPRINT_PATH.read_text() if BLUEPRINT_PATH.exists() else ""

    # Known binary/noise extensions to skip — everything else is attempted as UTF-8 text.
    _BINARY_EXTENSIONS = {
        ".pyc", ".pyo", ".pyd",
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp", ".tiff",
        ".woff", ".woff2", ".ttf", ".eot", ".otf",
        ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
        ".exe", ".dll", ".so", ".dylib",
        ".db", ".sqlite", ".sqlite3",
        ".mp3", ".mp4", ".wav", ".avi", ".mov",
    }
    # Directories that are never part of the generated product.
    _SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".mypy_cache", ".pytest_cache"}

    _LANG_MAP = {".py": "python", ".js": "javascript", ".ts": "typescript",
                 ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
                 ".json": "json", ".html": "html", ".css": "css", ".md": "markdown"}

    generated: dict[str, str] = {}
    for target_dir in ("app", "tests"):
        d = PRODUCT_ROOT / target_dir
        if d.exists():
            for f in sorted(d.rglob("*")):
                if not f.is_file():
                    continue
                if any(part in _SKIP_DIRS for part in f.parts):
                    continue
                if f.suffix.lower() in _BINARY_EXTENSIONS:
                    continue
                try:
                    generated[str(f.relative_to(PRODUCT_ROOT))] = f.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    pass  # genuinely binary or unreadable — skip silently

    if not generated:
        log("No generated files found for review.")
        return {"approved": False, "reason": "No generated files found."}

    def _fence(path: str, content: str) -> str:
        lang = _LANG_MAP.get(Path(path).suffix.lower(), "")
        return f"### {path}\n```{lang}\n{content[:MAX_REVIEW_FILE_CHARS]}\n```"

    files_block = "\n\n".join(
        _fence(path, content)
        for path, content in list(generated.items())[:MAX_REVIEW_FILES]
    )
    system = (
        "You are a Code Reviewer auditing generated code (Python, HTML templates, CSS, JS) "
        "against a Definition of Done. "
        "Output ONLY a JSON object — no prose, no preamble, no markdown fences. "
        "Schema: {\"approved\": bool, \"rejection_reason\": \"\"}"
    )
    prompt = f"""Review the generated code against the blueprint DoD.
You are shown the COMPLETE contents of all generated files below — Python source,
HTML templates, CSS, and JS. Do not assume any file type is missing unless it is
absent from the list. Judge functional correctness and DoD coverage only.

Blueprint (DoD summary):
{blueprint[:8000]}

Generated files:
{files_block[:MAX_REVIEW_TOTAL_CHARS]}

Output ONLY valid JSON, nothing else:
{{"approved": true, "rejection_reason": ""}}
"""
    result = call_llm_json(prompt, system=system, fallback=_REVIEW_FALLBACK, model=REVIEWER_MODEL or None)
    approved = result.get("approved", False)
    reason = result.get("rejection_reason", "")

    if approved:
        log("Code review: APPROVED.")
        for key in story_keys:
            try:
                jira.transition_issue(key, "To QA")
            except ValueError as e:
                log(str(e))
    else:
        log(f"Code review: REJECTED — {reason}")
        for key in story_keys:
            jira.add_comment_adf(key, make_adf_doc(f"Code review rejected:\n{reason}"))
            try:
                jira.transition_issue(key, "Development")
            except ValueError as e:
                log(str(e))

    return {"approved": approved, "reason": reason}
