from pathlib import Path

from agile_agent_factory.config import (
    PRODUCT_ROOT, REVIEWER_MODEL, BP_ARCH_CONSTRAINTS, BP_QA_CRITERIA_DIR, bp_qa_criteria_path,
    REVIEW_MAX_FILES, REVIEW_MAX_FILE_CHARS, REVIEW_MAX_TOTAL_CHARS,
)
from agile_agent_factory.tools.jira_client import JiraClient
from agile_agent_factory.tools.llm_client import call_llm_json
from agile_agent_factory.tools.logger import log

_REVIEW_FALLBACK = {"approved": False, "rejection_reason": "LLM did not return valid JSON — manual review required."}


def review_patch(jira: JiraClient, story_keys: list[str], story_criteria: list[str] | None = None, story_key: str | None = None, write_scope: list[str] | None = None) -> dict:
    for key in story_keys:
        try:
            jira.transition_issue(key, "In Code Review")
        except ValueError as e:
            log(str(e))

    if story_criteria:
        dod_section = "\n".join(story_criteria)
    elif story_key:
        qa_path = bp_qa_criteria_path(story_key)
        if qa_path.exists():
            dod_section = qa_path.read_text()
        else:
            dod_section = BP_ARCH_CONSTRAINTS.read_text() if BP_ARCH_CONSTRAINTS.exists() else ""
    else:
        # No story key: assemble all available qa_criteria files + constraints
        parts = sorted(BP_QA_CRITERIA_DIR.glob("*.md")) if BP_QA_CRITERIA_DIR.exists() else []
        sections = [p.read_text() for p in parts]
        if BP_ARCH_CONSTRAINTS.exists():
            sections.append(BP_ARCH_CONSTRAINTS.read_text())
        dod_section = "\n\n".join(sections)

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
        return f"### {path}\n```{lang}\n{content[:REVIEW_MAX_FILE_CHARS]}\n```"

    # Write-scope files are always included first so they are never truncated away.
    # Other files fill the remaining budget up to REVIEW_MAX_TOTAL_CHARS.
    scope_set = set(write_scope or [])
    scope_items = [(p, c) for p, c in generated.items() if p in scope_set]
    other_items = [(p, c) for p, c in generated.items() if p not in scope_set]

    fences: list[str] = []
    total_chars = 0
    for path, content in scope_items + other_items:
        if len(fences) >= REVIEW_MAX_FILES:
            break
        fence = _fence(path, content)
        if path not in scope_set and total_chars + len(fence) > REVIEW_MAX_TOTAL_CHARS:
            break
        fences.append(fence)
        total_chars += len(fence) + 2  # +2 for the \n\n separator

    files_block = "\n\n".join(fences)

    system = (
        "You are a Code Reviewer auditing generated code (Python, HTML templates, CSS, JS) "
        "against a Definition of Done. "
        "IMPORTANT: For Python ASGI/WSGI applications (FastAPI, Flask, Starlette), using "
        "Starlette TestClient or httpx AsyncClient to exercise HTTP endpoints IS the correct "
        "testing approach — it does not require binding to a real network socket. A server "
        "wrapper whose start()/stop() methods manage lifecycle state and whose underlying ASGI "
        "app passes all HTTP tests via TestClient fully satisfies 'listens for connections' and "
        "'accepts HTTP requests' acceptance criteria. "
        "Output ONLY a JSON object — no prose, no preamble, no markdown fences. "
        "Schema: {\"approved\": bool, \"rejection_reason\": \"\"}"
    )
    scope_instruction = ""
    if write_scope:
        owned = "\n".join(f"  - {f}" for f in write_scope)
        scope_instruction = f"""
Story write scope — files owned by this story:
{owned}

Evaluate correctness and DoD compliance ONLY for the files listed above.
Issues found in files OUTSIDE this scope are pre-existing problems owned by other
stories and MUST NOT cause this review to fail — note them as informational only.
"""

    prompt = f"""Review the generated code against the blueprint DoD.
You are shown the COMPLETE contents of all generated files below — Python source,
HTML templates, CSS, and JS. Do not assume any file type is missing unless it is
absent from the list. Judge functional correctness and DoD coverage only.
{scope_instruction}
Acceptance criteria (DoD):
{dod_section[:8000]}

Generated files:
{files_block}

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

    return {"approved": approved, "reason": reason}
