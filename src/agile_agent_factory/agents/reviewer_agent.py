from pathlib import Path

from agile_agent_factory.agents.contract import AgentResult
from agile_agent_factory.config import (
    PRODUCT_ROOT, BP_ARCH_CONSTRAINTS, BP_QA_CRITERIA_DIR, bp_qa_criteria_path,
    REVIEW_MAX_FILES, REVIEW_MAX_FILE_CHARS, REVIEW_MAX_TOTAL_CHARS,
)
from agile_agent_factory.tools.jira_client import JiraClient
from agile_agent_factory.tools.jira_facade import JiraFacade
from agile_agent_factory.tools.llm_adapters.reviewer import generate_review
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.workflow import WorkflowState

_REVIEW_FALLBACK = {"approved": False, "rejection_reason": "LLM did not return valid JSON — manual review required."}


def _truncate_for_review(content: str, char_limit: int) -> tuple[str, bool]:
    """Trim to a prompt budget without ending mid-line when possible."""
    if len(content) <= char_limit:
        return content, False

    snippet = content[:char_limit]
    newline = snippet.rfind("\n")
    if newline > 0:
        snippet = snippet[:newline]
    return snippet.rstrip(), True


def review_patch(jira: JiraClient, story_keys: list[str], story_criteria: list[str] | None = None, story_key: str | None = None, write_scope: list[str] | None = None) -> AgentResult:
    facade = JiraFacade(jira)
    facade.move_all(story_keys, WorkflowState.IN_CODE_REVIEW)

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
        return AgentResult(success=False, payload={"approved": False, "reason": "No generated files found."})

    def _fence(path: str, content: str) -> str:
        lang = _LANG_MAP.get(Path(path).suffix.lower(), "")
        snippet, truncated = _truncate_for_review(content, REVIEW_MAX_FILE_CHARS)
        header = f"### {path}"
        if truncated:
            header += f" (TRUNCATED: showing first {len(snippet)} of {len(content)} chars)"
        block = f"{header}\n```{lang}\n{snippet}\n```"
        if truncated:
            block += "\n[TRUNCATED FOR REVIEW BUDGET]"
        return block

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

    result = generate_review(dod_section, files_block, write_scope, _REVIEW_FALLBACK)
    approved = result.get("approved", False)
    reason = result.get("rejection_reason", "")

    if approved:
        log("Code review: APPROVED.")
        facade.move_all(story_keys, WorkflowState.TO_QA)
    else:
        log(f"Code review: REJECTED — {reason}")

    return AgentResult(success=approved, payload={"approved": approved, "reason": reason})
