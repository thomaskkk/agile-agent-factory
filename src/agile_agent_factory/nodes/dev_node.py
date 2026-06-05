"""Developer node + code-generation helpers.

dev_node generates code for a single story (via aider when available, else
LLM-direct). The codegen helpers (_generate_code_with_llm, _correct_code,
_extract_error_summary, _load_dev_context) are shared with test_node.
"""

from __future__ import annotations

import json

from agile_agent_factory.config import PRODUCT_ROOT, DEV_MODEL, bp_task_path
from agile_agent_factory.tools.jira_client import JiraClient
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded, call_llm_json
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.path_utils import normalize_generated_path
from agile_agent_factory.state import PipelineState
from agile_agent_factory.nodes.helpers import _active_story, _safe_transition, raise_quota_interrupt

_FILE_FALLBACK = [
    {"path": "app/__init__.py", "content": ""},
    {"path": "tests/__init__.py", "content": ""},
]


def _load_dev_context(story_key: str) -> str:
    task_path = bp_task_path(story_key)
    return task_path.read_text() if task_path.exists() else ""


def _write_generated_files(files: list) -> list[str]:
    written: list[str] = []
    for f in files:
        try:
            target = normalize_generated_path(f["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.get("content", ""))
            log(f"Wrote: {target}")
            written.append(f["path"])
        except (ValueError, KeyError) as e:
            log(f"Skipped invalid path {f.get('path', '?')}: {e}")
    return written


def _generate_code_with_llm(blueprint: str, review_feedback: str = "", model: str | None = None) -> None:
    if review_feedback:
        system = (
            "You are a senior Python developer fixing a code review rejection. "
            "Return a JSON list of ONLY the files that must change to fix the rejection. "
            "Make minimal targeted changes. Do NOT rewrite unrelated files. "
            "Use only paths starting with app/ or tests/. "
            "Never use absolute paths or nested app/app/ paths."
        )
        prompt = f"""Fix the following code review rejection. Return JSON only — only the files that must change:
[
  {{"path": "app/module.py", "content": "# full corrected content"}},
  ...
]

Reviewer rejection (you MUST fix this):
{review_feedback}

Architecture context (read-only — do not re-implement from scratch):
{blueprint}
"""
    else:
        system = (
            "You are a senior Python developer implementing a feature from a technical blueprint. "
            "Return a JSON list of files to write. Use only paths starting with app/ or tests/. "
            "Never use absolute paths or nested app/app/ paths."
        )
        prompt = f"""Implement the following blueprint. Return JSON only:
[
  {{"path": "app/module.py", "content": "# full file content here"}},
  ...
]

Blueprint:
{blueprint}
"""
    files = call_llm_json(prompt, system=system, fallback=_FILE_FALLBACK, model=model)
    _write_generated_files(files)


def _correct_code(blueprint: str, traceback: str, model: str | None = None) -> list[str]:
    """Ask the LLM to fix failing tests. Returns list of written file paths, or []."""
    generated: dict[str, str] = {}
    for target_dir in ("app", "tests"):
        d = PRODUCT_ROOT / target_dir
        if d.exists():
            for f in sorted(d.rglob("*.py")):
                rel = str(f.relative_to(PRODUCT_ROOT))
                generated[rel] = f.read_text()

    files_block = "\n\n".join(
        f"### {path}\n```python\n{content[:6000]}\n```"
        for path, content in list(generated.items())[:20]
    )

    system = (
        "You are a Python developer fixing a failing test suite. "
        "Your ENTIRE response must be ONLY a valid JSON array — "
        "no other text, no explanation, no markdown fencing outside the array. "
        'Each element: {"path": "app/…", "content": "full file content"}. '
        "Return ONLY the files you must change to fix the error — do NOT return files "
        "that are already correct. Return the full content of each file you do change. "
        "Use only paths starting with app/ or tests/. "
        "Do NOT create requirements.txt, setup.py, setup.cfg, pyproject.toml or other "
        "config/dependency files — dependencies are managed separately."
    )
    prompt = f"""[
  {{"path": "app/file_to_fix.py", "content": "FULL CORRECTED CONTENT HERE"}}
]

Replace the template above with ONLY the files that must change to fix the failing
tests below. Do not include unchanged files. Do not add any text before or after the JSON array.

Traceback:
{traceback}

Current source files:
{files_block}
"""
    try:
        files = call_llm_json(prompt, system=system, model=model, prefill="[")
    except json.JSONDecodeError:
        log("LLM correction produced unparseable response — no files written.")
        return []
    if isinstance(files, dict):
        files = [files]
    elif isinstance(files, list):
        # Filter out any non-dict elements (stray strings, nested lists, etc.)
        files = [f for f in files if isinstance(f, dict)]
    else:
        log(f"LLM correction returned unexpected type ({type(files).__name__}) — no files written.")
        return []
    if not files:
        log("LLM correction returned no usable file entries — no files written.")
        return []
    written = _write_generated_files(files)
    log(f"Correction applied: {len(written)} file(s) updated.")
    return written


def _extract_error_summary(output: str) -> str:
    prefixes = ("FAILED ", "ERROR ", "AssertionError", "assert ", "E   ", "E ")
    lines = []
    for line in output.splitlines():
        if any(line.startswith(p) for p in prefixes):
            lines.append(line.rstrip())
            if len(lines) >= 3:
                break
    summary = "\n".join(lines)
    return summary[:400] if summary else output[-400:].strip()


def dev_node(state: PipelineState) -> dict:
    """Developer agent: generate code for ONE story."""
    jira = JiraClient()
    sk, story = _active_story(state)
    log(f"Dev: generating code for {sk}.")

    is_rework = story.get("column") == "code_review"

    if not is_rework:
        _safe_transition(jira, sk, "Development")
        for ek in state.get("epic_keys", []):
            _safe_transition(jira, ek, "Development")
        for subtask_key in (story.get("subtasks") or state.get("subtasks", {})).values():
            _safe_transition(jira, subtask_key, "Development")

    blueprint = _load_dev_context(sk)
    review_feedback = story.get("review_rejection_reason", "")

    try:
        from agile_agent_factory.tools.aider_client import is_available, run_task
        if is_available():
            log("Using aider for code generation.")
            run_task(
                "Implement the product described in the blueprint. "
                "Write code to app/ and tests to tests/.",
                blueprint,
                review_feedback=review_feedback,
            )
        else:
            log("Aider unavailable — using LLM-direct code generation.")
            _generate_code_with_llm(blueprint, review_feedback=review_feedback, model=DEV_MODEL or None)
    except LLMQuotaExceeded as e:
        raise_quota_interrupt(jira, sk, e)
        _generate_code_with_llm(blueprint, review_feedback=review_feedback, model=DEV_MODEL or None)

    if is_rework:
        return {
            "stories": {sk: {
                "review_status": "pending_review",
                "review_rejection_reason": "",
            }},
        }
    return {
        "stories": {sk: {"column": "testing", "retries": 0, "correction_failures": 0}},
    }
