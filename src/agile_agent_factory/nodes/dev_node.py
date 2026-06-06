"""Developer node + code-generation helpers.

dev_node generates code for a single story (via aider when available, else
LLM-direct). The codegen helpers (_generate_code_with_llm, _correct_code,
_extract_error_summary, _load_dev_context) are shared with test_node.
"""

from __future__ import annotations

import json
from pathlib import Path

from agile_agent_factory.config import PRODUCT_ROOT, DEV_MODEL, LLM_MAX_TOKENS, bp_task_path
from agile_agent_factory.tools.jira_client import JiraClient
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded, call_llm, call_llm_json
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.path_utils import normalize_generated_path
from agile_agent_factory.tools.workflow import WorkflowState
from agile_agent_factory.state import PipelineState
from agile_agent_factory.nodes.helpers import _active_story, _safe_transition, raise_quota_interrupt

_FILE_FALLBACK = [
    {"path": "app/__init__.py", "content": ""},
    {"path": "tests/__init__.py", "content": ""},
]


def _load_dev_context(story_key: str) -> str:
    task_path = bp_task_path(story_key)
    return task_path.read_text() if task_path.exists() else ""


def _resolve_namespace_collision(target: Path) -> bool:
    """Prevent Python namespace collisions between same-named package dirs and .py files.

    Writing X/__init__.py: removes X.py if it exists (package shadows the file).
    Writing X.py: skips the write if X/__init__.py already exists (package wins).
    Returns True to proceed, False to skip.
    """
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


def _write_generated_files(files: list, write_scope: list[str] | None = None) -> list[str]:
    written: list[str] = []
    for f in files:
        try:
            target = normalize_generated_path(f["path"])
            path_str = f["path"]
            if write_scope:
                if path_str not in write_scope:
                    log(f"Dev: skipping out-of-scope write: {path_str} (not in write_scope)")
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not _resolve_namespace_collision(target):
                continue
            target.write_text(f.get("content", ""))
            log(f"Wrote: {target}")
            written.append(path_str)
        except (ValueError, KeyError) as e:
            log(f"Skipped invalid path {f.get('path', '?')}: {e}")
    return written


def _generate_code_with_llm(
    blueprint: str,
    review_feedback: str = "",
    write_scope: list[str] | None = None,
    model: str | None = None,
) -> None:
    if review_feedback:
        scope_instruction = ""
        if write_scope:
            scope_instruction = (
                f"\nYou MUST fix the files in the write scope: {', '.join(write_scope)}. "
                "Only change other files if strictly required by the fix. "
                "Do not touch files outside the write scope unless they directly cause this rejection.\n"
            )
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
{scope_instruction}
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
    _write_generated_files(files, write_scope=write_scope)


def _is_truncated_json(raw: str) -> bool:
    """Heuristic: detect if the LLM's JSON array response was cut short."""
    stripped = raw.strip()
    if not stripped:
        return False
    # Unterminated array: starts with [ but last non-whitespace char is not ]
    if stripped[0] == "[" and stripped[-1] != "]":
        return True
    # Near max-token limit (rough estimate: 4 chars per token)
    if len(stripped) >= LLM_MAX_TOKENS * 4 * 0.95:
        return True
    return False


def _parse_correction_response(raw: str) -> list | None:
    """Multi-strategy JSON parser for correction responses. Returns None on total failure."""
    import re as _re

    fence = _re.search(r"```(?:json)?\s*([\s\S]*?)```", raw.strip())
    cleaned = fence.group(1).strip() if fence else raw.strip()

    # Direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return [f for f in result if isinstance(f, dict)]
    except json.JSONDecodeError:
        pass

    # Outermost array/object extraction
    for sc, ec in (("[", "]"), ("{", "}")):
        s = cleaned.find(sc)
        e = cleaned.rfind(ec)
        if s != -1 and e > s:
            try:
                result = json.loads(cleaned[s:e + 1])
                if isinstance(result, dict):
                    return [result]
                if isinstance(result, list):
                    return [f for f in result if isinstance(f, dict)]
            except json.JSONDecodeError:
                pass

    # json-repair as last resort
    try:
        from json_repair import repair_json
        repaired = repair_json(cleaned, return_objects=True)
        if isinstance(repaired, list) and repaired:
            return [f for f in repaired if isinstance(f, dict)]
    except Exception:
        pass

    return None


def _correct_code(blueprint: str, traceback: str, model: str | None = None, write_scope: list[str] | None = None) -> tuple[str, list[str]]:
    """Ask the LLM to fix failing tests.

    Returns (status, written_files):
        "ok"        — correction produced and wrote usable files
        "truncated" — LLM response was truncated; caller should use strategy-retry budget
        "empty"     — LLM returned nothing usable; caller should use correction-failure budget
    """
    from agile_agent_factory.nodes.failure_recovery import files_from_traceback

    generated: dict[str, str] = {}
    for target_dir in ("app", "tests"):
        d = PRODUCT_ROOT / target_dir
        if d.exists():
            for f in sorted(d.rglob("*.py")):
                rel = str(f.relative_to(PRODUCT_ROOT))
                generated[rel] = f.read_text()

    # Prioritize traceback-named files: show at FULL content so they're never truncated.
    tb_files = files_from_traceback(traceback)
    tb_set = set(tb_files)
    priority_fences = [
        f"### {path}\n```python\n{generated[path]}\n```"
        for path in tb_files if path in generated
    ]
    rest_budget = max(0, 20 - len(priority_fences))
    rest_fences = [
        f"### {path}\n```python\n{content[:6000]}\n```"
        for path, content in ((p, c) for p, c in generated.items() if p not in tb_set)
    ][:rest_budget]
    files_block = "\n\n".join(priority_fences + rest_fences)

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
        raw = call_llm(prompt, system=system, model=model, prefill="[")
    except LLMQuotaExceeded:
        raise

    is_truncated = _is_truncated_json(raw)
    files = _parse_correction_response(raw)

    if files is None:
        if is_truncated:
            log("Correction response truncated — retrying with reduced scope (worst file only).")
            worst = tb_files[0] if tb_files else (list(generated.keys())[0] if generated else None)
            if worst and worst in generated:
                reduced_block = f"### {worst}\n```python\n{generated[worst]}\n```"
                reduced_prompt = f"""[
  {{"path": "app/file_to_fix.py", "content": "FULL CORRECTED CONTENT HERE"}}
]

Fix ONLY the single most critical file causing the failure below.

Traceback:
{traceback}

Current source files:
{reduced_block}
"""
                try:
                    raw2 = call_llm(reduced_prompt, system=system, model=model, prefill="[")
                    files = _parse_correction_response(raw2)
                except (LLMQuotaExceeded, Exception) as e:
                    log(f"Reduced-scope retry failed: {e}")
                    files = None
            if files is None:
                log("Truncation retry also failed — flagging as strategy retry.")
                return ("truncated", [])
        else:
            log("LLM correction produced unparseable response — no files written.")
            return ("empty", [])

    files = [f for f in files if isinstance(f, dict)]
    if not files:
        log("LLM correction returned no usable file entries — no files written.")
        return ("empty", [])

    written = _write_generated_files(files, write_scope=write_scope)
    log(f"Correction applied: {len(written)} file(s) updated.")
    return ("ok", written)


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
        _safe_transition(jira, sk, WorkflowState.DEVELOPMENT)
        for ek in state.get("epic_keys", []):
            _safe_transition(jira, ek, WorkflowState.DEVELOPMENT)
        for subtask_key in (story.get("subtasks") or state.get("subtasks", {})).values():
            _safe_transition(jira, subtask_key, WorkflowState.DEVELOPMENT)

    blueprint = _load_dev_context(sk)
    review_feedback = story.get("review_rejection_reason", "")

    # Derive write_scope from test_contract so rework is targeted to owned files
    tc = story.get("test_contract", {})
    write_scope: list[str] = []
    if tc and is_rework:
        if tc.get("test_file"):
            write_scope.append(tc["test_file"])
        for imp in (tc.get("target_imports") or []):
            if isinstance(imp, str) and imp.strip():
                path_str = imp.split("import")[-1].strip().replace(".", "/") + ".py"
                if path_str not in write_scope:
                    write_scope.append(path_str)

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
            _generate_code_with_llm(
                blueprint,
                review_feedback=review_feedback,
                write_scope=write_scope or None,
                model=DEV_MODEL or None,
            )
    except LLMQuotaExceeded as e:
        raise_quota_interrupt(jira, sk, e)
        _generate_code_with_llm(
            blueprint,
            review_feedback=review_feedback,
            write_scope=write_scope or None,
            model=DEV_MODEL or None,
        )

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
