"""Developer node + code-generation helpers.

dev_node generates code for a single story (via aider when available, else
LLM-direct). The codegen helpers (_generate_code_with_llm, _correct_code,
_extract_error_summary, _load_dev_context) are shared with test_node.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agile_agent_factory.config import PRODUCT_ROOT, DEV_MODEL, LLM_MAX_TOKENS, bp_task_path
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_doc
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded, call_llm, call_llm_json
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.path_utils import ROOT_WRITE_ALLOWLIST, normalize_generated_path, resolve_namespace_collision
from agile_agent_factory.tools.workflow import WorkflowState
from agile_agent_factory.state import PipelineState
from agile_agent_factory.nodes.helpers import (
    _active_story,
    _changed_product_files,
    _path_in_write_scope,
    derive_story_write_scope,
    _restore_product_files,
    _safe_transition,
    _snapshot_product_files,
    _write_scope_violations,
    raise_quota_interrupt,
)

_FILE_FALLBACK = [
    {"path": "app/__init__.py", "content": ""},
    {"path": "tests/__init__.py", "content": ""},
]


def _load_dev_context(story_key: str) -> str:
    task_path = bp_task_path(story_key)
    return task_path.read_text() if task_path.exists() else ""


def _guard_generation_writes(
    before_snapshot: dict[str, bytes],
    write_scope: list[str] | None,
    *,
    rollback_all: bool = False,
) -> tuple[list[str], list[str]]:
    """Return (changed_files, violations); roll back when a generator strays out of scope."""
    changed = _changed_product_files(before_snapshot)
    violations = _write_scope_violations(changed, write_scope)
    if violations:
        _restore_product_files(before_snapshot, changed if rollback_all else violations)
        log("Generation wrote outside write_scope; rolled back: " + ", ".join(violations))
        return _changed_product_files(before_snapshot), violations
    return changed, []


def _write_generated_files(files: list, write_scope: list[str] | None = None) -> list[str]:
    written: list[str] = []
    for f in files:
        if not isinstance(f, dict):
            log(f"Skipped malformed generated entry of type {type(f).__name__}.")
            continue
        path_str = f.get("path", "")
        if write_scope and not _path_in_write_scope(path_str, write_scope):
            log(f"Dev: skipping out-of-scope write: {path_str} (not in write_scope)")
            continue
        try:
            target = normalize_generated_path(path_str, allowed_root_paths=write_scope)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not resolve_namespace_collision(target):
                continue
            new_content = f.get("content", "")
            if target.exists() and target.is_file():
                try:
                    current_content = target.read_text(encoding="utf-8")
                except OSError:
                    current_content = None
                if current_content == new_content:
                    log(f"Skipped unchanged generated file: {path_str}")
                    continue
            target.write_text(new_content, encoding="utf-8")
            log(f"Wrote: {target}")
            written.append(path_str)
        except (ValueError, KeyError) as e:
            log(f"Skipped invalid path {f.get('path', '?')}: {e}")
    return written


def _normalize_generation_payload(payload) -> list[dict]:
    """Coerce LLM JSON output into a list of file dicts, skipping malformed entries."""
    if isinstance(payload, dict):
        if "files" in payload and isinstance(payload["files"], list):
            payload = payload["files"]
        else:
            payload = [payload]

    if not isinstance(payload, list):
        log(f"LLM generation returned unsupported JSON shape: {type(payload).__name__}")
        return []

    normalized = [entry for entry in payload if isinstance(entry, dict)]
    skipped = len(payload) - len(normalized)
    if skipped:
        log(f"LLM generation skipped {skipped} malformed file entr{'' if skipped == 1 else 'ies'}.")
    return normalized


def _build_write_scope_file_context(write_scope: list[str] | None) -> str:
    """Return existing exact-scope file contents to help rework prompts modify the right file."""
    if not write_scope:
        return ""

    sections: list[str] = []
    for scoped_path in write_scope:
        if scoped_path.endswith("/"):
            continue
        rel_path = scoped_path.strip().lstrip("./")
        if not rel_path:
            continue
        if "/" not in rel_path and rel_path not in ROOT_WRITE_ALLOWLIST:
            continue
        target = PRODUCT_ROOT / rel_path
        if not target.exists() or not target.is_file():
            continue
        try:
            content = target.read_text(encoding="utf-8")
        except OSError:
            continue
        suffix = target.suffix.lstrip(".")
        fence_lang = suffix if suffix else "text"
        sections.append(f"### {rel_path}\n```{fence_lang}\n{content[:8000]}\n```")

    if not sections:
        return ""
    return "\n\nCurrent write-scope files:\n" + "\n\n".join(sections[:12]) + "\n"


def _should_try_readme_rework_fallback(review_feedback: str, write_scope: list[str] | None) -> bool:
    if not review_feedback or not write_scope or "README.md" not in write_scope:
        return False
    return bool(re.search(r"\bREADME\b|\bsetup\b|\bstartup\b|\brun\b|\binstall", review_feedback, re.IGNORECASE))


def _augment_noop_rework_feedback(review_feedback: str, write_scope: list[str] | None) -> str:
    scope_note = f"Owned write scope: {', '.join(write_scope)}." if write_scope else ""
    return (
        f"{review_feedback}\n\n"
        "Factory note: the previous rework attempt produced no material file changes. "
        "You MUST materially change at least one file in the owned write scope. "
        "Returning identical content counts as failure.\n"
        f"{scope_note}"
    ).strip()


def _generate_readme_rework_guarded(write_scope: list[str] | None) -> list[str]:
    """Regenerate README as a targeted fallback for README-only review loops."""
    from agile_agent_factory.agents.readme_agent import generate_readme

    before_snapshot = _snapshot_product_files()
    generate_readme({})
    changed_files, violations = _guard_generation_writes(
        before_snapshot,
        write_scope,
        rollback_all=False,
    )
    if violations:
        if changed_files:
            log(
                "README fallback exceeded write_scope; kept in-scope files after rollback: "
                + ", ".join(changed_files)
            )
        else:
            log("README fallback produced no in-scope files after write_scope rollback.")
    return changed_files


def _generate_code_with_llm(
    blueprint: str,
    review_feedback: str = "",
    write_scope: list[str] | None = None,
    model: str | None = None,
) -> None:
    if review_feedback:
        current_files_block = _build_write_scope_file_context(write_scope)
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
            "Use only paths that are explicitly listed in the write scope. "
            "Those paths may be under app/, under tests/, or product-root files such as "
            "README.md, requirements.txt, pyproject.toml, main.py, app.py, or wsgi.py "
            "when those exact files appear in scope. "
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
{current_files_block}
Architecture context (read-only — do not re-implement from scratch):
{blueprint}
"""
    else:
        system = (
            "You are a senior Python developer implementing a feature from a technical blueprint. "
            "Return a JSON list of files to write. Use only paths that are explicitly listed "
            "in the write scope. Those paths may be under app/, under tests/, or product-root "
            "files such as README.md, requirements.txt, pyproject.toml, main.py, app.py, or "
            "wsgi.py when those exact files appear in scope. "
            "Never use absolute paths or nested app/app/ paths."
        )
        scope_instruction = (
            "\nYou MUST only create or modify files in this write_scope:\n"
            + "\n".join(f"  - {p}" for p in write_scope)
            + "\nDo NOT create any files outside this list."
        ) if write_scope else ""
        prompt = f"""Implement the following blueprint. Return JSON only:
[
  {{"path": "app/module.py", "content": "# full file content here"}},
  ...
]
{scope_instruction}
Blueprint:
{blueprint}
"""
    payload = call_llm_json(prompt, system=system, fallback=_FILE_FALLBACK, model=model)
    files = _normalize_generation_payload(payload)
    if not files:
        log("LLM generation returned no usable file entries — no files written.")
        return
    _write_generated_files(files, write_scope=write_scope)


def _generate_code_with_llm_guarded(
    blueprint: str,
    review_feedback: str = "",
    write_scope: list[str] | None = None,
    model: str | None = None,
) -> list[str]:
    """Run LLM generation and keep only in-scope filesystem effects."""
    before_snapshot = _snapshot_product_files()
    _generate_code_with_llm(
        blueprint,
        review_feedback=review_feedback,
        write_scope=write_scope,
        model=model,
    )
    changed_files, violations = _guard_generation_writes(
        before_snapshot,
        write_scope,
        rollback_all=False,
    )
    if violations:
        if changed_files:
            log(
                "LLM-direct generation exceeded write_scope; "
                "kept in-scope files after rollback: "
                + ", ".join(changed_files)
            )
        else:
            log("LLM-direct generation produced no in-scope files after write_scope rollback.")
    return changed_files


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


def _build_spec_block(test_contract: dict | None, gherkin_criteria: list[str] | None) -> str:
    """Build a highlighted spec section for the correction prompt."""
    parts = []
    if test_contract:
        funcs = test_contract.get("test_functions") or test_contract.get("expected_tests") or []
        if funcs:
            parts.append("Expected test functions:\n" + "\n".join(f"  - {f}" for f in funcs))
        imports = test_contract.get("target_imports", [])
        if imports:
            parts.append("Target imports:\n" + "\n".join(f"  - {i}" for i in imports))
        edges = test_contract.get("edge_cases", [])
        if edges:
            parts.append("Edge cases to handle:\n" + "\n".join(f"  - {e}" for e in edges))
    if gherkin_criteria:
        parts.append("Acceptance criteria:\n" + "\n".join(f"  {c}" for c in gherkin_criteria[:3]))
    return "\n\n".join(parts)


def _correct_code(
    blueprint: str,
    traceback: str,
    model: str | None = None,
    write_scope: list[str] | None = None,
    test_contract: dict | None = None,
    gherkin_criteria: list[str] | None = None,
) -> tuple[str, list[str]]:
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

    scope_clause = ""
    if write_scope:
        scope_clause = (
            f" You MUST only modify files in this write scope: {', '.join(write_scope)}."
            " Do NOT touch any other file, even if you think it would help."
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
        f"config/dependency files — dependencies are managed separately.{scope_clause}"
    )
    spec_block = _build_spec_block(test_contract, gherkin_criteria)
    spec_section = f"\nWHAT TO IMPLEMENT (from spec):\n{spec_block}\n" if spec_block else ""
    prompt = f"""[
  {{"path": "app/file_to_fix.py", "content": "FULL CORRECTED CONTENT HERE"}}
]

Replace the template above with ONLY the files that must change to fix the failing
tests below. Do not include unchanged files. Do not add any text before or after the JSON array.
{spec_section}
FAILING TESTS:
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
                except LLMQuotaExceeded:
                    raise
                except Exception as e:
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
    from langgraph.types import interrupt

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
    human_feedback = story.get("hitl_feedback", "")
    effective_review_feedback = review_feedback
    if human_feedback:
        effective_review_feedback = (
            f"{effective_review_feedback}\n\nHuman feedback after HITL:\n{human_feedback}".strip()
            if effective_review_feedback
            else f"Human feedback after HITL:\n{human_feedback}"
        )
    changed_files: list[str] = []

    write_scope = derive_story_write_scope(sk, story)

    try:
        from agile_agent_factory.tools.aider_client import is_available, run_task
        aider_available = is_available()
        if aider_available and write_scope:
            log("Aider available but bypassed because strict write_scope must be enforced.")
        if aider_available and not write_scope:
            log("Using aider for code generation.")
            before_snapshot = _snapshot_product_files()
            aider_result = run_task(
                "Implement the product described in the blueprint. "
                "Write code to app/ and tests to tests/.",
                blueprint,
                review_feedback=effective_review_feedback,
                write_scope=write_scope or None,
            )
            changed_files, violations = _guard_generation_writes(
                before_snapshot,
                write_scope or None,
                rollback_all=not aider_result.get("success", False),
            )
            if not aider_result.get("success", False):
                if changed_files:
                    _restore_product_files(before_snapshot, changed_files)
                log("Aider failed — falling back to LLM-direct code generation.")
                changed_files = _generate_code_with_llm_guarded(
                    blueprint,
                    review_feedback=effective_review_feedback,
                    write_scope=write_scope or None,
                    model=DEV_MODEL or None,
                )
            elif violations:
                log("Aider violated ownership boundaries — retrying with LLM-direct generation.")
                changed_files = _generate_code_with_llm_guarded(
                    blueprint,
                    review_feedback=effective_review_feedback,
                    write_scope=write_scope or None,
                    model=DEV_MODEL or None,
                )
        else:
            log("Using LLM-direct code generation.")
            changed_files = _generate_code_with_llm_guarded(
                blueprint,
                review_feedback=effective_review_feedback,
                write_scope=write_scope or None,
                model=DEV_MODEL or None,
            )
    except LLMQuotaExceeded as e:
        patch = raise_quota_interrupt(jira, sk, e, state=state)
        if patch:
            return patch

    if is_rework:
        if not changed_files:
            log(f"Dev: rework for {sk} produced no material changes.")
            if _should_try_readme_rework_fallback(effective_review_feedback, write_scope or None):
                log("Dev: retrying rework with README fallback.")
                changed_files = _generate_readme_rework_guarded(write_scope or None)
            if not changed_files:
                log("Dev: retrying rework with stronger no-op instructions.")
                changed_files = _generate_code_with_llm_guarded(
                    blueprint,
                    review_feedback=_augment_noop_rework_feedback(
                        effective_review_feedback or review_feedback,
                        write_scope or None,
                    ),
                    write_scope=write_scope or None,
                    model=DEV_MODEL or None,
                )
            if not changed_files:
                summary = (
                    "Developer rework HITL: automated rework produced no material file changes "
                    "after multiple attempts.\n\n"
                    f"Latest review rejection:\n\n{review_feedback or '(no rejection reason provided)'}"
                )
                if human_feedback:
                    summary += f"\n\nLatest human feedback:\n\n{human_feedback}"
                jira.add_comment_adf(sk, make_adf_doc(summary))
                jira.set_flag(sk)
                interrupt({"type": "intervention", "blocking_key": sk, "source": "dev_noop_rework"})
                return {
                    "stories": {sk: {
                        "review_status": "rework_needed",
                        "review_rejection_reason": review_feedback,
                        "last_changed_files": [],
                        "hitl_type": "intervention",
                    }},
                }
        return {
            "stories": {sk: {
                "review_status": "pending_review",
                "review_rejection_reason": "",
                "hitl_feedback": "",
                "last_changed_files": changed_files,
            }},
        }
    return {
        "stories": {sk: {
            "column": "testing",
            "retries": 0,
            "correction_failures": 0,
            "failure_streak": 0,
            "last_failure_signature": "",
            "last_failure_class": "",
            "last_changed_files": changed_files,
        }},
    }
