"""Shared helpers for the LangGraph node modules.

These small utilities are used across the lifecycle nodes (pipeline.py) and the
dev/test/review node modules. They contain no node entrypoints themselves.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from langgraph.types import interrupt

from agile_agent_factory.config import PRODUCT_ROOT, bp_task_path
from agile_agent_factory.config import LLM_QUOTA_MAX_RETRIES, LLM_RETRY_BACKOFF_SECONDS
from agile_agent_factory.tools.jira_client import JiraClient
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.path_utils import ROOT_WRITE_ALLOWLIST
from agile_agent_factory.tools.workflow import WorkflowState
from agile_agent_factory.state import PipelineState


def _story_keys(state: PipelineState) -> list[str]:
    return list(state.get("stories", {}).keys())


def _active_story(state: PipelineState) -> tuple[str, dict]:
    """Return (story_key, story_dict) for the active story in this node invocation."""
    sk = state.get("active_story_key") or _story_keys(state)[0]
    story = state.get("stories", {}).get(sk, {})
    return sk, story


def _safe_transition(jira: JiraClient, key: str, target: WorkflowState) -> None:
    try:
        jira.transition_to(key, target)
    except ValueError as e:
        log(str(e))


def _to_legacy_state(state: PipelineState, story_key: str | None = None) -> dict:
    """Build a legacy flat-state dict compatible with existing agent interfaces."""
    stories = state.get("stories", {})
    story: dict = {}
    if story_key:
        story = stories.get(story_key, {})

    return {
        "status": "READY",
        "current_phase": None,
        "blocking_issue_key": None,
        "hitl_feedback": story.get("hitl_feedback", ""),
        "epic_keys": state.get("epic_keys", []),
        "story_keys": _story_keys(state),
        "subtasks": story.get("subtasks") or state.get("subtasks", {}),
        "review_retries": story.get("review_retries", state.get("review_retries", 0)),
        "has_ui": story.get("has_ui", state.get("has_ui", False)),
        "dependencies": story.get("dependencies") or state.get("dependencies", []),
        "gherkin_criteria": {story_key: story.get("gherkin_criteria", [])} if story_key else state.get("gherkin_criteria", {}),
        "ux_spec": story.get("ux_spec") or state.get("ux_spec", {}),
        "architecture": story.get("architecture") or state.get("architecture", {}),
    }


def _notify_quota(jira: JiraClient, issue_key: str | None, exc: LLMQuotaExceeded) -> None:
    from agile_agent_factory.config import JIRA_HUMAN_ACCOUNT_ID
    from agile_agent_factory.tools.jira_client import make_adf_mention_doc
    provider = getattr(exc, "provider", "unknown")
    if issue_key:
        try:
            jira.set_flag(issue_key)
            jira.add_comment_adf(
                issue_key,
                make_adf_mention_doc(
                    JIRA_HUMAN_ACCOUNT_ID,
                    f"LLM quota exceeded (provider: {provider}). "
                    "Resolve the quota limit and clear this flag to resume the pipeline.",
                ),
            )
        except Exception as notify_err:
            log(f"Failed to post quota notification on {issue_key}: {notify_err}")
    log(f"Quota exceeded ({provider}). Pipeline pausing for human intervention.")


def _story_summary(jira: JiraClient, story_key: str) -> str:
    try:
        issue = jira._request("GET", f"issue/{story_key}?fields=summary")
        return issue.get("fields", {}).get("summary", "") or ""
    except Exception as e:
        log(f"Could not fetch story summary for {story_key}: {e}")
        return ""


def _module_path_from_import(import_stmt: str) -> str | None:
    if not isinstance(import_stmt, str) or not import_stmt.strip():
        return None
    match = re.match(r"from (app(?:\.\w+)+) import", import_stmt)
    if not match:
        return None
    return match.group(1).replace(".", "/") + ".py"


def _task_file_contract_paths(story_key: str) -> list[str]:
    task_path = bp_task_path(story_key)
    if not task_path.exists():
        return []
    seen: dict[str, None] = {}
    for match in re.finditer(r"^- `([^`]+)`:", task_path.read_text(encoding="utf-8"), re.MULTILINE):
        seen.setdefault(match.group(1).strip(), None)
    return list(seen.keys())


def _write_scope_hints_from_story_task(story_key: str) -> list[str]:
    task_path = bp_task_path(story_key)
    if not task_path.exists():
        return []
    text = task_path.read_text(encoding="utf-8")
    hints: dict[str, None] = {}
    if "uvicorn main:app" in text:
        hints["main.py"] = None
    if "uvicorn app:app" in text:
        hints["app.py"] = None
    if "uvicorn wsgi:app" in text:
        hints["wsgi.py"] = None
    return list(hints.keys())


def _write_scope_hints_from_test_contract(story: dict, product_root: Path) -> list[str]:
    tc = story.get("test_contract", {}) or {}
    seen: dict[str, None] = {}

    for sample in tc.get("sample_data", []) or []:
        if isinstance(sample, dict):
            directory = sample.get("directory")
            if isinstance(directory, str) and directory.strip():
                seen.setdefault(directory.strip(), None)
            file_path = sample.get("file")
            if isinstance(file_path, str) and file_path.strip():
                seen.setdefault(file_path.strip(), None)

    test_file_rel = tc.get("test_file", "")
    if not test_file_rel:
        return list(seen.keys())

    test_file_path = product_root / test_file_rel
    if not test_file_path.exists():
        return list(seen.keys())

    try:
        source = test_file_path.read_text(encoding="utf-8")
    except OSError:
        return list(seen.keys())

    for hint in re.findall(r"\b(?:app|tests)/[A-Za-z0-9_./-]+\b", source):
        seen.setdefault(hint, None)
    for hint in re.findall(r"\b(?:README(?:\.[A-Za-z0-9]+)?|requirements\.txt|pyproject\.toml)\b", source):
        seen.setdefault(hint, None)

    return list(seen.keys())


def derive_story_write_scope(
    story_key: str,
    story: dict,
    product_root: Path | None = None,
) -> list[str]:
    """Build an exact file-level write scope from test contracts plus architecture hints.

    Base scope comes from the QA test contract (test file + target imports). For
    scaffold/setup stories that assert directory or root-file structure, expand the
    scope with directory-prefix allowances and explicit root-file hints so the story
    can satisfy its own validated readiness contract without implicitly requiring
    every architecture file in that directory to exist already.
    """
    product_root = product_root or PRODUCT_ROOT
    tc = story.get("test_contract", {}) or {}
    scope: list[str] = []

    def add(path: str | None) -> None:
        if isinstance(path, str) and path.strip() and path not in scope:
            scope.append(path)

    add(tc.get("test_file"))
    for import_stmt in tc.get("target_imports", []) or []:
        add(_module_path_from_import(import_stmt))

    architecture_paths = _task_file_contract_paths(story_key)
    for scoped_path in list(scope):
        path_obj = Path(scoped_path)
        for parent in path_obj.parents:
            parent_str = parent.as_posix()
            if parent_str in {".", ""}:
                break
            init_path = f"{parent_str}/__init__.py"
            if init_path in architecture_paths:
                add(init_path)

    hints = _write_scope_hints_from_test_contract(story, product_root)
    for task_hint in _write_scope_hints_from_story_task(story_key):
        add(task_hint)
    for hint in hints:
        normalized = hint.strip().lstrip("./")
        if not normalized:
            continue
        if normalized in architecture_paths:
            add(normalized)
            continue
        if normalized.startswith("README"):
            for arch_path in architecture_paths:
                if arch_path.startswith("README"):
                    add(arch_path)
            continue

        name = Path(normalized).name
        if "." in name:
            if normalized.startswith(("app/", "tests/")):
                add(normalized)
            continue

        if normalized.startswith(("app/", "tests/")):
            add(normalized.rstrip("/") + "/")

    return scope


def _snapshot_product_files() -> dict[str, bytes]:
    """Capture app/, tests/, and tracked root files so node-level generators can validate and roll back writes."""
    snapshot: dict[str, bytes] = {}
    for target_dir in ("app", "tests"):
        root = PRODUCT_ROOT / target_dir
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                snapshot[str(path.relative_to(PRODUCT_ROOT))] = path.read_bytes()
    for root_name in ROOT_WRITE_ALLOWLIST:
        path = PRODUCT_ROOT / root_name
        if path.exists() and path.is_file():
            snapshot[str(path.relative_to(PRODUCT_ROOT))] = path.read_bytes()
    return snapshot


def _changed_product_files(before: dict[str, bytes]) -> list[str]:
    current = _snapshot_product_files()
    changed: list[str] = []
    for rel_path in sorted(set(before) | set(current)):
        if before.get(rel_path) != current.get(rel_path):
            changed.append(rel_path)
    return changed


def _restore_product_files(before: dict[str, bytes], paths: list[str]) -> None:
    for rel_path in paths:
        target = PRODUCT_ROOT / rel_path
        original = before.get(rel_path)
        if original is None:
            if target.exists():
                target.unlink()
                _prune_empty_dirs(target.parent)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original)


def _path_in_write_scope(path: str, write_scope: list[str] | None) -> bool:
    if not write_scope:
        return True
    for entry in write_scope:
        if entry.endswith("/"):
            if path.startswith(entry):
                return True
        elif path == entry:
            return True
    return False


def _write_scope_violations(changed_files: list[str], write_scope: list[str] | None) -> list[str]:
    if not write_scope:
        return []
    return [path for path in changed_files if not _path_in_write_scope(path, write_scope)]


def _prune_empty_dirs(start: Path) -> None:
    stop_dirs = {PRODUCT_ROOT, PRODUCT_ROOT / "app", PRODUCT_ROOT / "tests"}
    for parent in (start, *start.parents):
        if parent in stop_dirs:
            break
        try:
            parent.rmdir()
        except OSError:
            break


def raise_quota_interrupt(
    jira: JiraClient,
    blocking_key: str | None,
    exc: LLMQuotaExceeded,
    state: dict | None = None,
    max_autonomous_retries: int | None = None,
    backoff_base_seconds: int | float | None = None,
) -> dict | None:
    """Notify Jira of the quota block; autonomously back off up to *max_autonomous_retries* times.

    On the first N quota errors, this function returns a partial state-update dict
    (with ``quota_retry_after`` set to a future Unix timestamp) so the caller can
    return it immediately to LangGraph.  ``main.py`` detects the timestamp, sleeps
    until it passes, clears the field, and re-invokes the graph — no human needed.

    Once the autonomous budget is exhausted the original ``interrupt()`` path fires,
    suspending the graph for a human to resolve the quota limit.

    The Jira notification (``_notify_quota``) fires on every quota error regardless
    of which path is taken, so the human always has visibility.

    Callers must propagate the return value::

        except LLMQuotaExceeded as e:
            patch = raise_quota_interrupt(jira, sk, e, state=state)
            if patch:
                return patch
            # only reached after human resume via interrupt()
    """
    _notify_quota(jira, blocking_key, exc)

    max_autonomous_retries = (
        LLM_QUOTA_MAX_RETRIES if max_autonomous_retries is None else max_autonomous_retries
    )
    backoff_base_seconds = (
        LLM_RETRY_BACKOFF_SECONDS if backoff_base_seconds is None else backoff_base_seconds
    )
    autonomous_retries = (state or {}).get("quota_autonomous_retries", 0)
    if autonomous_retries < max_autonomous_retries:
        backoff = backoff_base_seconds * (2 ** autonomous_retries)
        retry_after = time.time() + backoff
        log(
            f"Quota exceeded — autonomous retry "
            f"{autonomous_retries + 1}/{max_autonomous_retries} in {backoff:.0f}s"
        )
        return {
            "quota_retry_after": retry_after,
            "quota_autonomous_retries": autonomous_retries + 1,
        }
    else:
        log("Quota exceeded — autonomous retries exhausted, suspending for human intervention")
        # Write hitl_type into story state only when blocking_key is a known story key.
        story_patch: dict = {}
        if blocking_key and state and blocking_key in (state.get("stories") or {}):
            story_patch = {"stories": {blocking_key: {"hitl_type": "quota"}}}
        interrupt({
            "type": "quota",
            "provider": getattr(exc, "provider", "unknown"),
            "blocking_key": blocking_key,
        })
        return story_patch  # unreachable after interrupt(), satisfies type checker
