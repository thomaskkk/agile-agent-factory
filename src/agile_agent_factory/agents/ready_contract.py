"""Definition-of-Ready contract generation and deterministic validation."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any


_PATH_UNSAFE_PARTS = {"", ".", ".."}


def build_ready_contract(
    *,
    story_key: str,
    story: dict[str, Any],
    summary: str,
    business_idea: str,
    acceptance_criteria: list[str],
    ux_spec: dict[str, Any] | None = None,
    test_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic per-story contract from existing refinement artifacts."""
    has_ui = bool(story.get("has_ui", False))
    criteria = [c for c in acceptance_criteria if isinstance(c, str) and c.strip()]
    ux_flow_reference = _ux_flow_reference(story_key, ux_spec or {}) if has_ui else None
    tc = test_contract or {}

    expected_tests = tc.get("test_functions") or _expected_tests(story_key, criteria)
    edge_cases = tc.get("edge_cases") or _edge_cases_from_criteria(criteria)

    tc_imports = [i for i in (tc.get("target_imports") or []) if isinstance(i, str) and i.strip()]
    tc_test_file = tc.get("test_file", "")
    if tc_imports or tc_test_file:
        target_ifaces: dict[str, Any] = {
            "paths": [tc_test_file] if tc_test_file else [],
            "imports": tc_imports,
        }
    else:
        target_ifaces = _target_interfaces(story)

    contract = {
        "story_key": story_key,
        "story_summary": summary.strip(),
        "full_user_intent": _join_nonempty([business_idea.strip(), summary.strip()]),
        "in_scope_behavior": _criteria_headlines(criteria),
        "out_of_scope_behavior": [
            "Behavior not described by this ready contract, its acceptance criteria, or its referenced UX flow."
        ],
        "acceptance_criteria": criteria,
        "examples": _examples_from_criteria(criteria),
        "edge_cases": edge_cases,
        "expected_tests": expected_tests,
        "target_interfaces": target_ifaces,
        "has_ui": has_ui,
        "ui_flow_reference": ux_flow_reference,
        "test_contract": tc,
        "open_questions": [],
    }

    questions: list[str] = []
    if not contract["story_summary"]:
        questions.append("Story summary is missing.")
    if not contract["full_user_intent"]:
        questions.append("Full user intent is missing.")
    if not criteria:
        questions.append("Acceptance criteria are missing.")
    if has_ui and not ux_flow_reference:
        questions.append("UI story has no mapped UX flow.")
    contract["open_questions"] = questions
    return contract


def validate_ready_contract(contract: dict[str, Any]) -> list[str]:
    """Return deterministic readiness errors; empty means the story can advance."""
    errors: list[str] = []

    if not _text(contract.get("story_summary")):
        errors.append("story_summary is required.")
    if not _text(contract.get("full_user_intent")):
        errors.append("full_user_intent is required.")
    if not _nonempty_text_list(contract.get("in_scope_behavior")):
        errors.append("in_scope_behavior must include at least one explicit behavior.")
    if not _nonempty_text_list(contract.get("out_of_scope_behavior")):
        errors.append("out_of_scope_behavior must include at least one explicit boundary.")
    if not _nonempty_text_list(contract.get("acceptance_criteria")):
        errors.append("acceptance_criteria must not be empty.")
    if contract.get("open_questions"):
        errors.append("open_questions must be empty before tech_design.")
    if contract.get("has_ui") and not _ui_flow_mapped(contract.get("ui_flow_reference")):
        errors.append("UI stories require a mapped ui_flow_reference.")
    if not _has_observable_behavior(contract):
        errors.append("Contract must include example input/output or observable expected behavior.")
    if not _has_testable_behavior(contract):
        errors.append("Contract must include testable behavior or expected tests.")

    interface_errors = _validate_target_interfaces(contract.get("target_interfaces"))
    errors.extend(interface_errors)

    return errors


def readiness_repair_update(errors: list[str]) -> dict[str, Any]:
    """Map validation failures to the upstream refinement step that can repair them."""
    update: dict[str, Any] = {
        "ready_validated": False,
        "ready_validation_errors": errors,
    }
    joined = " ".join(errors).lower()
    if "acceptance_criteria" in joined or "testable behavior" in joined or "observable expected behavior" in joined:
        update["refinement_qa_done"] = False
    if "ui" in joined and "flow" in joined:
        update["refinement_ux_done"] = False
    can_repair_upstream = "refinement_qa_done" in update or "refinement_ux_done" in update
    if not can_repair_upstream and ("open_questions" in joined or "target_interfaces" in joined):
        update["hitl_type"] = "refinement"
    return update


def _join_nonempty(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)


def _criteria_headlines(criteria: list[str]) -> list[str]:
    headlines: list[str] = []
    for item in criteria:
        first = item.strip().splitlines()[0].strip()
        if first:
            headlines.append(first)
    return headlines


def _examples_from_criteria(criteria: list[str]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for item in criteria:
        lines = [line.strip() for line in item.splitlines() if line.strip()]
        given_when = [line for line in lines if line.lower().startswith(("given ", "when ", "and "))]
        then = [line for line in lines if line.lower().startswith(("then ", "expect "))]
        if given_when or then:
            examples.append({
                "input": "; ".join(given_when) or lines[0],
                "expected_output": "; ".join(then) or lines[-1],
            })
    if examples:
        return examples
    return [{"input": "Execute the story behavior described by the acceptance criteria.", "expected_output": c} for c in criteria]


def _edge_cases_from_criteria(criteria: list[str]) -> list[str]:
    edge_cases = []
    for item in criteria:
        if re.search(r"\b(empty|invalid|missing|error|duplicate|not found|unauthorized)\b", item, re.I):
            edge_cases.append(item.strip().splitlines()[0].strip())
    return edge_cases or ["Empty input is handled without crashing.", "Invalid input returns observable feedback or a safe no-op."]


def _expected_tests(story_key: str, criteria: list[str]) -> list[str]:
    slug = re.sub(r"[^a-z0-9]+", "_", story_key.lower()).strip("_")
    tests = []
    for idx, item in enumerate(criteria, start=1):
        headline = re.sub(r"^scenario:\s*", "", item.strip().splitlines()[0], flags=re.I)
        behavior = re.sub(r"[^a-z0-9]+", "_", headline.lower()).strip("_")[:60]
        tests.append(f"test_{slug}_{behavior or idx}")
    return tests


def _target_interfaces(story: dict[str, Any]) -> dict[str, Any]:
    known = story.get("target_interfaces") or story.get("public_interfaces") or {}
    return known if isinstance(known, dict) else {}


def _ux_flow_reference(story_key: str, ux_spec: dict[str, Any]) -> dict[str, Any] | None:
    for flow in ux_spec.get("screens_or_flows", []):
        if isinstance(flow, dict) and flow.get("story_key") == story_key:
            return {
                "name": flow.get("name", ""),
                "purpose": flow.get("purpose", ""),
                "key_elements": flow.get("key_elements", []),
                "story_key": story_key,
            }
    return None


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_text_list(value: Any) -> bool:
    return isinstance(value, list) and any(_text(item) for item in value)


def _ui_flow_mapped(value: Any) -> bool:
    return isinstance(value, dict) and _text(value.get("name")) and _text(value.get("story_key"))


def _has_observable_behavior(contract: dict[str, Any]) -> bool:
    examples = contract.get("examples")
    if isinstance(examples, list):
        for example in examples:
            if isinstance(example, dict) and (_text(example.get("expected_output")) or _text(example.get("observable_behavior"))):
                return True
            if _text(example):
                return True
    return _nonempty_text_list(contract.get("acceptance_criteria"))


def _has_testable_behavior(contract: dict[str, Any]) -> bool:
    return _nonempty_text_list(contract.get("expected_tests")) or _nonempty_text_list(contract.get("acceptance_criteria"))


def _validate_target_interfaces(value: Any) -> list[str]:
    if not value:
        return []
    if not isinstance(value, dict):
        return ["target_interfaces must be an object when provided."]

    errors: list[str] = []
    for field in ("paths", "imports"):
        items = value.get(field, [])
        if not items:
            continue
        if not isinstance(items, list):
            errors.append(f"target_interfaces.{field} must be a list.")
            continue
        for item in items:
            if not isinstance(item, str) or not item.strip():
                errors.append(f"target_interfaces.{field} contains an empty value.")
                continue
            if field == "paths" and _path_is_unsafe(item):
                errors.append(f"target_interfaces.paths contains path-unsafe value: {item}")
            if field == "imports" and _import_is_unsafe(item):
                errors.append(f"target_interfaces.imports contains unsafe import: {item}")
    return errors


def _path_is_unsafe(path: str) -> bool:
    normalized = PurePosixPath(path)
    parts = normalized.parts
    if path.startswith("/") or any(part in _PATH_UNSAFE_PARTS for part in parts):
        return True
    if "app/app" in path or "tests/tests" in path:
        return True
    return not (path.startswith("app/") or path.startswith("tests/"))


def _import_is_unsafe(import_ref: str) -> bool:
    return "/" in import_ref or "\\" in import_ref or ".." in import_ref or import_ref.startswith(".")
