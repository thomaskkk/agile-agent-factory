"""LangGraph state definitions for the agile-agent-factory pipeline."""

from __future__ import annotations

from typing import Annotated, TypedDict


def merge_dict(current: dict, update: dict) -> dict:
    """Shallow-merge two dicts — used as a reducer for dict-valued state fields
    that can be written by multiple parallel nodes in the same superstep."""
    return {**current, **update}


def merge_stories(current: dict, update: dict) -> dict:
    """Merge per-story state updates without overwriting unmodified stories.

    Used as the LangGraph reducer for PipelineState.stories so that parallel
    agent nodes can update different stories simultaneously.
    """
    merged = {**current}
    for key, story in update.items():
        merged[key] = {**merged.get(key, {}), **story}
    return merged


class StoryState(TypedDict, total=False):
    """Per-story state — each story on the kanban board is one entry in PipelineState.stories."""

    story_key: str
    epic_key: str
    column: str  # backlog | refinement | tech_design | development | testing | code_review | done
    has_ui: bool
    gherkin_criteria: list
    test_contract: dict
    refinement_qa_done: bool   # True once QA agent has processed this story
    refinement_ux_done: bool   # True once UX agent has processed this story (or has_ui=False)
    ux_spec: dict
    ready_contract: dict
    ready_validation_errors: list[str]
    ready_validated: bool
    architecture: dict
    subtasks: dict  # title → jira_key
    dependencies: list
    blueprint: str
    retries: int
    correction_failures: int
    last_test_output: str
    review_retries: int
    review_rejection_reason: str  # populated by review_node on rejection; cleared on next dev pass
    review_status: str  # "pending_review" | "rework_needed" — only meaningful in code_review column
    hitl_type: str  # "refinement" | "intervention" | "quota" | None
    hitl_feedback: str


class PipelineState(TypedDict, total=False):
    """Top-level pipeline state managed by LangGraph's SqliteSaver checkpointer."""

    # Board
    epic_keys: list
    stories: Annotated[dict, merge_stories]
    wip_limits: dict  # column_name → max concurrent stories
    business_idea: str
    done_count: int
    total_count: int

    # Active story key — set by the dispatcher to tell a per-story node which story to process
    active_story_key: str

    # Shared pipeline-level state (Phase 1 — migrated to per-story in Phase 3+)
    has_ui: bool
    gherkin_criteria: Annotated[dict, merge_dict]  # story_key → list[str]
    ux_spec: Annotated[dict, merge_dict]
    architecture: dict
    subtasks: dict  # title → jira_key
    dependencies: list
    review_retries: int
    review_approved: bool
