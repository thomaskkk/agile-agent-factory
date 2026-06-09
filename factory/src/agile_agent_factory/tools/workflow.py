"""Canonical Jira workflow states — single source of truth for transition targets.

``JiraClient.transition_issue`` resolves a target by transition NAME or by the
destination STATE name (case-insensitive), so these values match the board's
column/status names.

The "To X" members and their bare "X" counterparts (e.g. ``TO_QA`` vs ``QA``,
``TO_DEVELOPMENT`` vs ``DEVELOPMENT``) intentionally coexist: the codebase uses
both forms at different call sites, and this refactor is behavior-preserving —
each member keeps the exact literal it replaced. The *one* deliberate
normalization is QA's old ``"To Tech Refinement"`` and TL's ``"Tech Refinement"``
collapsing onto a single ``TECH_REFINEMENT`` (both resolve to the same
destination state).
"""
from enum import Enum


class WorkflowState(str, Enum):
    BUSINESS_REFINEMENT = "Business Refinement"
    TECH_REFINEMENT = "Tech Refinement"
    TO_DEVELOPMENT = "To Development"
    DEVELOPMENT = "Development"
    TO_CODE_REVIEW = "To Code Review"
    IN_CODE_REVIEW = "In Code Review"
    TO_QA = "To QA"
    QA = "QA"
    DONE = "Done"
