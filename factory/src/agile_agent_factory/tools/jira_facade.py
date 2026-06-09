"""Domain-level adapter over JiraClient.

Lets agents speak the pipeline's language (advance a story, move a batch, flag
for a human) instead of Jira's primitive verbs. Each method composes existing
``JiraClient`` / ADF-builder calls — no new HTTP logic lives here.
"""
from __future__ import annotations

from agile_agent_factory.tools.jira_client import (
    JiraClient,
    make_adf_heading,
    make_adf_bullet_list,
)
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.workflow import WorkflowState


class JiraFacade:
    def __init__(self, jira: JiraClient) -> None:
        self._jira = jira

    def advance(self, issue_key: str, state: WorkflowState) -> None:
        """Move a single issue to the given workflow state."""
        self._jira.transition_to(issue_key, state)

    def move_all(self, keys: list[str], state: WorkflowState) -> None:
        """Move every issue to the given state, skipping (and logging) any that
        have no available transition — mirrors the old _transition_all contract."""
        for key in keys:
            try:
                self._jira.transition_to(key, state)
            except ValueError as e:
                log(str(e))

    def flag_for_human(self, issue_key: str, adf_comment: dict) -> None:
        """Post a human-directed comment and raise the Jira flag for HITL."""
        self._jira.add_comment_adf(issue_key, adf_comment)
        self._jira.set_flag(issue_key)

    def post_section(self, issue_key: str, heading: str, bullets: list[str]) -> None:
        """Append a heading + bullet-list section as an ADF comment."""
        doc = {
            "version": 1,
            "type": "doc",
            "content": [make_adf_heading(heading), make_adf_bullet_list(bullets)],
        }
        self._jira.add_comment_adf(issue_key, doc)
