"""Tests for reviewer_agent.review_patch — specifically that rejection does
not trigger a backwards Jira transition to "Development"."""

from unittest.mock import MagicMock, patch

import pytest

from agile_agent_factory.agents import reviewer_agent


def _make_jira():
    jira = MagicMock()
    jira.transition_issue.return_value = None
    jira.add_comment_adf.return_value = None
    return jira


def _patch_review(approved: bool, reason: str = ""):
    return patch(
        "agile_agent_factory.agents.reviewer_agent.call_llm_json",
        return_value={"approved": approved, "rejection_reason": reason},
    )


# ---------------------------------------------------------------------------
# Rejection must NOT transition to Development
# ---------------------------------------------------------------------------

def test_rejection_never_transitions_to_development(tmp_path):
    jira = _make_jira()

    with _patch_review(approved=False, reason="Missing edge-case tests"), \
         patch("agile_agent_factory.agents.reviewer_agent.BLUEPRINT_PATH") as mock_bp, \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        mock_bp.exists.return_value = False
        # Create a dummy generated file so the agent doesn't short-circuit
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("print('hello')")

        result = reviewer_agent.review_patch(jira, ["F1-1", "F1-2"])

    assert result["approved"] is False
    for call in jira.transition_issue.call_args_list:
        target = call.args[1] if len(call.args) > 1 else call.kwargs.get("status", "")
        assert target != "Development", (
            f"review_patch must not transition to 'Development' on rejection, but got: {target}"
        )


def test_rejection_does_not_add_comment(tmp_path):
    """rejection comment is pipeline.py's responsibility, not reviewer_agent's."""
    jira = _make_jira()

    with _patch_review(approved=False, reason="Bad code"), \
         patch("agile_agent_factory.agents.reviewer_agent.BLUEPRINT_PATH") as mock_bp, \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        mock_bp.exists.return_value = False
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(jira, ["F1-1"])

    jira.add_comment_adf.assert_not_called()


# ---------------------------------------------------------------------------
# Approval path still transitions to "To QA"
# ---------------------------------------------------------------------------

def test_approval_transitions_to_qa(tmp_path):
    jira = _make_jira()

    with _patch_review(approved=True), \
         patch("agile_agent_factory.agents.reviewer_agent.BLUEPRINT_PATH") as mock_bp, \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        mock_bp.exists.return_value = False
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        result = reviewer_agent.review_patch(jira, ["F1-1"])

    assert result["approved"] is True
    targets = [
        (call.args[1] if len(call.args) > 1 else call.kwargs.get("status", ""))
        for call in jira.transition_issue.call_args_list
    ]
    assert "To QA" in targets
    assert "Development" not in targets


# ---------------------------------------------------------------------------
# story_criteria narrows the LLM prompt to the active story only
# ---------------------------------------------------------------------------

def test_story_criteria_replaces_full_blueprint_in_prompt(tmp_path):
    """When story_criteria is provided, the LLM prompt must contain only those
    criteria and must NOT include unrelated story criteria from the full blueprint."""
    jira = _make_jira()

    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    blueprint_text = (
        "**F3-417:**\nScenario: List recipes\n\n"
        "**F3-418:**\nScenario: CRUD repository update/delete"
    )
    story_criteria = ["Scenario: List recipes\n  Given I am a user\n  Then I see recipes"]

    with patch("agile_agent_factory.agents.reviewer_agent.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BLUEPRINT_PATH") as mock_bp, \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        mock_bp.exists.return_value = True
        mock_bp.read_text.return_value = blueprint_text
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(jira, ["F3-417"], story_criteria=story_criteria)

    assert captured, "call_llm_json was never called"
    prompt = captured[0]
    assert "F3-418" not in prompt, "Prompt must not include other stories' criteria"
    assert "List recipes" in prompt, "Prompt must include the active story's criteria"


def test_review_patch_falls_back_to_blueprint_when_no_story_criteria(tmp_path):
    """When story_criteria is omitted, the full blueprint is still used (backward compat)."""
    jira = _make_jira()

    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    blueprint_text = "**F3-417:**\nScenario: List recipes\n\n**F3-418:**\nScenario: CRUD"

    with patch("agile_agent_factory.agents.reviewer_agent.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BLUEPRINT_PATH") as mock_bp, \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        mock_bp.exists.return_value = True
        mock_bp.read_text.return_value = blueprint_text
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(jira, ["F3-417"])  # no story_criteria

    assert captured, "call_llm_json was never called"
    assert "F3-418" in captured[0], "Without story_criteria, full blueprint must appear in prompt"


def test_uses_qa_criteria_file_when_no_story_criteria(tmp_path):
    """When story_key is provided but no story_criteria, reviewer must use the QA criteria file."""
    jira = _make_jira()

    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    qa_content = "# QA Acceptance Criteria — F3-1\n\nScenario: Check from file\n  Given X\n  Then Y"
    qa_file = tmp_path / "F3-1.md"
    qa_file.write_text(qa_content)

    blueprint_text = "full blueprint that should NOT appear"

    with patch("agile_agent_factory.agents.reviewer_agent.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BLUEPRINT_PATH") as mock_bp, \
         patch("agile_agent_factory.agents.reviewer_agent.bp_qa_criteria_path", return_value=qa_file), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        mock_bp.exists.return_value = True
        mock_bp.read_text.return_value = blueprint_text
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(jira, ["F3-1"], story_key="F3-1")

    assert captured, "call_llm_json was never called"
    prompt = captured[0]
    assert "Check from file" in prompt, "Prompt must include QA criteria file content"
    assert blueprint_text not in prompt, "Prompt must not fall back to full blueprint when QA file exists"
