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
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa_criteria"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
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
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa_criteria"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
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
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa_criteria"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
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
    criteria and must NOT include unrelated story criteria."""
    jira = _make_jira()

    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    story_criteria = ["Scenario: List recipes\n  Given I am a user\n  Then I see recipes"]
    constraints_text = "**F3-418:**\nScenario: CRUD repository update/delete"

    qa_dir = tmp_path / "qa_criteria"
    qa_dir.mkdir()
    constraints_file = tmp_path / "constraints.md"
    constraints_file.write_text(constraints_text)

    with patch("agile_agent_factory.agents.reviewer_agent.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", qa_dir), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", constraints_file), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(jira, ["F3-417"], story_criteria=story_criteria)

    assert captured, "call_llm_json was never called"
    prompt = captured[0]
    assert "F3-418" not in prompt, "Prompt must not include constraints when story_criteria is provided"
    assert "List recipes" in prompt, "Prompt must include the active story's criteria"


def test_review_patch_uses_all_criteria_when_no_story_key(tmp_path):
    """When no story_key or story_criteria, reviewer assembles all qa_criteria files."""
    jira = _make_jira()

    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    qa_dir = tmp_path / "qa_criteria"
    qa_dir.mkdir()
    (qa_dir / "F3-417.md").write_text("**F3-417:**\nScenario: List recipes")
    (qa_dir / "F3-418.md").write_text("**F3-418:**\nScenario: CRUD")

    with patch("agile_agent_factory.agents.reviewer_agent.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", qa_dir), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(jira, ["F3-417"])  # no story_criteria, no story_key

    assert captured, "call_llm_json was never called"
    assert "F3-418" in captured[0], "Without story_key, all qa_criteria files must appear in prompt"


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

    with patch("agile_agent_factory.agents.reviewer_agent.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.bp_qa_criteria_path", return_value=qa_file), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(jira, ["F3-1"], story_key="F3-1")

    assert captured, "call_llm_json was never called"
    prompt = captured[0]
    assert "Check from file" in prompt, "Prompt must include QA criteria file content"


# ---------------------------------------------------------------------------
# write_scope scopes the review verdict to owned files only
# ---------------------------------------------------------------------------

def test_write_scope_injected_into_prompt(tmp_path):
    """When write_scope is provided, the prompt must name owned files and tell the
    LLM not to fail the review for issues in files outside that scope."""
    jira = _make_jira()
    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    with patch("agile_agent_factory.agents.reviewer_agent.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa_criteria"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(
            jira, ["FAKE_STORY"],
            write_scope=["tests/test_models.py", "app/models.py"],
        )

    assert captured, "call_llm_json was never called"
    prompt = captured[0]
    assert "tests/test_models.py" in prompt, "Owned test file must appear in scope instruction"
    assert "app/models.py" in prompt, "Owned app file must appear in scope instruction"
    assert "outside" in prompt.lower() or "scope" in prompt.lower(), \
        "Prompt must include a scope/outside-scope instruction"


def test_write_scope_files_always_included_despite_char_budget(tmp_path, monkeypatch):
    """Write-scope files must appear in the LLM prompt even when the total codebase
    exceeds MAX_REVIEW_TOTAL_CHARS — they must be prioritized over other files."""
    import agile_agent_factory.agents.reviewer_agent as ra
    monkeypatch.setattr(ra, "MAX_REVIEW_TOTAL_CHARS", 200)  # tiny budget

    jira = _make_jira()
    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    with patch("agile_agent_factory.agents.reviewer_agent.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa_criteria"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "tests").mkdir()
        # Filler files that easily exhaust the budget alphabetically before the scope file
        for i in range(10):
            (tmp_path / "app" / f"aaa_filler_{i:02d}.py").write_text("x = " + "1" * 50)
        # The write_scope file — comes last alphabetically, would be cut without prioritization
        scope_file = tmp_path / "tests" / "test_web_server.py"
        scope_file.write_text("def test_server(): assert True")

        reviewer_agent.review_patch(
            jira, ["F3-507"],
            write_scope=["tests/test_web_server.py"],
        )

    assert captured, "call_llm_json was never called"
    assert "tests/test_web_server.py" in captured[0], (
        "Write-scope file must appear in the prompt even when total codebase exceeds char budget"
    )


def test_write_scope_absent_when_not_provided(tmp_path):
    """When write_scope is None, the prompt must NOT contain a scope restriction section."""
    jira = _make_jira()
    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    with patch("agile_agent_factory.agents.reviewer_agent.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa_criteria"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(jira, ["FAKE_STORY"])  # no write_scope

    assert captured
    assert "Story write scope" not in captured[0], \
        "Scope section must be absent when write_scope is not provided"
