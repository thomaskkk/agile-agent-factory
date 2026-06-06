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
        "agile_agent_factory.tools.llm_adapters.reviewer.call_llm_json",
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

    assert result.success is False
    for call in jira.transition_to.call_args_list:
        target = call.args[1] if len(call.args) > 1 else call.kwargs.get("state", "")
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

    assert result.success is True
    targets = [
        (call.args[1] if len(call.args) > 1 else call.kwargs.get("state", ""))
        for call in jira.transition_to.call_args_list
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

    with patch("agile_agent_factory.tools.llm_adapters.reviewer.call_llm_json", side_effect=capture_llm), \
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

    with patch("agile_agent_factory.tools.llm_adapters.reviewer.call_llm_json", side_effect=capture_llm), \
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

    with patch("agile_agent_factory.tools.llm_adapters.reviewer.call_llm_json", side_effect=capture_llm), \
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

    with patch("agile_agent_factory.tools.llm_adapters.reviewer.call_llm_json", side_effect=capture_llm), \
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
    exceeds REVIEW_MAX_TOTAL_CHARS — they must be prioritized over other files."""
    import agile_agent_factory.agents.reviewer_agent as ra
    monkeypatch.setattr(ra, "REVIEW_MAX_TOTAL_CHARS", 200)  # tiny budget

    jira = _make_jira()
    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    with patch("agile_agent_factory.tools.llm_adapters.reviewer.call_llm_json", side_effect=capture_llm), \
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

    with patch("agile_agent_factory.tools.llm_adapters.reviewer.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa_criteria"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")

        reviewer_agent.review_patch(jira, ["FAKE_STORY"])  # no write_scope

    assert captured
    assert "Story write scope" not in captured[0], \
        "Scope section must be absent when write_scope is not provided"


def test_truncated_review_file_stays_on_line_boundary(tmp_path, monkeypatch):
    """A capped file should not be sliced mid-line into syntactically misleading text."""
    import agile_agent_factory.agents.reviewer_agent as ra

    monkeypatch.setattr(ra, "REVIEW_MAX_FILE_CHARS", 45)

    jira = _make_jira()
    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    file_content = (
        "def test_recipe_creation():\n"
        "    recipe = Recipe(name='Soup')\n"
        "    assert recipe.name == 'Soup'\n"
    )

    with patch("agile_agent_factory.tools.llm_adapters.reviewer.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa_criteria"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_recipe_repository.py").write_text(file_content)

        reviewer_agent.review_patch(jira, ["F3-999"])

    assert captured, "call_llm_json was never called"
    prompt = captured[0]
    assert "recipe = Re" not in prompt
    assert "TRUNCATED" in prompt


# ---------------------------------------------------------------------------
# Task 1: _truncate_for_review None-guard
# ---------------------------------------------------------------------------

def test_truncate_for_review_none_limit_returns_full_content():
    from agile_agent_factory.agents.reviewer_agent import _truncate_for_review
    content = "line1\n" * 2000  # well over 8000 chars
    snippet, truncated = _truncate_for_review(content, None)
    assert snippet == content
    assert truncated is False


# ---------------------------------------------------------------------------
# Task 2: write-scope full inclusion + context file truncation logging
# ---------------------------------------------------------------------------

def _fake_generate_capturing(captured: dict):
    """Returns a side_effect function that records files_block and approves."""
    def _side_effect(dod, fb, ws, fallback):
        captured["fb"] = fb
        return {"approved": True, "rejection_reason": ""}
    return _side_effect


def test_write_scope_file_never_truncated(tmp_path):
    """A write-scope file larger than REVIEW_MAX_FILE_CHARS must appear in full."""
    jira = MagicMock()
    long_content = "x = 1\n" * 2000  # ~12 000 chars, well above 8000 limit

    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_repo.py").write_text(long_content)

    captured = {}
    with patch("agile_agent_factory.agents.reviewer_agent.generate_review",
               side_effect=_fake_generate_capturing(captured)), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"):
        reviewer_agent.review_patch(jira, ["F1-1"], write_scope=["tests/test_repo.py"])

    assert "[TRUNCATED FOR REVIEW BUDGET]" not in captured["fb"], (
        "Write-scope file must not be truncated"
    )
    assert long_content.strip() in captured["fb"], (
        "Full content of write-scope file must appear in files_block"
    )


def test_context_file_is_truncated_with_marker(tmp_path):
    """A non-scope file larger than REVIEW_MAX_FILE_CHARS must be truncated with a marker."""
    jira = MagicMock()
    long_content = "x = 1\n" * 2000  # ~12 000 chars

    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "big_module.py").write_text(long_content)
    (tmp_path / "tests" / "test_small.py").write_text("def test_pass(): pass\n")

    captured = {}
    with patch("agile_agent_factory.agents.reviewer_agent.generate_review",
               side_effect=_fake_generate_capturing(captured)), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"):
        reviewer_agent.review_patch(jira, ["F1-1"], write_scope=["tests/test_small.py"])

    assert "[TRUNCATED FOR REVIEW BUDGET]" in captured["fb"], (
        "Context file exceeding REVIEW_MAX_FILE_CHARS must have truncation marker"
    )


# ---------------------------------------------------------------------------
# Task 3: scope-only retry guard
# ---------------------------------------------------------------------------

def test_truncated_scope_file_triggers_scope_only_retry(tmp_path):
    """If a write-scope file ends up truncated (edge case), a scope-only retry fires."""
    jira = MagicMock()

    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_repo.py").write_text("def test_pass(): pass\n")

    call_count = {"n": 0}
    call_blocks: list[str] = []

    def fake_generate(dod, fb, ws, fallback):
        call_count["n"] += 1
        call_blocks.append(fb)
        if call_count["n"] == 1:
            return {"approved": False, "rejection_reason": "syntax error in truncated section"}
        return {"approved": True, "rejection_reason": ""}

    original_truncate = reviewer_agent._truncate_for_review

    def fake_truncate(content, char_limit):
        if char_limit is None:
            return content[:50], True
        return original_truncate(content, char_limit)

    with patch("agile_agent_factory.agents.reviewer_agent.generate_review", side_effect=fake_generate), \
         patch("agile_agent_factory.agents.reviewer_agent._truncate_for_review", side_effect=fake_truncate), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"):

        result = reviewer_agent.review_patch(
            jira, ["F1-1"], write_scope=["tests/test_repo.py"]
        )

    assert call_count["n"] == 2, "Expected two generate_review calls (initial + scope-only retry)"
    assert result.success is True, "Scope-only retry approved — result should be success"


# ---------------------------------------------------------------------------
# Task 4: prompt honesty
# ---------------------------------------------------------------------------

def test_prompt_does_not_claim_complete_and_includes_truncation_instruction():
    """The prompt must not lie about completeness and must instruct against truncation-based rejection."""
    from agile_agent_factory.tools.llm_adapters.reviewer import build_review_prompt
    files_block = "### tests/test_repo.py\n```python\ndef test_pass(): pass\n```\n[TRUNCATED FOR REVIEW BUDGET]"
    prompt = build_review_prompt(dod_section="pass all tests", files_block=files_block)

    assert "COMPLETE contents" not in prompt, (
        "Prompt must not claim completeness — some files may be truncated"
    )
    assert "TRUNCATED" in prompt or "truncat" in prompt.lower(), (
        "Prompt must acknowledge that truncation can occur"
    )
    assert "do NOT reject" in prompt or "do not reject" in prompt.lower(), (
        "Prompt must instruct reviewer not to reject at truncation boundaries"
    )


# ---------------------------------------------------------------------------
# Task 5: line-boundary truncation regression
# ---------------------------------------------------------------------------

def test_truncate_for_review_never_ends_mid_line():
    """Truncated output must end at a line boundary, never mid-token."""
    import re
    from agile_agent_factory.agents.reviewer_agent import _truncate_for_review
    line = "recipe = Recipe(name='test', steps=['boil', 'serve'])\n"
    content = line * 200  # 200 * ~52 chars = ~10 400 chars; limit of 8000 falls mid-line
    snippet, truncated = _truncate_for_review(content, 8000)
    assert truncated is True
    assert snippet.endswith("\n") or "\n" not in snippet, (
        f"Snippet must end at a line boundary, got tail: {repr(snippet[-40:])}"
    )
    assert not re.search(r"[a-zA-Z0-9_]$", snippet), (
        f"Snippet must not end mid-identifier, got tail: {repr(snippet[-20:])}"
    )


# ---------------------------------------------------------------------------
# Original test (retained below)
# ---------------------------------------------------------------------------

def test_review_prompt_explains_truncation_behavior(tmp_path, monkeypatch):
    """The reviewer prompt must not describe truncated excerpts as complete files."""
    import agile_agent_factory.agents.reviewer_agent as ra

    monkeypatch.setattr(ra, "REVIEW_MAX_FILE_CHARS", 20)

    jira = _make_jira()
    captured: list[str] = []

    def capture_llm(prompt, **kwargs):
        captured.append(prompt)
        return {"approved": True, "rejection_reason": ""}

    with patch("agile_agent_factory.tools.llm_adapters.reviewer.call_llm_json", side_effect=capture_llm), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_QA_CRITERIA_DIR", tmp_path / "qa_criteria"), \
         patch("agile_agent_factory.agents.reviewer_agent.BP_ARCH_CONSTRAINTS", tmp_path / "constraints.md"), \
         patch("agile_agent_factory.agents.reviewer_agent.PRODUCT_ROOT", tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("def handler():\n    return 'ok'\n")

        reviewer_agent.review_patch(jira, ["F3-1000"])

    assert captured, "call_llm_json was never called"
    prompt = captured[0]
    assert "COMPLETE contents" not in prompt
    assert "do not fail review solely because visible content ends abruptly" in " ".join(prompt.split())
