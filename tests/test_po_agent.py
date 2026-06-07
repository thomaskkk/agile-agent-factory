from unittest.mock import patch

import pytest


def test_analyze_and_provision_skips_when_story_keys_already_exist(monkeypatch, tmp_path, mock_jira):
    """Idempotency: if story_keys are already in state, no Jira issues should be created."""
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    existing_state = {
        "status": "READY",
        "current_phase": "upstream_po_done",
        "story_keys": ["TEST-10", "TEST-11"],
        "epic_keys": ["TEST-9"],
        "blocking_issue_key": None,
        "subtasks": {},
        "review_retries": 0,
    }

    jira = mock_jira

    result = analyze_and_provision(jira, existing_state)

    jira.create_issue.assert_not_called()
    assert result.payload["story_keys"] == ["TEST-10", "TEST-11"]
    assert result.payload["epic_keys"] == ["TEST-9"]


def test_analyze_and_provision_creates_issues_when_story_keys_empty(tmp_path, monkeypatch, mock_jira):
    """When story_keys is empty, issues should be created as normal."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a todo app.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)

    llm_response = {
        "has_ambiguity": False,
        "ambiguity_description": "",
        "has_ui": True,
        "epics": [
            {
                "title": "Core App",
                "description": "The main epic",
                "stories": [
                    {
                        "title": "User can add tasks",
                        "description": "As a user, I want to add tasks.",
                        "definition_of_done": ["Task is saved"],
                    }
                ],
            }
        ],
    }

    jira = mock_jira
    jira.create_issue.side_effect = [
        {"key": "TEST-1"},  # epic
        {"key": "TEST-2"},  # story
    ]

    empty_state = {
        "status": "READY",
        "current_phase": None,
        "story_keys": [],
        "epic_keys": [],
        "blocking_issue_key": None,
        "subtasks": {},
        "review_retries": 0,
    }

    with patch("agile_agent_factory.tools.llm_adapters.po.call_llm_json", return_value=llm_response):
        result = analyze_and_provision(jira, empty_state)

    assert jira.create_issue.call_count == 2
    assert result.payload["story_keys"] == ["TEST-2"]


def test_analyze_and_provision_includes_hitl_feedback_in_prompt(tmp_path, monkeypatch, mock_jira):
    """When hitl_feedback is in state, it must appear in the LLM prompt."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a todo app.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)

    captured_prompt = {}

    def fake_llm_json(prompt, **kwargs):
        captured_prompt["value"] = prompt
        return {"has_ambiguity": False, "ambiguity_description": "", "epics": []}

    jira = mock_jira
    state_with_feedback = {
        "status": "READY",
        "current_phase": None,
        "story_keys": [],
        "epic_keys": [],
        "blocking_issue_key": None,
        "hitl_feedback": "The app only needs to run on Linux. No Windows support required.",
        "subtasks": {},
        "review_retries": 0,
    }

    with patch("agile_agent_factory.tools.llm_adapters.po.call_llm_json", side_effect=fake_llm_json):
        analyze_and_provision(jira, state_with_feedback)

    assert "only needs to run on Linux" in captured_prompt["value"]


def test_writes_business_intent(tmp_path, monkeypatch, mock_jira):
    """After analyze_and_provision, BP_BUSINESS_INTENT must exist and contain idea text."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a kanban board app.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)
    monkeypatch.setattr("agile_agent_factory.agents.po_agent.BP_BUSINESS_INTENT", tmp_path / "business_intent.md")

    llm_response = {
        "has_ambiguity": False,
        "ambiguity_description": "",
        "has_ui": True,
        "epics": [
            {
                "title": "Board",
                "description": "Main epic",
                "stories": [{"title": "Add card", "description": "As a user...", "definition_of_done": ["card saved"]}],
            }
        ],
    }

    jira = mock_jira
    jira.create_issue.side_effect = [{"key": "KB-1"}, {"key": "KB-2"}]

    with patch("agile_agent_factory.tools.llm_adapters.po.call_llm_json", return_value=llm_response):
        analyze_and_provision(jira, {"status": "READY", "current_phase": None, "story_keys": [], "epic_keys": [], "blocking_issue_key": None, "subtasks": {}, "review_retries": 0})

    intent_file = tmp_path / "business_intent.md"
    assert intent_file.exists(), "BP_BUSINESS_INTENT must be written after provisioning"
    content = intent_file.read_text()
    assert "kanban board app" in content


def test_non_critical_ambiguity_records_assumption_not_hitl(tmp_path, monkeypatch, mock_jira):
    """When PO returns hitl_required=False with assumptions, no HITL fires and ledger is recorded."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a todo app.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)
    monkeypatch.setattr("agile_agent_factory.agents.po_agent.BP_BUSINESS_INTENT", tmp_path / "business_intent.md")

    llm_response = {
        "has_ambiguity": False,
        "hitl_required": False,
        "ambiguity_description": "",
        "assumptions": [
            {"description": "App targets Linux only (unspecified in requirements)", "confidence": 0.9, "impact": 0.2}
        ],
        "has_ui": False,
        "epics": [
            {"title": "Core", "description": "Main", "stories": [
                {"title": "Add task", "description": "As a user...", "definition_of_done": ["saved"]}
            ]}
        ],
    }

    jira = mock_jira
    jira.create_issue.side_effect = [{"key": "TD-1"}, {"key": "TD-2"}]

    with patch("agile_agent_factory.tools.llm_adapters.po.call_llm_json", return_value=llm_response):
        result = analyze_and_provision(jira, {"status": "READY", "current_phase": None,
                                              "story_keys": [], "epic_keys": [],
                                              "blocking_issue_key": None, "subtasks": {}, "review_retries": 0})

    assert result.payload.get("hitl_required") is not True
    assert len(result.payload.get("assumption_ledger", [])) == 1
    assert result.payload["assumption_ledger"][0]["confidence"] == 0.9


def test_must_escalate_ambiguity_sets_hitl_required(tmp_path, monkeypatch, mock_jira):
    """When PO returns hitl_required=True, the result carries hitl_required=True."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build something.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)

    llm_response = {
        "has_ambiguity": True,
        "hitl_required": True,
        "ambiguity_description": "Scope contradicts budget",
        "assumptions": [],
        "has_ui": False,
        "epics": [],
    }

    jira = mock_jira
    jira.create_issue.return_value = {"key": "HITL-1"}

    with patch("agile_agent_factory.tools.llm_adapters.po.call_llm_json", return_value=llm_response):
        result = analyze_and_provision(jira, {"status": "READY", "current_phase": None,
                                              "story_keys": [], "epic_keys": [],
                                              "blocking_issue_key": None, "subtasks": {}, "review_retries": 0})

    assert result.payload.get("hitl_required") is True


def test_risk_score_computed_from_assumptions(tmp_path, monkeypatch, mock_jira):
    """unresolved_risk_score must be computed from assumption confidence levels."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build todo.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)
    monkeypatch.setattr("agile_agent_factory.agents.po_agent.BP_BUSINESS_INTENT", tmp_path / "bi.md")

    # low-confidence, high-impact assumption → high risk score
    llm_response = {
        "has_ambiguity": False,
        "hitl_required": False,
        "ambiguity_description": "",
        "assumptions": [
            {"description": "guessing the auth strategy", "confidence": 0.1, "impact": 0.9}
        ],
        "has_ui": False,
        "epics": [
            {"title": "Core", "description": "desc", "stories": [
                {"title": "Task", "description": "desc", "definition_of_done": ["done"]}
            ]}
        ],
    }

    jira = mock_jira
    jira.create_issue.side_effect = [{"key": "T-1"}, {"key": "T-2"}]

    with patch("agile_agent_factory.tools.llm_adapters.po.call_llm_json", return_value=llm_response):
        result = analyze_and_provision(jira, {"status": "READY", "current_phase": None,
                                              "story_keys": [], "epic_keys": [],
                                              "blocking_issue_key": None, "subtasks": {}, "review_retries": 0})

    score = result.payload.get("unresolved_risk_score", 0.0)
    # impact=0.9, confidence=0.1 → 0.9 * 0.9 = 0.81
    assert score > 0.5, f"Low-confidence high-impact assumption should produce high risk score, got {score}"


# ---------------------------------------------------------------------------
# Impact-weighted formula discriminating tests (Milestone 6)
# ---------------------------------------------------------------------------

def _base_state():
    return {
        "status": "READY",
        "current_phase": None,
        "story_keys": [],
        "epic_keys": [],
        "blocking_issue_key": None,
        "subtasks": {},
        "review_retries": 0,
    }


def _single_story_response(assumptions: list) -> dict:
    return {
        "has_ambiguity": False,
        "hitl_required": False,
        "ambiguity_description": "",
        "assumptions": assumptions,
        "has_ui": False,
        "epics": [
            {
                "title": "Core",
                "description": "Main epic",
                "stories": [
                    {"title": "Task", "description": "As a user...", "definition_of_done": ["done"]}
                ],
            }
        ],
    }


def test_high_impact_low_confidence_triggers_hitl(tmp_path, monkeypatch, mock_jira):
    """impact=0.9, confidence=0.1 → score = 0.9 * 0.9 = 0.81 > threshold (0.7) → HITL."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision
    from agile_agent_factory.config import ASSUMPTION_RISK_THRESHOLD

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a payment system.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)
    monkeypatch.setattr("agile_agent_factory.agents.po_agent.BP_BUSINESS_INTENT", tmp_path / "bi.md")

    llm_response = _single_story_response([
        {"description": "Payment provider unknown", "confidence": 0.1, "impact": 0.9}
    ])

    jira = mock_jira
    jira.create_issue.side_effect = [{"key": "P-1"}, {"key": "P-2"}]

    with patch("agile_agent_factory.tools.llm_adapters.po.call_llm_json", return_value=llm_response):
        result = analyze_and_provision(jira, _base_state())

    score = result.payload.get("unresolved_risk_score", 0.0)
    # 0.9 * (1 - 0.1) = 0.81
    assert abs(score - 0.81) < 1e-9, f"Expected 0.81, got {score}"
    assert score > ASSUMPTION_RISK_THRESHOLD, (
        f"Score {score:.3f} should exceed threshold {ASSUMPTION_RISK_THRESHOLD} to trigger HITL"
    )


def test_low_impact_low_confidence_does_not_trigger_hitl(tmp_path, monkeypatch, mock_jira):
    """impact=0.1, confidence=0.1 → score = 0.1 * 0.9 = 0.09 < threshold (0.7) → no HITL."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision
    from agile_agent_factory.config import ASSUMPTION_RISK_THRESHOLD

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a cli tool.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)
    monkeypatch.setattr("agile_agent_factory.agents.po_agent.BP_BUSINESS_INTENT", tmp_path / "bi.md")

    llm_response = _single_story_response([
        {"description": "Minor colour scheme unspecified", "confidence": 0.1, "impact": 0.1}
    ])

    jira = mock_jira
    jira.create_issue.side_effect = [{"key": "C-1"}, {"key": "C-2"}]

    with patch("agile_agent_factory.tools.llm_adapters.po.call_llm_json", return_value=llm_response):
        result = analyze_and_provision(jira, _base_state())

    score = result.payload.get("unresolved_risk_score", 0.0)
    # 0.1 * (1 - 0.1) = 0.09
    assert abs(score - 0.09) < 1e-9, f"Expected 0.09, got {score}"
    assert score < ASSUMPTION_RISK_THRESHOLD, (
        f"Score {score:.3f} should be below threshold {ASSUMPTION_RISK_THRESHOLD} — low impact should not trigger HITL"
    )


def test_mixed_assumptions_weighted_correctly(tmp_path, monkeypatch, mock_jira):
    """Two assumptions: (impact=0.9, confidence=0.1) and (impact=0.1, confidence=0.9)
    score = (0.9*0.9 + 0.1*0.1) / 2 = (0.81 + 0.01) / 2 = 0.41."""
    import agile_agent_factory.agents.po_agent as po_agent
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    business_idea_file = tmp_path / "business_idea.md"
    business_idea_file.write_text("Build a mixed app.")
    monkeypatch.setattr(po_agent, "BUSINESS_IDEA_PATH", business_idea_file)
    monkeypatch.setattr("agile_agent_factory.agents.po_agent.BP_BUSINESS_INTENT", tmp_path / "bi.md")

    llm_response = _single_story_response([
        {"description": "Critical unknown", "confidence": 0.1, "impact": 0.9},
        {"description": "Minor cosmetic detail", "confidence": 0.9, "impact": 0.1},
    ])

    jira = mock_jira
    jira.create_issue.side_effect = [{"key": "M-1"}, {"key": "M-2"}]

    with patch("agile_agent_factory.tools.llm_adapters.po.call_llm_json", return_value=llm_response):
        result = analyze_and_provision(jira, _base_state())

    score = result.payload.get("unresolved_risk_score", 0.0)
    # (0.9 * 0.9 + 0.1 * 0.1) / 2 = (0.81 + 0.01) / 2 = 0.41
    assert abs(score - 0.41) < 1e-9, f"Expected 0.41, got {score}"
