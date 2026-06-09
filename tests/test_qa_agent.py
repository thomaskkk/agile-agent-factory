from unittest.mock import patch


def test_writes_per_story_criteria_file(tmp_path, monkeypatch, mock_jira):
    """inject_gherkin_criteria must write one QA criteria file per story."""
    import agile_agent_factory.agents.qa_agent as qa_agent
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    criteria_path = tmp_path / "F1-1.md"
    monkeypatch.setattr(
        "agile_agent_factory.agents.qa_agent.bp_qa_criteria_path",
        lambda sk: tmp_path / f"{sk}.md",
    )

    jira = mock_jira
    jira._request.return_value = {
        "fields": {
            "summary": "User can log in",
            "description": {"content": []},
        }
    }

    llm_response = {
        "acceptance_criteria": [
            "Scenario: Login success\n  Given valid credentials\n  When user logs in\n  Then access granted"
        ]
    }

    with patch("agile_agent_factory.tools.llm_adapters.qa.call_llm_json", return_value=llm_response):
        inject_gherkin_criteria(jira, ["F1-1"])

    assert criteria_path.exists(), "QA criteria file must be written for story F1-1"
    content = criteria_path.read_text()
    assert "F1-1" in content
    assert "Login success" in content


def test_writes_separate_files_for_multiple_stories(tmp_path, monkeypatch, mock_jira):
    """Two stories must produce two separate QA criteria files."""
    import agile_agent_factory.agents.qa_agent as qa_agent
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    monkeypatch.setattr(
        "agile_agent_factory.agents.qa_agent.bp_qa_criteria_path",
        lambda sk: tmp_path / f"{sk}.md",
    )

    jira = mock_jira
    jira._request.return_value = {
        "fields": {
            "summary": "A story",
            "description": {"content": []},
        }
    }

    llm_response = {"acceptance_criteria": ["Scenario: X\n  Given x\n  When y\n  Then z"]}

    with patch("agile_agent_factory.tools.llm_adapters.qa.call_llm_json", return_value=llm_response):
        inject_gherkin_criteria(jira, ["F1-1", "F1-2"])

    assert (tmp_path / "F1-1.md").exists()
    assert (tmp_path / "F1-2.md").exists()


def test_returns_test_contract_alongside_criteria(tmp_path, monkeypatch, mock_jira):
    """inject_gherkin_criteria must return (criteria_dict, test_contracts_dict) tuple."""
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    monkeypatch.setattr(
        "agile_agent_factory.agents.qa_agent.bp_qa_criteria_path",
        lambda sk: tmp_path / f"{sk}.md",
    )
    jira = mock_jira
    jira._request.return_value = {
        "fields": {"summary": "User can log in", "description": {"content": []}}
    }
    llm_response = {
        "acceptance_criteria": [
            "Scenario: Login success\n  Given valid credentials\n  When user logs in\n  Then access granted"
        ],
        "test_contract": {
            "test_file": "tests/test_auth.py",
            "test_functions": ["test_login_with_valid_credentials", "test_login_fails_with_wrong_password"],
            "target_imports": ["from app.auth import login_user"],
            "fixtures": [{"name": "registered_user", "description": "A user already in the system"}],
            "sample_data": [{"username": "alice", "password": "correct_horse"}],
            "edge_cases": ["empty password string", "non-existent username"],
        },
    }
    with patch("agile_agent_factory.tools.llm_adapters.qa.call_llm_json", return_value=llm_response):
        result = inject_gherkin_criteria(jira, ["F1-1"])

    criteria_dict = result.payload["gherkin_criteria"]
    test_contracts_dict = result.payload["test_contracts"]
    assert criteria_dict["F1-1"][0].startswith("Scenario:")
    assert test_contracts_dict["F1-1"]["test_file"] == "tests/test_auth.py"
    assert "test_login_with_valid_credentials" in test_contracts_dict["F1-1"]["test_functions"]
    assert test_contracts_dict["F1-1"]["target_imports"] == ["from app.auth import login_user"]


def test_test_contract_falls_back_to_empty_when_llm_omits_it(tmp_path, monkeypatch, mock_jira):
    """If LLM omits test_contract key, return empty dict for that story."""
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    monkeypatch.setattr(
        "agile_agent_factory.agents.qa_agent.bp_qa_criteria_path",
        lambda sk: tmp_path / f"{sk}.md",
    )
    jira = mock_jira
    jira._request.return_value = {
        "fields": {"summary": "A story", "description": {"content": []}}
    }
    with patch(
        "agile_agent_factory.tools.llm_adapters.qa.call_llm_json",
        return_value={"acceptance_criteria": ["Scenario: X\n  Given x\n  When y\n  Then z"]},
    ):
        result = inject_gherkin_criteria(jira, ["F1-1"])

    test_contracts_dict = result.payload["test_contracts"]
    assert test_contracts_dict["F1-1"] == {}


# ---------------------------------------------------------------------------
# Fix #9: _adf_to_text helper + description passthrough
# ---------------------------------------------------------------------------

def test_adf_to_text_extracts_paragraph_text():
    """_adf_to_text must return plain text from a nested ADF paragraph node."""
    from agile_agent_factory.agents.qa_agent import _adf_to_text

    adf = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "This is the story description."},
                ],
            }
        ],
    }
    result = _adf_to_text(adf)
    assert "This is the story description." in result


def test_adf_to_text_returns_empty_for_none():
    from agile_agent_factory.agents.qa_agent import _adf_to_text
    assert _adf_to_text(None) == ""


def test_adf_to_text_returns_empty_for_empty_dict():
    from agile_agent_factory.agents.qa_agent import _adf_to_text
    assert _adf_to_text({}) == ""


def test_description_text_passed_to_generate_criteria(tmp_path, monkeypatch, mock_jira):
    """inject_gherkin_criteria must pass the ADF description text to generate_criteria."""
    from unittest.mock import patch
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    monkeypatch.setattr(
        "agile_agent_factory.agents.qa_agent.bp_qa_criteria_path",
        lambda sk: tmp_path / f"{sk}.md",
    )
    jira = mock_jira
    jira._request.return_value = {
        "fields": {
            "summary": "User login",
            "description": {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Users can authenticate with email."}],
                    }
                ],
            },
        }
    }

    captured = {}

    def fake_generate(summary, description_text, fallback, model=None):
        captured["description_text"] = description_text
        return {"acceptance_criteria": [], "test_contract": {}}

    with patch("agile_agent_factory.agents.qa_agent.generate_criteria", side_effect=fake_generate):
        inject_gherkin_criteria(jira, ["F1-1"])

    assert "Users can authenticate with email." in captured.get("description_text", ""), (
        "Description text must be extracted from ADF and passed to generate_criteria"
    )


def test_hitl_feedback_is_passed_to_generate_criteria(tmp_path, monkeypatch, mock_jira):
    """Human clarification captured during refinement should be folded into QA regeneration context."""
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    monkeypatch.setattr(
        "agile_agent_factory.agents.qa_agent.bp_qa_criteria_path",
        lambda sk: tmp_path / f"{sk}.md",
    )
    jira = mock_jira
    jira._request.return_value = {
        "fields": {
            "summary": "User login",
            "description": {"type": "doc", "content": []},
        }
    }

    captured = {}

    def fake_generate(summary, description_text, fallback, model=None):
        captured["description_text"] = description_text
        return {"acceptance_criteria": [], "test_contract": {}}

    with patch("agile_agent_factory.agents.qa_agent.generate_criteria", side_effect=fake_generate):
        inject_gherkin_criteria(jira, ["F1-1"], story_feedback={"F1-1": "The task title is required."})

    assert "The task title is required." in captured.get("description_text", "")
