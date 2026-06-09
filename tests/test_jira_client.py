import pytest
import requests
import responses as resp_lib


@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    import agile_agent_factory.tools.jira_client as jira_client
    monkeypatch.setattr(jira_client, "DRY_RUN", False)
    monkeypatch.setattr(jira_client, "JIRA_BASE_URL", "https://test.atlassian.net")
    monkeypatch.setattr(jira_client, "JIRA_PROJECT_KEY", "TEST")
    monkeypatch.setattr(jira_client, "JIRA_FLAG_FIELD_ID", "customfield_10021")
    monkeypatch.setattr(jira_client, "JIRA_FLAG_VALUE", "Impediment")


@resp_lib.activate
def test_search_jql_uses_post():
    import agile_agent_factory.tools.jira_client as jira_client
    resp_lib.add(
        resp_lib.POST,
        "https://test.atlassian.net/rest/api/3/search/jql",
        json={"issues": [{"key": "TEST-1"}]},
        status=200,
    )
    client = jira_client.JiraClient()
    result = client.search_jql("project = TEST")
    assert result["issues"][0]["key"] == "TEST-1"


@resp_lib.activate
def test_create_issue():
    import agile_agent_factory.tools.jira_client as jira_client
    resp_lib.add(
        resp_lib.POST,
        "https://test.atlassian.net/rest/api/3/issue",
        json={"key": "TEST-2", "id": "100"},
        status=201,
    )
    client = jira_client.JiraClient()
    result = client.create_issue("My Story", "Story")
    assert result["key"] == "TEST-2"


@resp_lib.activate
def test_transition_matches_destination_status():
    import agile_agent_factory.tools.jira_client as jira_client
    resp_lib.add(
        resp_lib.GET,
        "https://test.atlassian.net/rest/api/3/issue/TEST-1/transitions",
        json={"transitions": [
            {"id": "21", "name": "Start Progress", "to": {"name": "In Progress"}}
        ]},
        status=200,
    )
    resp_lib.add(
        resp_lib.POST,
        "https://test.atlassian.net/rest/api/3/issue/TEST-1/transitions",
        body=b"",
        status=204,
    )
    client = jira_client.JiraClient()
    client.transition_issue("TEST-1", "In Progress")  # match by destination status name


@resp_lib.activate
def test_transition_raises_with_available_list_when_not_found():
    import agile_agent_factory.tools.jira_client as jira_client
    resp_lib.add(
        resp_lib.GET,
        "https://test.atlassian.net/rest/api/3/issue/TEST-1/transitions",
        json={"transitions": [
            {"id": "21", "name": "Start Progress", "to": {"name": "In Progress"}}
        ]},
        status=200,
    )
    client = jira_client.JiraClient()
    with pytest.raises(ValueError, match="No Jira transition to 'Done' found"):
        client.transition_issue("TEST-1", "Done")


@resp_lib.activate
def test_get_subtask_issue_type_returns_name_with_subtask_flag():
    import agile_agent_factory.tools.jira_client as jira_client
    resp_lib.add(
        resp_lib.GET,
        "https://test.atlassian.net/rest/api/3/project/TEST",
        json={
            "issueTypes": [
                {"name": "Story", "subtask": False},
                {"name": "Bug", "subtask": False},
                {"name": "Sub-task", "subtask": True},
            ]
        },
        status=200,
    )
    client = jira_client.JiraClient()
    assert client.get_subtask_issue_type() == "Sub-task"


@resp_lib.activate
def test_get_subtask_issue_type_raises_when_none_found():
    import agile_agent_factory.tools.jira_client as jira_client
    resp_lib.add(
        resp_lib.GET,
        "https://test.atlassian.net/rest/api/3/project/TEST",
        json={"issueTypes": [{"name": "Story", "subtask": False}]},
        status=200,
    )
    client = jira_client.JiraClient()
    with pytest.raises(ValueError, match="No subtask issue type"):
        client.get_subtask_issue_type()


def test_append_adf_doc_merges_content():
    import agile_agent_factory.tools.jira_client as jira_client
    existing = jira_client.make_adf_doc("Original description.")
    result = jira_client.append_adf_doc(existing, "Acceptance Criteria:\n\nScenario: ...")
    assert result["version"] == 1
    assert len(result["content"]) == 2
    assert result["content"][0]["content"][0]["text"] == "Original description."
    assert "Acceptance Criteria" in result["content"][1]["content"][0]["text"]


def test_append_adf_doc_empty_existing():
    import agile_agent_factory.tools.jira_client as jira_client
    result = jira_client.append_adf_doc({}, "Acceptance Criteria:\n\nScenario: ...")
    assert len(result["content"]) == 1
    assert "Acceptance Criteria" in result["content"][0]["content"][0]["text"]


def test_upsert_adf_section_replaces_existing_section():
    import agile_agent_factory.tools.jira_client as jira_client

    existing = {
        "version": 1,
        "type": "doc",
        "content": [
            jira_client.make_adf_heading("Acceptance Criteria"),
            jira_client.make_adf_bullet_list(["Old item"]),
            jira_client.make_adf_heading("Definition of Done"),
            jira_client.make_adf_bullet_list(["Keep me"]),
        ],
    }
    replacement = [
        jira_client.make_adf_heading("Acceptance Criteria"),
        jira_client.make_adf_bullet_list(["New item"]),
    ]

    result = jira_client.upsert_adf_section(existing, "Acceptance Criteria", replacement)
    text = jira_client._extract_text_from_adf(result)
    assert "New item" in text
    assert "Old item" not in text
    assert "Keep me" in text


def test_extract_text_from_adf_reads_nested_lists_and_mentions():
    import agile_agent_factory.tools.jira_client as jira_client

    adf = {
        "version": 1,
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "mention", "attrs": {"id": "uid123", "text": "@uid123"}},
                    {"type": "text", "text": " please review"},
                ],
            },
            jira_client.make_adf_bullet_list(["Line one\nLine two"]),
        ],
    }

    text = jira_client._extract_text_from_adf(adf)
    assert "@uid123" in text
    assert "please review" in text
    assert "Line one" in text
    assert "Line two" in text


@resp_lib.activate
def test_update_issue_description_calls_put():
    import agile_agent_factory.tools.jira_client as jira_client
    resp_lib.add(
        resp_lib.PUT,
        "https://test.atlassian.net/rest/api/3/issue/TEST-5",
        body=b"",
        status=204,
    )
    client = jira_client.JiraClient()
    adf = jira_client.make_adf_doc("Full description with Gherkin appended.")
    client.update_issue_description("TEST-5", adf)
    assert len(resp_lib.calls) == 1
    assert resp_lib.calls[0].request.method == "PUT"


def test_update_issue_description_dry_run_skips(monkeypatch):
    import agile_agent_factory.tools.jira_client as jira_client
    monkeypatch.setattr(jira_client, "DRY_RUN", True)
    client = jira_client.JiraClient()
    adf = jira_client.make_adf_doc("Should not be sent.")
    client.update_issue_description("TEST-5", adf)  # must not raise (no HTTP call)


@resp_lib.activate
def test_transition_issue_dry_run_skips_get_transitions(monkeypatch):
    import agile_agent_factory.tools.jira_client as jira_client
    monkeypatch.setattr(jira_client, "DRY_RUN", True)
    client = jira_client.JiraClient()
    client.transition_issue("TEST-9", "Development")
    assert len(resp_lib.calls) == 0


def test_make_adf_doc_structure():
    import agile_agent_factory.tools.jira_client as jira_client
    doc = jira_client.make_adf_doc("Hello world")
    assert doc["version"] == 1
    assert doc["type"] == "doc"
    assert doc["content"][0]["content"][0]["text"] == "Hello world"


def test_make_adf_mention_doc():
    import agile_agent_factory.tools.jira_client as jira_client
    doc = jira_client.make_adf_mention_doc("uid123", "Please review.")
    para = doc["content"][0]["content"]
    assert para[0]["type"] == "mention"
    assert para[0]["attrs"]["id"] == "uid123"
    assert "Please review" in para[1]["text"]


def test_make_adf_heading_default_level():
    import agile_agent_factory.tools.jira_client as jira_client
    node = jira_client.make_adf_heading("Definition of Done")
    assert node["type"] == "heading"
    assert node["attrs"]["level"] == 3
    assert node["content"][0]["text"] == "Definition of Done"


def test_make_adf_heading_custom_level():
    import agile_agent_factory.tools.jira_client as jira_client
    node = jira_client.make_adf_heading("Summary", level=2)
    assert node["attrs"]["level"] == 2


def test_make_adf_bullet_list_simple_items():
    import agile_agent_factory.tools.jira_client as jira_client
    node = jira_client.make_adf_bullet_list(["Item A", "Item B"])
    assert node["type"] == "bulletList"
    assert len(node["content"]) == 2
    first_item = node["content"][0]
    assert first_item["type"] == "listItem"
    assert first_item["content"][0]["content"][0]["text"] == "Item A"


def test_make_adf_bullet_list_multiline_item_uses_hardbreak():
    import agile_agent_factory.tools.jira_client as jira_client
    node = jira_client.make_adf_bullet_list(["Scenario: Title\n  Given context\n  When action"])
    inline = node["content"][0]["content"][0]["content"]
    types = [n["type"] for n in inline]
    assert "hardBreak" in types
    texts = [n.get("text", "") for n in inline if n["type"] == "text"]
    assert "Scenario: Title" in texts
    assert "  Given context" in texts


def test_request_passes_timeout(monkeypatch):
    import agile_agent_factory.tools.jira_client as jira_client

    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200
        content = b'{"ok": true}'

        def json(self):
            return {"ok": True}

    def fake_request(method, url, auth=None, headers=None, timeout=None, **kwargs):
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(jira_client.requests, "request", fake_request)

    client = jira_client.JiraClient()
    result = client._request("GET", "issue/TEST-1")

    assert result == {"ok": True}
    assert captured["timeout"] == jira_client.JIRA_TIMEOUT_SECONDS


def test_request_retries_on_timeout(monkeypatch):
    import agile_agent_factory.tools.jira_client as jira_client

    attempts = {"count": 0}

    class FakeResponse:
        ok = True
        status_code = 200
        content = b'{"ok": true}'

        def json(self):
            return {"ok": True}

    def fake_request(method, url, auth=None, headers=None, timeout=None, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise requests.Timeout("boom")
        return FakeResponse()

    monkeypatch.setattr(jira_client.requests, "request", fake_request)
    monkeypatch.setattr(jira_client.time, "sleep", lambda *_: None)

    client = jira_client.JiraClient()
    result = client._request("GET", "issue/TEST-1")

    assert result == {"ok": True}
    assert attempts["count"] == 2
