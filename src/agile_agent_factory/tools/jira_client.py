from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

from agile_agent_factory.config import (
    DRY_RUN,
    JIRA_API_KEY,
    JIRA_BASE_URL,
    JIRA_FLAG_FIELD_ID,
    JIRA_FLAG_VALUE,
    JIRA_HUMAN_ACCOUNT_ID,
    JIRA_PROJECT_KEY,
    JIRA_USER_EMAIL,
)
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.workflow import WorkflowState


class JiraClient:
    def __init__(self) -> None:
        self.auth = HTTPBasicAuth(JIRA_USER_EMAIL, JIRA_API_KEY)
        self.base = JIRA_BASE_URL.rstrip("/")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base}/rest/api/3/{path}"
        response = requests.request(
            method, url, auth=self.auth, headers=self.headers, **kwargs
        )
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} {response.reason} — {response.text[:500]}",
                response=response,
            )
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def search_jql(self, jql: str, fields: Optional[list] = None) -> dict:
        payload: dict = {"jql": jql, "maxResults": 50}
        if fields:
            payload["fields"] = fields
        log(f"Jira JQL search: {jql}")
        return self._request("POST", "search/jql", json=payload)

    def create_issue(
        self,
        summary: str,
        issue_type: str,
        description_adf: Optional[dict] = None,
        parent_key: Optional[str] = None,
    ) -> dict:
        if DRY_RUN:
            log(f"[DRY_RUN] Would create {issue_type}: {summary}")
            return {"key": f"DRY-{abs(hash(summary)) % 9000 + 1000}", "id": "dry-run"}
        fields: dict = {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
        if description_adf:
            fields["description"] = description_adf
        if parent_key:
            fields["parent"] = {"key": parent_key}
        log(f"Creating Jira {issue_type}: {summary}")
        return self._request("POST", "issue", json={"fields": fields})

    def get_transitions(self, issue_key: str) -> list:
        result = self._request("GET", f"issue/{issue_key}/transitions")
        return result.get("transitions", [])

    def transition_issue(self, issue_key: str, target: str) -> None:
        if DRY_RUN:
            log(f"[DRY_RUN] Would transition {issue_key} to '{target}'")
            return
        transitions = self.get_transitions(issue_key)
        match = next(
            (
                t for t in transitions
                if t["name"].lower() == target.lower()
                or t.get("to", {}).get("name", "").lower() == target.lower()
            ),
            None,
        )
        if not match:
            names = ", ".join(
                f"{t['name']} -> {t.get('to', {}).get('name', '?')}"
                for t in transitions
            )
            raise ValueError(
                f"No Jira transition to '{target}' found for {issue_key}.\n"
                f"Available transitions: {names}"
            )
        log(f"Transitioning {issue_key} via '{match['name']}'")
        self._request(
            "POST",
            f"issue/{issue_key}/transitions",
            json={"transition": {"id": match["id"]}},
        )

    def transition_to(self, issue_key: str, state: "WorkflowState") -> None:
        """Typed convenience wrapper around transition_issue (see tools/workflow.py)."""
        self.transition_issue(issue_key, state.value)

    def add_comment_adf(self, issue_key: str, adf_body: dict) -> None:
        if DRY_RUN:
            log(f"[DRY_RUN] Would add ADF comment to {issue_key}")
            return
        log(f"Adding comment to {issue_key}")
        self._request("POST", f"issue/{issue_key}/comment", json={"body": adf_body})

    def update_issue_description(self, issue_key: str, description_adf: dict) -> None:
        if DRY_RUN:
            log(f"[DRY_RUN] Would update description on {issue_key}")
            return
        log(f"Updating description on {issue_key}")
        self._request("PUT", f"issue/{issue_key}", json={"fields": {"description": description_adf}})

    def set_flag(self, issue_key: str) -> None:
        if DRY_RUN:
            log(f"[DRY_RUN] Would set flag on {issue_key}")
            return
        log(f"Setting flag on {issue_key}")
        self._request(
            "PUT",
            f"issue/{issue_key}",
            json={"fields": {JIRA_FLAG_FIELD_ID: [{"value": JIRA_FLAG_VALUE}]}},
        )

    def clear_flag(self, issue_key: str) -> None:
        if DRY_RUN:
            log(f"[DRY_RUN] Would clear flag on {issue_key}")
            return
        self._request(
            "PUT",
            f"issue/{issue_key}",
            json={"fields": {JIRA_FLAG_FIELD_ID: None}},
        )

    def get_last_comment_text(self, issue_key: str) -> Optional[str]:
        result = self._request(
            "GET",
            f"issue/{issue_key}/comment?orderBy=-created&maxResults=1",
        )
        comments = result.get("comments", [])
        if not comments:
            return None
        return _extract_text_from_adf(comments[0].get("body", {}))

    def get_subtask_issue_type(self) -> str:
        result = self._request("GET", f"project/{JIRA_PROJECT_KEY}")
        for issue_type in result.get("issueTypes", []):
            if issue_type.get("subtask"):
                return issue_type["name"]
        raise ValueError(
            f"No subtask issue type found in project {JIRA_PROJECT_KEY}. "
            "Check the project's issue type configuration in Jira."
        )

    def is_flagged(self, issue_key: str) -> bool:
        result = self._request(
            "GET", f"issue/{issue_key}?fields={JIRA_FLAG_FIELD_ID}"
        )
        flag = result.get("fields", {}).get(JIRA_FLAG_FIELD_ID) or []
        return bool(flag)


def append_adf_doc(existing_adf: dict, extra_text: str) -> dict:
    existing_content = existing_adf.get("content", []) if existing_adf else []
    new_paragraph = {"type": "paragraph", "content": [{"type": "text", "text": extra_text}]}
    return {"version": 1, "type": "doc", "content": existing_content + [new_paragraph]}


def upsert_adf_section(existing_adf: dict | None, heading_text: str, section_nodes: list[dict], level: int = 3) -> dict:
    """Replace or append a top-level section identified by its heading text."""
    content = list((existing_adf or {}).get("content", []))
    start = next(
        (
            idx for idx, node in enumerate(content)
            if node.get("type") == "heading"
            and node.get("attrs", {}).get("level") == level
            and _extract_text_from_adf(node).strip() == heading_text
        ),
        None,
    )
    if start is None:
        return {"version": 1, "type": "doc", "content": content + section_nodes}

    end = start + 1
    while end < len(content):
        node = content[end]
        if node.get("type") == "heading" and node.get("attrs", {}).get("level", level) <= level:
            break
        end += 1

    return {"version": 1, "type": "doc", "content": content[:start] + section_nodes + content[end:]}


def make_adf_doc(text: str) -> dict:
    return {
        "version": 1,
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": text}]}
        ],
    }


def make_adf_heading(text: str, level: int = 3) -> dict:
    return {
        "type": "heading",
        "attrs": {"level": level},
        "content": [{"type": "text", "text": text}],
    }


def _text_to_inline_nodes(text: str) -> list[dict]:
    """Split text on newlines and insert hardBreak nodes between lines."""
    nodes: list[dict] = []
    for i, line in enumerate(text.split("\n")):
        if i > 0:
            nodes.append({"type": "hardBreak"})
        if line:
            nodes.append({"type": "text", "text": line})
    return nodes or [{"type": "text", "text": ""}]


def make_adf_bullet_list(items: list[str]) -> dict:
    return {
        "type": "bulletList",
        "content": [
            {
                "type": "listItem",
                "content": [
                    {"type": "paragraph", "content": _text_to_inline_nodes(item)}
                ],
            }
            for item in items
        ],
    }


def make_adf_mention_doc(account_id: str, text: str) -> dict:
    return {
        "version": 1,
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "mention",
                        "attrs": {
                            "id": account_id,
                            "text": f"@{account_id}",
                            "accessLevel": "NONE",
                        },
                    },
                    {"type": "text", "text": f" {text}"},
                ],
            }
        ],
    }


def _extract_text_from_adf(body: dict) -> str:
    def _walk(node) -> str:
        if isinstance(node, list):
            return "".join(_walk(child) for child in node)
        if not isinstance(node, dict):
            return ""

        node_type = node.get("type")
        if node_type == "text":
            return node.get("text", "")
        if node_type == "hardBreak":
            return "\n"
        if node_type == "mention":
            attrs = node.get("attrs", {})
            return attrs.get("text") or attrs.get("id", "")

        content = "".join(_walk(child) for child in node.get("content", []))
        if node_type in {"paragraph", "heading", "listItem"}:
            return content + "\n"
        if node_type in {"bulletList", "orderedList", "doc"}:
            return content
        return content

    return _walk(body).strip()
