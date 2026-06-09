from agile_agent_factory.agents.contract import AgentResult


def test_agent_result_defaults_to_success():
    r = AgentResult()
    assert r.success is True and r.payload == {} and r.errors == []


def test_agent_result_carries_payload_and_errors():
    r = AgentResult(success=False, payload={"reason": "bad"}, errors=["e1"])
    assert r.success is False
    assert r.payload == {"reason": "bad"}
    assert r.errors == ["e1"]
