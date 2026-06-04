import json
import pytest


def test_call_llm_returns_text_from_anthropic(mocker):
    mock_openai = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_openai.return_value.invoke.side_effect = Exception("openai unavailable")
    mock_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    mock_chat.return_value.invoke.return_value = mocker.MagicMock(content="hello world")

    import agile_agent_factory.tools.llm_client as llm_client
    assert llm_client.call_llm("test prompt") == "hello world"


def test_call_llm_falls_back_to_openai_on_anthropic_failure(mocker):
    mock_anthropic_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    mock_anthropic_chat.return_value.invoke.side_effect = Exception("network error")
    mock_openai_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_openai_chat.return_value.invoke.return_value = mocker.MagicMock(content="openai response")

    import agile_agent_factory.tools.llm_client as llm_client
    assert llm_client.call_llm("test prompt") == "openai response"


def test_call_llm_json_parses_clean_json(mocker):
    mocker.patch("agile_agent_factory.tools.llm_client.call_llm", return_value='{"key": "value"}')
    import agile_agent_factory.tools.llm_client as llm_client
    assert llm_client.call_llm_json("p") == {"key": "value"}


def test_call_llm_json_strips_markdown_fences(mocker):
    mocker.patch("agile_agent_factory.tools.llm_client.call_llm", return_value='```json\n{"key": "value"}\n```')
    import agile_agent_factory.tools.llm_client as llm_client
    assert llm_client.call_llm_json("p") == {"key": "value"}


def test_call_llm_json_extracts_json_from_preamble_text(mocker):
    mocker.patch("agile_agent_factory.tools.llm_client.call_llm", return_value='Here are the files:\n[{"path": "app/x.py", "content": ""}]')
    import agile_agent_factory.tools.llm_client as llm_client
    result = llm_client.call_llm_json("p")
    assert result == [{"path": "app/x.py", "content": ""}]


def test_call_llm_json_extracts_object_from_preamble_text(mocker):
    mocker.patch("agile_agent_factory.tools.llm_client.call_llm", return_value='Sure! Here: {"approved": true, "reason": "ok"}')
    import agile_agent_factory.tools.llm_client as llm_client
    result = llm_client.call_llm_json("p")
    assert result == {"approved": True, "reason": "ok"}


def test_call_llm_json_returns_fallback_on_bad_json(mocker):
    mocker.patch("agile_agent_factory.tools.llm_client.call_llm", return_value="no json here whatsoever")
    import agile_agent_factory.tools.llm_client as llm_client
    result = llm_client.call_llm_json("p", fallback={"status": "fallback"})
    assert result == {"status": "fallback"}


def test_call_llm_json_raises_without_fallback(mocker):
    mocker.patch("agile_agent_factory.tools.llm_client.call_llm", return_value="definitely not json")
    import agile_agent_factory.tools.llm_client as llm_client
    with pytest.raises(json.JSONDecodeError):
        llm_client.call_llm_json("p")


def test_call_anthropic_raises_quota_exceeded_on_rate_limit(mocker):
    import anthropic as anthropic_sdk
    import agile_agent_factory.tools.llm_client as llm_client

    mock_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    mock_chat.return_value.invoke.side_effect = anthropic_sdk.RateLimitError(
        message="rate limit", response=mocker.MagicMock(status_code=429, headers={}), body={}
    )

    with pytest.raises(llm_client.LLMQuotaExceeded) as exc_info:
        llm_client._call_anthropic("prompt")
    assert exc_info.value.provider == "anthropic"


def test_call_openai_raises_quota_exceeded_on_rate_limit(mocker):
    import httpx
    import openai as openai_sdk
    import agile_agent_factory.tools.llm_client as llm_client

    mock_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_http_response = httpx.Response(429, request=mock_request)
    mock_chat.return_value.invoke.side_effect = openai_sdk.RateLimitError(
        "rate limit exceeded", response=mock_http_response, body={}
    )

    with pytest.raises(llm_client.LLMQuotaExceeded) as exc_info:
        llm_client._call_openai("prompt")
    assert exc_info.value.provider == "openai"


def test_call_llm_falls_back_to_openai_on_anthropic_quota(mocker):
    """When Anthropic hits quota, OpenAI fallback is still tried."""
    import anthropic as anthropic_sdk
    import agile_agent_factory.tools.llm_client as llm_client

    mock_anthropic_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    mock_anthropic_chat.return_value.invoke.side_effect = anthropic_sdk.RateLimitError(
        message="rate limit", response=mocker.MagicMock(status_code=429, headers={}), body={}
    )
    mock_openai_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_openai_chat.return_value.invoke.return_value = mocker.MagicMock(content="openai ok")

    result = llm_client.call_llm("prompt")
    assert result == "openai ok"


def test_call_llm_raises_quota_exceeded_when_both_providers_fail(mocker):
    """When both providers stay quota-limited across all retries, LLMQuotaExceeded propagates."""
    import anthropic as anthropic_sdk
    import httpx
    import openai as openai_sdk
    import agile_agent_factory.tools.llm_client as llm_client

    mocker.patch("agile_agent_factory.tools.llm_client.time.sleep")
    mock_anthropic_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    mock_anthropic_chat.return_value.invoke.side_effect = anthropic_sdk.RateLimitError(
        message="rate limit", response=mocker.MagicMock(status_code=429, headers={}), body={}
    )
    mock_openai_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_http_response = httpx.Response(429, request=mock_request)
    mock_openai_chat.return_value.invoke.side_effect = openai_sdk.RateLimitError(
        "openai quota", response=mock_http_response, body={}
    )

    with pytest.raises(llm_client.LLMQuotaExceeded) as exc_info:
        llm_client.call_llm("prompt")
    assert exc_info.value.provider in ("openai", "anthropic")


def test_call_llm_retries_and_recovers_after_transient_quota(mocker):
    """A transient quota error should be retried; once Anthropic recovers, call succeeds."""
    import anthropic as anthropic_sdk
    import httpx
    import openai as openai_sdk
    import agile_agent_factory.tools.llm_client as llm_client

    mock_sleep = mocker.patch("agile_agent_factory.tools.llm_client.time.sleep")
    rate_limit = anthropic_sdk.RateLimitError(
        message="rate limit", response=mocker.MagicMock(status_code=429, headers={}), body={}
    )
    ok_resp = mocker.MagicMock(content="recovered")

    mock_anthropic_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    # First whole-chain attempt: Anthropic 429. OpenAI also fails so the chain raises quota.
    # Second attempt: Anthropic succeeds.
    mock_anthropic_chat.return_value.invoke.side_effect = [rate_limit, ok_resp]

    mock_openai_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_http_response = httpx.Response(429, request=mock_request)
    mock_openai_chat.return_value.invoke.side_effect = openai_sdk.RateLimitError(
        "openai quota", response=mock_http_response, body={}
    )

    result = llm_client.call_llm("prompt")
    assert result == "recovered"
    mock_sleep.assert_called_once()  # backed off exactly once before the successful retry


def test_call_openai_raises_quota_exceeded_on_api_timeout(mocker):
    """openai.APITimeoutError from OpenAI must be caught and raised as LLMQuotaExceeded."""
    import httpx
    import openai as openai_sdk
    import agile_agent_factory.tools.llm_client as llm_client

    mock_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_chat.return_value.invoke.side_effect = openai_sdk.APITimeoutError(
        request=mock_request
    )

    with pytest.raises(llm_client.LLMQuotaExceeded) as exc_info:
        llm_client._call_openai("prompt")
    assert exc_info.value.provider == "openai"
    assert "Network error" in str(exc_info.value)


def test_call_openai_raises_quota_exceeded_on_connection_error(mocker):
    """openai.APIConnectionError from OpenAI must be caught and raised as LLMQuotaExceeded."""
    import httpx
    import openai as openai_sdk
    import agile_agent_factory.tools.llm_client as llm_client

    mock_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_chat.return_value.invoke.side_effect = openai_sdk.APIConnectionError(
        request=mock_request
    )

    with pytest.raises(llm_client.LLMQuotaExceeded) as exc_info:
        llm_client._call_openai("prompt")
    assert exc_info.value.provider == "openai"
    assert "Network error" in str(exc_info.value)


def test_detect_provider_identifies_anthropic_models():
    import agile_agent_factory.tools.llm_client as llm_client
    assert llm_client._detect_provider("claude-sonnet-4-6") == "anthropic"
    assert llm_client._detect_provider("claude-haiku-4-5-20251001") == "anthropic"
    assert llm_client._detect_provider("claude-opus-4-8") == "anthropic"


def test_detect_provider_identifies_openai_models():
    import agile_agent_factory.tools.llm_client as llm_client
    assert llm_client._detect_provider("gpt-4.1") == "openai"
    assert llm_client._detect_provider("gpt-4o") == "openai"
    assert llm_client._detect_provider("o1-mini") == "openai"


def test_call_llm_uses_specified_anthropic_model(mocker):
    mock_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    mock_chat.return_value.invoke.return_value = mocker.MagicMock(content="haiku response")

    import agile_agent_factory.tools.llm_client as llm_client
    result = llm_client.call_llm("prompt", model="claude-haiku-4-5-20251001")

    assert result == "haiku response"
    assert mock_chat.call_args[1]["model"] == "claude-haiku-4-5-20251001"


def test_call_llm_uses_specified_openai_model(mocker):
    mock_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_chat.return_value.invoke.return_value = mocker.MagicMock(content="mini response")

    import agile_agent_factory.tools.llm_client as llm_client
    result = llm_client.call_llm("prompt", model="gpt-4.1-mini")

    assert result == "mini response"
    assert mock_chat.call_args[1]["model_name"] == "gpt-4.1-mini"


def test_call_llm_falls_back_to_global_openai_when_specified_anthropic_model_fails(mocker):
    import anthropic as anthropic_sdk
    import agile_agent_factory.tools.llm_client as llm_client

    mock_anthropic = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    mock_anthropic.return_value.invoke.side_effect = anthropic_sdk.RateLimitError(
        message="rate limit", response=mocker.MagicMock(status_code=429, headers={}), body={}
    )
    mock_openai = mocker.patch("agile_agent_factory.tools.llm_client.ChatOpenAI")
    mock_openai.return_value.invoke.return_value = mocker.MagicMock(content="openai fallback")

    result = llm_client.call_llm("prompt", model="claude-haiku-4-5-20251001")
    assert result == "openai fallback"


def test_anthropic_prefill_prepended_to_response(mocker):
    """When prefill is provided, _call_anthropic must include it as AIMessage and prepend
    it to the response, so callers receive the complete text starting with the prefill."""
    from langchain_core.messages import AIMessage

    mock_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    mock_chat.return_value.invoke.return_value = mocker.MagicMock(
        content='{"path": "app/x.py", "content": "x"}]'  # continuation after "["
    )

    import agile_agent_factory.tools.llm_client as llm_client
    result = llm_client._call_anthropic("fix this", prefill="[")

    # The full response must start with the prefill
    assert result.startswith("[")
    # AIMessage with "[" was included in the messages passed to invoke
    call_args = mock_chat.return_value.invoke.call_args[0][0]
    assert any(isinstance(m, AIMessage) and m.content == "[" for m in call_args)


def test_anthropic_prefill_retries_without_prefill_on_unsupported_model(mocker):
    """When the model rejects prefill with a 400, _call_anthropic must retry without AIMessage."""
    from langchain_core.messages import AIMessage

    mock_chat = mocker.patch("agile_agent_factory.tools.llm_client.ChatAnthropic")
    ok_response = mocker.MagicMock(content="plain response")
    mock_chat.return_value.invoke.side_effect = [
        Exception("does not support assistant message prefill"),
        ok_response,
    ]

    import agile_agent_factory.tools.llm_client as llm_client
    result = llm_client._call_anthropic("fix this", prefill="[")

    assert result == "plain response"
    assert mock_chat.return_value.invoke.call_count == 2
    # Second call must have no AIMessage
    second_call_messages = mock_chat.return_value.invoke.call_args_list[1][0][0]
    assert not any(isinstance(m, AIMessage) for m in second_call_messages)


def test_call_llm_json_passes_prefill_to_call_llm(mocker):
    """call_llm_json must forward the prefill parameter to call_llm."""
    mock_call_llm = mocker.patch(
        "agile_agent_factory.tools.llm_client.call_llm",
        return_value='[{"path": "app/x.py", "content": "x"}]',
    )
    import agile_agent_factory.tools.llm_client as llm_client
    llm_client.call_llm_json("prompt", prefill="[")
    assert mock_call_llm.call_args[1]["prefill"] == "["
