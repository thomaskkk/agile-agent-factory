import json
import re
import time
from typing import Optional

import anthropic
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from agile_agent_factory.config import (
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    LLM_TIMEOUT_SECONDS, LLM_MAX_TOKENS,
    LLM_QUOTA_MAX_RETRIES, LLM_RETRY_BACKOFF_SECONDS,
    LLM_PRIMARY_PROVIDER,
)
from agile_agent_factory.tools.logger import log


class LLMQuotaExceeded(Exception):
    def __init__(self, provider: str, message: str):
        super().__init__(message)
        self.provider = provider


class LLMTransientError(Exception):
    """Timeout or transient network error — retried with its own patient backoff, not a quota event."""
    def __init__(self, provider: str, message: str):
        super().__init__(message)
        self.provider = provider


_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")


def _detect_provider(model: str) -> str:
    """Infer provider from model name. Returns 'anthropic', 'openai', or LLM_PRIMARY_PROVIDER."""
    if model.startswith("claude-"):
        return "anthropic"
    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    return LLM_PRIMARY_PROVIDER


def _call_anthropic(prompt: str, system: str = "", model: str | None = None, prefill: str = "") -> str:
    effective_model = model or ANTHROPIC_MODEL
    log(f"Calling LLM provider: anthropic ({effective_model}).")
    llm = ChatAnthropic(
        model=effective_model,
        max_tokens=LLM_MAX_TOKENS,
        default_request_timeout=LLM_TIMEOUT_SECONDS,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_retries=0,
    )
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    if prefill:
        # Anthropic assistant-prefill: model continues from this exact text,
        # returning only the continuation. Prepend it to reconstruct the full output.
        # Not supported on all model versions — caught and retried below if rejected.
        messages.append(AIMessage(content=prefill))
    try:
        response = llm.invoke(messages)
    except anthropic.RateLimitError as e:
        raise LLMQuotaExceeded("anthropic", str(e)) from e
    except Exception as e:
        if prefill and "does not support assistant message prefill" in str(e):
            log("Model does not support prefill — retrying without it.")
            messages_no_prefill = [m for m in messages if not isinstance(m, AIMessage)]
            try:
                response = llm.invoke(messages_no_prefill)
            except anthropic.RateLimitError as e2:
                raise LLMQuotaExceeded("anthropic", str(e2)) from e2
            prefill = ""  # no continuation to prepend
        else:
            raise
    content = response.content
    if isinstance(content, list):
        # langchain_anthropic returns a list of content blocks for multi-part responses
        content = "".join(
            (b["text"] if isinstance(b, dict) else b.text)
            for b in content
            if (isinstance(b, dict) and b.get("type") == "text")
            or getattr(b, "type", None) == "text"
        )
    return (prefill + content) if prefill else content


def _call_openai(prompt: str, system: str = "", model: str | None = None) -> str:
    effective_model = model or OPENAI_MODEL
    log(f"Calling LLM provider: openai ({effective_model}).")
    llm = ChatOpenAI(
        model_name=effective_model,
        max_tokens=LLM_MAX_TOKENS,
        request_timeout=LLM_TIMEOUT_SECONDS,
        openai_api_key=OPENAI_API_KEY,
        max_retries=0,
    )
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    try:
        response = llm.invoke(messages)
    except openai.RateLimitError as e:
        raise LLMQuotaExceeded("openai", str(e)) from e
    except (openai.APITimeoutError, openai.APIConnectionError) as e:
        raise LLMTransientError("openai", f"Network error: {e}") from e
    return response.content


def _call_one_model(prompt: str, system: str, model: str, prefill: str) -> str:
    """Call a single named model, dispatching by provider."""
    if _detect_provider(model) == "openai":
        return _call_openai(prompt, system, model)
    return _call_anthropic(prompt, system, model, prefill=prefill)


def _call_with_fallback(prompt: str, system: str = "", model: str | None = None, prefill: str = "") -> str:
    # Comma-separated chain: "claude-opus-4-8,claude-sonnet-4-6,gpt-4o"
    if model and "," in model:
        chain = [m.strip() for m in model.split(",") if m.strip()]
        last_exc: Exception | None = None
        for m in chain:
            try:
                return _call_one_model(prompt, system, m, prefill)
            except LLMQuotaExceeded as e:
                log(f"{m} quota exceeded — trying next model in chain.")
                last_exc = e
            except Exception as e:
                log(f"{m} failed: {e} — trying next model in chain.")
                last_exc = e
        # All models in chain exhausted
        if isinstance(last_exc, LLMQuotaExceeded):
            raise last_exc
        raise LLMQuotaExceeded("chain", f"All models in chain failed. Last: {last_exc}")

    if model:
        detected = _detect_provider(model)
        if detected == "openai":
            try:
                return _call_openai(prompt, system, model)
            except LLMQuotaExceeded:
                log(f"openai ({model}) quota exceeded. Trying anthropic fallback.")
                return _call_anthropic(prompt, system, prefill=prefill)
            except Exception as e:
                log(f"openai ({model}) failed: {e}. Trying anthropic fallback.")
                return _call_anthropic(prompt, system, prefill=prefill)
        else:
            try:
                return _call_anthropic(prompt, system, model, prefill=prefill)
            except LLMQuotaExceeded:
                log(f"anthropic ({model}) quota exceeded. Trying openai fallback.")
                return _call_openai(prompt, system)
            except Exception as e:
                log(f"anthropic ({model}) failed: {e}. Trying openai fallback.")
                return _call_openai(prompt, system)

    if LLM_PRIMARY_PROVIDER == "openai":
        primary, primary_name, fallback, fallback_name = (
            _call_openai, "openai", _call_anthropic, "anthropic"
        )
    else:
        primary, primary_name, fallback, fallback_name = (
            _call_anthropic, "anthropic", _call_openai, "openai"
        )
    try:
        return primary(prompt, system, prefill=prefill) if primary is _call_anthropic else primary(prompt, system)
    except LLMQuotaExceeded:
        log(f"{primary_name} quota exceeded. Trying {fallback_name} fallback.")
        return fallback(prompt, system, prefill=prefill) if fallback is _call_anthropic else fallback(prompt, system)
    except Exception as e:
        log(f"Primary LLM provider ({primary_name}) failed: {e}. Trying {fallback_name}.")
        return fallback(prompt, system, prefill=prefill) if fallback is _call_anthropic else fallback(prompt, system)


@traceable(name="call_llm", run_type="llm")
def call_llm(prompt: str, system: str = "", model: str | None = None, prefill: str = "") -> str:
    for attempt in range(LLM_QUOTA_MAX_RETRIES + 1):
        try:
            return _call_with_fallback(prompt, system, model, prefill=prefill)
        except LLMTransientError as e:
            if attempt >= LLM_QUOTA_MAX_RETRIES:
                raise
            wait = LLM_RETRY_BACKOFF_SECONDS * (2 ** attempt)
            log(
                f"LLM transient error ({e.provider}): {e}. "
                f"Backing off {wait}s before retry {attempt + 1}/{LLM_QUOTA_MAX_RETRIES}."
            )
            time.sleep(wait)
        except LLMQuotaExceeded as e:
            if attempt >= LLM_QUOTA_MAX_RETRIES:
                raise
            wait = LLM_RETRY_BACKOFF_SECONDS * (2 ** attempt)
            log(
                f"Both LLM providers unavailable (quota/network: {e.provider}). "
                f"Backing off {wait}s before retry {attempt + 1}/{LLM_QUOTA_MAX_RETRIES}."
            )
            time.sleep(wait)
    raise AssertionError("unreachable")  # loop either returns or raises


def call_llm_json(
    prompt: str, system: str = "", fallback: Optional[dict] = None, model: str | None = None, prefill: str = ""
) -> dict:
    raw = call_llm(prompt, system, model, prefill=prefill)

    # Primary: strip markdown fences if the entire response is a fenced block
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw.strip())
    cleaned = fence_match.group(1).strip() if fence_match else raw.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback extraction: find outermost JSON array or object within the text
    for start_char, end_char in (("[", "]"), ("{", "}")):
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass

    # Last resort: json-repair handles unescaped quotes and other LLM escaping mistakes
    try:
        from json_repair import repair_json
        repaired = repair_json(cleaned, return_objects=True)
        if repaired not in (None, "", [], {}):
            return repaired
    except Exception:
        pass

    # Nothing parseable — log and use fallback or raise
    log(f"JSON parse failed. Raw response (first 300 chars): {repr(raw[:300])}.")
    if fallback is not None:
        log("Using deterministic fallback.")
        return fallback
    raise json.JSONDecodeError("No valid JSON found in LLM response", raw, 0)
