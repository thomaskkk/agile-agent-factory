import json
import re
import time
from typing import Optional

import anthropic
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
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


_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")


def _detect_provider(model: str) -> str:
    """Infer provider from model name. Returns 'anthropic', 'openai', or LLM_PRIMARY_PROVIDER."""
    if model.startswith("claude-"):
        return "anthropic"
    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    return LLM_PRIMARY_PROVIDER


def _call_anthropic(prompt: str, system: str = "", model: str | None = None) -> str:
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
    try:
        response = llm.invoke(messages)
    except anthropic.RateLimitError as e:
        raise LLMQuotaExceeded("anthropic", str(e)) from e
    content = response.content
    if isinstance(content, list):
        # langchain_anthropic returns a list of content blocks for multi-part responses
        return "".join(
            (b["text"] if isinstance(b, dict) else b.text)
            for b in content
            if (isinstance(b, dict) and b.get("type") == "text")
            or getattr(b, "type", None) == "text"
        )
    return content


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
        raise LLMQuotaExceeded("openai", f"Network error: {e}") from e
    return response.content


def _call_with_fallback(prompt: str, system: str = "", model: str | None = None) -> str:
    if model:
        detected = _detect_provider(model)
        if detected == "openai":
            try:
                return _call_openai(prompt, system, model)
            except LLMQuotaExceeded:
                log(f"openai ({model}) quota exceeded. Trying anthropic fallback.")
                return _call_anthropic(prompt, system)
            except Exception as e:
                log(f"openai ({model}) failed: {e}. Trying anthropic fallback.")
                return _call_anthropic(prompt, system)
        else:
            try:
                return _call_anthropic(prompt, system, model)
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
        return primary(prompt, system)
    except LLMQuotaExceeded:
        log(f"{primary_name} quota exceeded. Trying {fallback_name} fallback.")
        return fallback(prompt, system)
    except Exception as e:
        log(f"Primary LLM provider ({primary_name}) failed: {e}. Trying {fallback_name}.")
        return fallback(prompt, system)


@traceable(name="call_llm", run_type="llm")
def call_llm(prompt: str, system: str = "", model: str | None = None) -> str:
    for attempt in range(LLM_QUOTA_MAX_RETRIES + 1):
        try:
            return _call_with_fallback(prompt, system, model)
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
    prompt: str, system: str = "", fallback: Optional[dict] = None, model: str | None = None
) -> dict:
    raw = call_llm(prompt, system, model)

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

    # Nothing parseable — log and use fallback or raise
    log(f"JSON parse failed. Raw response (first 300 chars): {repr(raw[:300])}.")
    if fallback is not None:
        log("Using deterministic fallback.")
        return fallback
    raise json.JSONDecodeError("No valid JSON found in LLM response", raw, 0)
