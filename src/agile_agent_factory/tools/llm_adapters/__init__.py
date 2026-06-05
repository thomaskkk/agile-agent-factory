"""Per-agent LLM adapters.

Each module isolates an agent's prompt construction (system text + prompt
builder) and exposes a thin ``generate_*`` caller that wraps ``call_llm_json``
(or ``call_llm`` for readme). Agents orchestrate; adapters own the prompt.
"""
