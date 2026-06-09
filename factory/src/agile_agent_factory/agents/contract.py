"""Uniform agent return contract.

Every agent entrypoint returns an :class:`AgentResult` so nodes unpack results
and propagate soft failures the same way regardless of which agent ran.

Note: quota exhaustion stays an exception (``LLMQuotaExceeded``) per the
pipeline's HITL invariant — it is never folded into ``AgentResult.errors``.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """Uniform agent return. ``payload`` holds agent-specific data; ``errors``
    is non-empty on soft failure."""
    success: bool = True
    payload: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
