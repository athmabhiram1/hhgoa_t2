"""Guardrails are first-class pipeline stages — they KNOW WHEN NOT TO ANSWER.

Each guard returns a GuardVerdict. Refusals flow out as a normal `answer` with
mode="refusal" and a reason, never as an exception.
"""

from .grounding import GroundingGuard
from .off_topic import OffTopicGuard
from .safety import SafetyGuard

__all__ = ["GroundingGuard", "OffTopicGuard", "SafetyGuard"]