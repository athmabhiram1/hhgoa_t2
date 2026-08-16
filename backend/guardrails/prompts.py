"""Judge prompt templates re-exported for the guardrail stages.

Kept in rag.prompts to keep all prompt text in one place; this module is the
thin bridge so guardrails never import from rag directly.
"""

from ..rag.prompts import (
    FAITHFULNESS_SCHEMA,
    OFFTOPIC_SCHEMA,
    SAFETY_SCHEMA,
    SYSTEM_FAITHFULNESS,
    SYSTEM_OFFTOPIC_JUDGE,
    SYSTEM_SAFETY_JUDGE,
)

__all__ = [
    "FAITHFULNESS_SCHEMA",
    "OFFTOPIC_SCHEMA",
    "SAFETY_SCHEMA",
    "SYSTEM_FAITHFULNESS",
    "SYSTEM_OFFTOPIC_JUDGE",
    "SYSTEM_SAFETY_JUDGE",
]