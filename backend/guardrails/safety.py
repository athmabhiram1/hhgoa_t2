"""Safety guard: abuse, hate, harassment, self-harm, illegal activity, explicit.

Heuristic keyword blocklist runs first (fast, no network). An optional LLM
judge refines uncertain cases when `GUARD_LLM_JUDGE=on`. Blocklist covers
English + common Indic scripts; anything risky without a keyword match falls to
the LLM judge or passes (we prefer precision on safety only for clear cases).
"""

from __future__ import annotations

import re

from ..config import Settings
from ..core.models import GuardVerdict
from ..core.providers import LLMClient
from .prompts import SAFETY_SCHEMA, SYSTEM_SAFETY_JUDGE

_SCHEMA_KEYS = {"unsafe", "category", "reason"}

_HARD_BLOCK = re.compile(
    r"\b(kill (him|her|yourself|me)|suicide|self[ -]?harm|paedophile|child porn|"
    r"bomb the|shoot (them|him|her|up)|rape (her|him|someone)|हत्या|आत्महत्या|को)\b",
    re.IGNORECASE,
)


class SafetyGuard:
    def __init__(self, cfg: Settings, client: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.client = client

    async def check(self, text: str) -> GuardVerdict:
        if not self.cfg.guard_safety:
            return GuardVerdict(allow=True)
        if _HARD_BLOCK.search(text):
            return GuardVerdict(allow=False, category="unsafe", reason="blocked keyword", is_refusal=True, score=1.0)
        if self.client is not None and self.cfg.guard_llm_judge:
            data = await self.client.complete_json(
                f"QUESTION: {text}", system=SYSTEM_SAFETY_JUDGE, schema_fields=_SCHEMA_KEYS
            )
            if data.get("unsafe"):
                return GuardVerdict(
                    allow=False,
                    category="unsafe",
                    reason=str(data.get("reason", "LLM safety flag")),
                    is_refusal=True,
                    score=1.0,
                )
        return GuardVerdict(allow=True)