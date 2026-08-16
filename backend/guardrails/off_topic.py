"""Off-topic guard: does the question look answerable from an Indic encyclopedic
corpus? Pure heuristics by default (no network): very short or non-interrogative
utterances, chit-chat, and questions clearly outside general knowledge (opinions
on persons, private/current state of the world) get refused.

The MSMARCO corpus is a huge general-knowledge QA set, so most questions pass.
We only refuse when the query is clearly not a knowledge question.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..core.models import GuardVerdict
from ..core.providers import LLMClient
from .prompts import OFFTOPIC_SCHEMA, SYSTEM_OFFTOPIC_JUDGE

_CHITCHAT = re.compile(
    r"^(hi|hello|hey|namaste|thanks|thank you|ok|okay|bye|good ?(morning|evening|night)|"
    r"how are you|kaise ho|kya haal|who are you|what can you do|help|tell me a joke)$",
    re.IGNORECASE,
)

_OPINION = re.compile(
    r"\b(opinion|feel about|think about|like about|is it good|do you (think|like|enjoy))\b",
    re.IGNORECASE,
)

# A request is off-topic chit-chat when it is ONLY a greeting/small-talk or an
# opinion-seeking phrase. Opinion questions phrased WITH a question word
# ("what is your opinion on X?") are still opinion questions, so the old
# `and not _QUESTION_WORD` guard was wrong — it let them through.
_OPINION_ONLY = re.compile(
    r"^(what('s| is)? (your|do you) (opinion|view)|do you (think|like)|how do you feel|"
    r"what do you think|tell me a joke|who are you)\b",
    re.IGNORECASE,
)

_QUESTION_WORD = re.compile(r"(what|who|when|where|why|how|which|का|क्या|कौन|कब|कहाँ|क्यों|कैसे|యెవరు|ఎవరు|என்ன|யார்|کیا|کیسے|কি|কী)", re.IGNORECASE)


class OffTopicGuard:
    def __init__(self, cfg: Settings, client: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.client = client

    async def check(self, text: str) -> GuardVerdict:
        if not self.cfg.guard_offtopic:
            return GuardVerdict(allow=True)
        stripped = text.strip().rstrip("?.!।॥")
        if not stripped or len(stripped) < 4:
            return GuardVerdict(allow=False, category="off_topic", reason="too short to be a question", is_refusal=True, score=1.0)
        if _CHITCHAT.match(stripped.lower()):
            return GuardVerdict(allow=False, category="off_topic", reason="chit-chat, not a knowledge question", is_refusal=True, score=1.0)
        if _OPINION_ONLY.match(stripped) or (_OPINION.search(stripped) and not _QUESTION_WORD.search(stripped)):
            return GuardVerdict(allow=False, category="off_topic", reason="subjective/opinion question", is_refusal=True, score=0.8)
        if self.client is not None and self.cfg.guard_llm_judge:
            data = await self.client.complete_json(
                f"QUESTION: {text}",
                system=SYSTEM_OFFTOPIC_JUDGE,
                schema_fields={"off_topic", "reason"},
            )
            if data.get("off_topic"):
                return GuardVerdict(allow=False, category="off_topic", reason=str(data.get("reason", "off-topic")), is_refusal=True, score=0.9)
        return GuardVerdict(allow=True)