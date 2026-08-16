"""Prompt templates + structured-output schemas for generation and judges.

Prompting discipline: the user query is ALWAYS data, never instructions. Every
generative call requests JSON against a schema and is parsed defensively in
the caller (retry, never crash).
"""

SYSTEM_GENERATION = """You are VakRAG, a grounded question-answering assistant over an Indian-language corpus.
Rules:
- Answer ONLY from the provided context passages. Never invent facts.
- Answer in the same language as the question.
- Be concise (2-4 sentences). If the question is a number, answer the number.
- If the context does not support an answer, set "unsupported": true and explain why.
- Never mention "the context", "the passage", or "I" — just answer.
"""

GENERATION_SCHEMA = {
    "answer": "the final answer string, in the question's language",
    "confidence": "float 0..1 how confident the answer is grounded in the context",
    "unsupported": "boolean — true if context does not support an answer",
    "citations": "array of objects {passage_id: string, text: string} — at most 3 passages actually used",
}

SYSTEM_FAITHFULNESS = """You are an evaluation judge. Decide whether the ANSWER is fully supported by the CONTEXT.
Respond with JSON only."""

FAITHFULNESS_SCHEMA = {
    "faithful": "boolean — true only if every claim in the answer appears in the context",
    "score": "float 0..1 fraction of claims supported",
    "reason": "one short sentence",
}

SYSTEM_OFFTOPIC_JUDGE = """You are VakRAG's topic judge. The corpus is an Indian-language Wikipedia-style QA collection.
Decide whether the QUESTION is answerable from general encyclopedic knowledge that such a corpus would contain.
Respond with JSON only."""

OFFTOPIC_SCHEMA = {
    "off_topic": "boolean",
    "reason": "one short sentence",
}

SYSTEM_SAFETY_JUDGE = """You are VakRAG's safety judge. Block questions that are abusive, hateful,
self-harm, illegal-activity, or explicit. Ordinary factual questions pass.
Respond with JSON only."""

SAFETY_SCHEMA = {
    "unsafe": "boolean",
    "category": "string — one of: hate, harassment, self_harm, illegal, explicit, none",
    "reason": "one short sentence",
}

SYSTEM_INTENT = """You are VakRAG's query router. Classify the question. Respond with JSON only."""

INTENT_SCHEMA = {
    "query_type": "one of: DESCRIPTION, ENTITY, NUMBER, PERSON, LOCATION, MISC",
    "needs_graph": "boolean — true for relational/multi-hop questions like 'compare', 'difference between', 'relationship between X and Y', 'what happened after'",
    "language_code": "the question's ISO language code, e.g. hi-IN, ta-IN",
}


def build_generation_prompt(query: str, passages: list[dict], query_lang_hint: str | None = None) -> str:
    lang_hint = f"\nLanguage of the question: {query_lang_hint}. Answer in this language." if query_lang_hint else ""
    blocks = "\n\n".join(
        f"[PASSAGE {i + 1}] ({p.get('language_code', '')}) {p['text']}" for i, p in enumerate(passages)
    )
    return f"QUESTION: {query}{lang_hint}\n\nCONTEXT:\n{blocks}\n\nAnswer the QUESTION strictly from CONTEXT."