"""Multi-provider LLM abstraction with a failover chain.

Order is locked in CONTEXT.md: Gemini -> Ollama -> Groq -> OpenAI.
Every call passes through `retry_with_backoff` + a per-provider circuit breaker.
Providers are called via plain httpx (no heavyweight SDKs) so the dependency
surface stays tiny and the fallback logic is one place.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

import httpx

from ..config import Settings
from .retry import CircuitBreaker, call_resilient

logger = logging.getLogger(__name__)

ProviderName = str  # "gemini" | "ollama" | "groq" | "openai"

DEFAULT_FALLBACK_ORDER: list[ProviderName] = ["gemini", "ollama", "groq", "openai"]


class ProviderError(RuntimeError):
    """Raised when a provider fails irrecoverably."""


# ---------------------------------------------------------------------------
# Individual provider calls (raw httpx)
# ---------------------------------------------------------------------------
async def _gemini_complete(cfg: Settings, model: str, prompt: str, *, system: str | None, temperature: float, max_tokens: int, json_mode: bool) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": cfg.gemini_api_key}
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    async with httpx.AsyncClient(timeout=cfg.llm_timeout_s) as client:
        resp = await client.post(url, params=params, json=body)
        resp.raise_for_status()
        data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"gemini malformed response: {data}") from exc


async def _ollama_complete(cfg: Settings, model: str, prompt: str, *, system: str | None, temperature: float, max_tokens: int, json_mode: bool) -> str:
    url = f"{cfg.ollama_base_url.rstrip('/')}/api/chat"
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if json_mode:
        body["format"] = "json"
    async with httpx.AsyncClient(timeout=cfg.llm_timeout_s) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
    return data.get("message", {}).get("content", "")


async def _openai_compat_complete(cfg: Settings, base: str, model: str, api_key: str, prompt: str, *, system: str | None, temperature: float, max_tokens: int, json_mode: bool) -> str:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=cfg.llm_timeout_s) as client:
        resp = await client.post(f"{base}/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]


async def _groq_complete(cfg: Settings, model: str, prompt: str, *, system: str | None, temperature: float, max_tokens: int, json_mode: bool) -> str:
    return await _openai_compat_complete(cfg, "https://api.groq.com/openai/v1", model, cfg.groq_api_key, prompt, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)


async def _openai_complete(cfg: Settings, model: str, prompt: str, *, system: str | None, temperature: float, max_tokens: int, json_mode: bool) -> str:
    return await _openai_compat_complete(cfg, "https://api.openai.com/v1", model, cfg.openai_api_key, prompt, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)


# ---------------------------------------------------------------------------
# Client with failover chain + circuit breakers
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._breakers: dict[ProviderName, CircuitBreaker] = {
            "gemini": CircuitBreaker(failure_threshold=3),
            "ollama": CircuitBreaker(failure_threshold=3),
            "groq": CircuitBreaker(failure_threshold=3),
            "openai": CircuitBreaker(failure_threshold=3),
        }

    def _call(self, provider: ProviderName, prompt: str, *, system: str | None, temperature: float, max_tokens: int, json_mode: bool) -> Callable[..., Awaitable[str]]:
        cfg = self.cfg
        model = self._model_for(provider)
        if provider == "gemini":
            return lambda: _gemini_complete(cfg, model, prompt, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
        if provider == "ollama":
            return lambda: _ollama_complete(cfg, model, prompt, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
        if provider == "groq":
            return lambda: _groq_complete(cfg, model, prompt, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
        if provider == "openai":
            return lambda: _openai_complete(cfg, model, prompt, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
        raise ProviderError(f"unknown provider {provider}")

    def _model_for(self, provider: ProviderName) -> str:
        cfg = self.cfg
        return {
            "gemini": cfg.gemini_model,
            "ollama": cfg.ollama_fallback_model,
            "groq": cfg.groq_model,
            "openai": cfg.openai_model,
        }[provider]

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        order: list[ProviderName] | None = None,
    ) -> str:
        """Try providers in order until one returns; last error propagates."""
        cfg = self.cfg
        temp = temperature if temperature is not None else cfg.generation_temperature
        tokens = max_tokens if max_tokens is not None else cfg.generation_max_tokens
        order = order or [cfg.primary_llm_provider, *[p for p in DEFAULT_FALLBACK_ORDER if p != cfg.primary_llm_provider]]

        last_exc: Exception | None = None
        for provider in order:
            if provider == "ollama" and not self._ollama_available():
                continue
            if provider in ("gemini", "groq", "openai") and not getattr(cfg, f"{provider}_api_key"):
                continue
            breaker = self._breakers[provider]
            try:
                fn = self._call(provider, prompt, system=system, temperature=temp, max_tokens=tokens, json_mode=json_mode)
                return await call_resilient(fn, breaker=breaker, attempts=cfg.llm_max_retries)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("LLM provider %s failed: %s — falling back", provider, exc)
        raise ProviderError(f"all LLM providers failed: {last_exc}") from last_exc

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema_fields: dict[str, str] | None = None,
        order: list[ProviderName] | None = None,
    ) -> dict[str, Any]:
        """Returns a parsed dict. Unparseable output triggers ONE retry on the
        same provider before moving on — never a crash."""
        instruction = ""
        if schema_fields:
            instruction = f'\nRespond ONLY with a JSON object containing exactly these keys: {", ".join(schema_fields)}.\n'
        attempt = 0
        while attempt < 2:
            raw = await self.complete(prompt + instruction, system=system, json_mode=True, order=order)
            try:
                return _parse_json_object(raw)
            except ValueError:
                attempt += 1
                if attempt >= 2:
                    raise ProviderError("LLM produced unparseable JSON after 2 attempts")
        raise ProviderError("unreachable")

    # -- LightRAG integration ---------------------------------------------
    def lightrag_llm_func(self, provider: ProviderName = "ollama") -> Callable[..., Awaitable[str]]:
        """Returns an async callable matching LightRAG's `llm_model_func`."""
        client = self

        async def llm_func(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, keyword_extraction: bool = False, **kwargs: Any) -> str:
            _ = history_messages, keyword_extraction
            return await client.complete(prompt, system=system_prompt, temperature=0.1, order=[provider])

        return llm_func

    @staticmethod
    def _ollama_available() -> bool:
        try:
            import socket

            with socket.create_connection(("localhost", 11434), timeout=0.5):
                return True
        except OSError:
            return False


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM string."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError(f"no JSON object found in: {raw[:200]!r}")