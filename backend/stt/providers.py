"""Speech-to-text providers.

Primary: Sarvam AI `saaras:v3`. Fallback: local faster-whisper on the RTX 5050
(free, no rate limits). The STTManager owns the failover chain.
"""

from __future__ import annotations

import base64
import io
import logging
import time

import httpx

from ..config import Settings
from ..core.models import Transcript
from ..core.retry import CircuitBreaker, call_resilient

logger = logging.getLogger(__name__)


class STTError(RuntimeError):
    pass


class SarvamSTT:
    """https://api.sarvam.ai/speech-to-text  (multipart, model=saaras:v3)"""

    ENDPOINT = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg

    async def transcribe(self, audio_bytes: bytes, *, language_code: str = "auto") -> Transcript:
        if not self.cfg.sarvam_api_key:
            raise STTError("SARVAM_API_KEY not set")
        start = time.perf_counter()
        headers = {"api-subscription-key": self.cfg.sarvam_api_key}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data: dict[str, str] = {"model": "saaras:v3", "mode": "transcribe"}
        if language_code and language_code != "auto":
            data["language_code"] = language_code

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.ENDPOINT, headers=headers, files=files, data=data)
            resp.raise_for_status()
            payload = resp.json()

        latency_ms = int((time.perf_counter() - start) * 1000)
        transcript = str(payload.get("transcript", "")).strip()
        if not transcript:
            raise STTError("Sarvam returned an empty transcript")
        return Transcript(
            text=transcript,
            language_code=str(payload.get("language_code") or "auto"),
            provider="sarvam",
            confidence=payload.get("language_probability"),
            latency_ms=latency_ms,
        )


class WhisperSTT:
    """Local faster-whisper (CTranslate2) — free fallback, runs on the RTX 5050."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.cfg.whisper_model_size,
                device=self.cfg.whisper_device,
                compute_type=self.cfg.whisper_compute_type,
            )
        return self._model

    async def transcribe(self, audio_bytes: bytes, *, language_code: str = "auto") -> Transcript:
        import asyncio
        import numpy as np

        start = time.perf_counter()
        model = self._ensure_model()

        def _run() -> tuple[str, str, float | None]:
            import soundfile as sf

            stream = io.BytesIO(audio_bytes)
            audio, sr = sf.read(stream, dtype="float32")
            if sr != 16000:
                import scipy.signal  # local import: heavy deps stay optional

                target = int(16000)
                audio = scipy.signal.resample_poly(audio, target, sr)
                sr = target
            segments, info = model.transcribe(
                audio.astype(np.float32),
                beam_size=1,
                vad_filter=True,
                language=None if language_code == "auto" else _map_bcp47(language_code),
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text, info.language, info.language_probability

        text, lang, conf = await asyncio.get_event_loop().run_in_executor(None, _run)
        latency_ms = int((time.perf_counter() - start) * 1000)
        if not text:
            raise STTError("Whisper returned an empty transcript")
        return Transcript(text=text, language_code=lang or "auto", provider="whisper", confidence=conf, latency_ms=latency_ms)


def _map_bcp47(language_code: str) -> str | None:
    """'hi-IN' -> 'hi' (faster-whisper uses ISO-639-1)."""
    return language_code.split("-")[0].lower() or None


class STTManager:
    """Owns the failover chain: Sarvam -> local Whisper. Every call is retried
    with backoff and guarded by a circuit breaker."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.sarvam = SarvamSTT(cfg)
        self.whisper = WhisperSTT(cfg)
        self._breaker = CircuitBreaker(failure_threshold=3)

    async def transcribe(self, audio_bytes: bytes, *, language_code: str = "auto") -> Transcript:
        order = ["sarvam", "whisper"]
        if self.cfg.stt_provider == "whisper":
            order = ["whisper"]
        last_exc: Exception | None = None
        for name in order:
            provider = self.sarvam if name == "sarvam" else self.whisper
            try:
                return await call_resilient(provider.transcribe, breaker=self._breaker, attempts=2, audio_bytes=audio_bytes, language_code=language_code)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("STT provider %s failed: %s", name, exc)
        raise STTError(f"all STT providers failed: {last_exc}") from last_exc