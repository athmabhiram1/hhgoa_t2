import asyncio
import inspect
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.core.models import Transcript
from backend.harness.pipeline import VakRagPipeline
from backend.stt.providers import STTManager, SarvamSTT, WhisperSTT


def _run(coro):
    return asyncio.run(coro)


def test_stt_transcribe_contract_has_no_filename_param():
    """Lock the real STT contract: (audio_bytes, *, language_code='auto').

    Regression guard for the production crash
    "STTManager.transcribe() got an unexpected keyword argument 'filename'":
    the manager and every provider must NOT grow a filename param without
    updating every call site.
    """
    for cls in (STTManager, SarvamSTT, WhisperSTT):
        params = inspect.signature(cls.transcribe).parameters
        assert "audio_bytes" in params, f"{cls.__name__}.transcribe lost audio_bytes"
        assert "filename" not in params, f"{cls.__name__}.transcribe must not accept filename"
        assert params["language_code"].kind is inspect.Parameter.KEYWORD_ONLY


def test_run_audio_forwards_only_audio_bytes_to_stt():
    """Audio-upload path (run_audio -> STTManager.transcribe) kwarg contract.

    run_audio must call stt.transcribe(audio_bytes) — the historical bug passed
    filename=, which the real method rejects with TypeError. The stub below
    mirrors the real signature exactly, so it reproduces the production crash
    if the bug returns.
    """
    pipe = VakRagPipeline(get_settings(), client=None)
    captured = {}

    async def fake_transcribe(audio_bytes, *, language_code="auto"):
        captured["audio_bytes"] = audio_bytes
        captured["language_code"] = language_code
        return Transcript(text="नमस्ते", language_code="hi", provider="sarvam", latency_ms=12)

    pipe.stt.transcribe = fake_transcribe
    pipe.run_transcript = AsyncMock(return_value="pipeline-result")

    audio = b"RIFF____WAVEfmt\x00 not-real-audio"
    result = _run(pipe.run_audio(audio))

    assert captured["audio_bytes"] == audio
    assert captured["language_code"] == "auto"
    assert result == "pipeline-result"
    pipe.run_transcript.assert_awaited_once_with("नमस्ते", lang="hi")


def test_run_audio_uses_stt_provider_language_hint():
    """STT-detected language flows into run_transcript as the lang hint."""
    pipe = VakRagPipeline(get_settings(), client=None)
    captured = {}

    async def fake_transcribe(audio_bytes, *, language_code="auto"):
        return Transcript(text="வணக்கம்", language_code="ta", provider="sarvam", latency_ms=9)

    pipe.stt.transcribe = fake_transcribe
    pipe.run_transcript = AsyncMock(return_value="ok")

    _run(pipe.run_audio(b"RIFF____WAVEfmt\x00x"))
    pipe.run_transcript.assert_awaited_once_with("வணக்கம்", lang="ta")