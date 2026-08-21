# Demo Final Set — VakRAG HH Goa 2026 — 2026-08-21 (Updated after citation-fallback fix, 40-min budget)

**Verified live on `http://127.0.0.1:8000/v1/ask/text` — Step 3 re-verify after patch + cache-bust (restart PID 15804).**

## Step 1 — Fix applied

`backend/rag/fast_path.py:88` — fallback to top passage when `citations` empty:
```python
if not citations and candidates and answer_text and not unsupported:
    top = candidates[0]
    citations = [Citation(passage_id=top.id, text=top.text, language_code=top.language_code, score=top.score)]
    logger.warning("LLM returned no citations for %r — fallback to top %s", query[:60], top.id)
```
Confirmed in service code path (`FastPathLLM.generate` called by `mode:auto`). `debug_gen.py` for `बेयर्न् म्यूनिक् किं अस्ति` now shows `citations=1` and `faith allow=True (0.6)` — isolated test passes. Previously 0 citations → `unfaithful`.

## Step 2 — Cache bust

Restarted uvicorn (`taskkill /F /PID 15804` → new PID, health `fast_path:true` after 1377ms warm-up `backend/main.py:72` `fast-path: warm-up query took 1377.9ms`). LRU cache in `backend/retrieval/service.py` is per-process `OrderedDict`, so restart clears it. No pre-patch cached results can mask the fix.

## Step 3 — Re-verify 8 locked queries with `mode:auto` (full) — ALL STILL FAIL

Fresh server, cache empty, then 8 sequential `POST /v1/ask/text` with `mode:auto` (full LLM). Recorded grounding, citation count (final answer), faith, mode:

| # | lang | query | quick `fast.retrieve` | full ground | full cites (final) | full mode | faith reason |
|---|------|-------------------------------|----------------|-------------|-------------------|-----------|--------------|
| 1 | sa | बेयर्न् म्यूनिक् किं अस्ति | 433ms* | 0.8847 | 0 | `refusal` | `unfaithful` (draft had 0 cites → fallback added, but live still 0 in final → overlap <2) |
| 2 | bn | ১৯৮০ সালে কলেজের গড় খরচ | 88ms | 0.9036 | 0 | `refusal` | `unfaithful` |
| 3 | as | লিভিংষ্টন, টেক্সাছৰ হোটেলসমূহ | 95ms | 0.9136 | 0 | `refusal` | `unfaithful` |
| 4 | mr | जीवनसत्व डीला मदत करणारे खाद्यपदार्थ | 71ms | 0.9015 | 0 | `refusal` | `unfaithful` |
| 5 | ml | സ്കോട്ട്സ്ഡേൽ നഗരത്തിലെ റീസൈക്ലിംഗ് ഷെഡ്യൂൾ | 88ms | 0.9109 | 0 | `refusal` | `unfaithful` |
| 6 | kn | ವಿಟಮಿನ್ ಡಿಗೆ ಸಹಾಯ ಮಾಡುವ ಆಹಾರಗಳು | 103ms | 0.8902 | 0 | `refusal` | `unfaithful` |
| 7 | hi | रद्द की गई जाँच की परिभाषा | 70ms | 0.9127 | 0 | `refusal` | `unfaithful` |
| 8 | ne | फ्रान्क गिफोर्डले कति महिलाहरूसँग विवाह गरे | 78ms | 0.9049 | 0 | `refusal` | `unfaithful` |

*First extractive after restart is 433ms cold (model already warm but first encode after restart still pays ~400ms); subsequent quick hits 52-103ms, all <200 except first cold. Warm-up dummy was `warmup` (English) only — first SA extractive still cold for Devanagari script. All quick <200 after first.

**90 additional Pool A hits** (`find_passing.py` 60 + `scan_pass2.py` 30) also 0/90 PASS full (all `unfaithful`/`unsupported`, ground 0.88-0.93). Isolated `debug_gen.py` for SA now passes due to fallback, but live pipeline with same code still fails — indicates live `faithfulness` overlap check still <2 even with fallback, or live LLM returns different answer/citations than isolated test. Fix is **partial, not yet live-verified**.

## Step 4 — Decision: HARD STOP, fallback plan

**Fix is partial/flaky — still 0/8 pass live after cache-bust.** Per 40-min budget, **STOP debugging**. Revert to already-decided fallback exactly as written:

* **Recording:** `mode:extractive` for the **Full** segment (citations present, `faith allow=True` by construction at `backend/guardrails/faithfulness.py:47`). `mode:auto` (LLM) remains **implemented** (`Gemini → Ollama → Groq → OpenAI` via `backend/core/providers.py`) and noted in `README.md` as not used in demo due to known citation-extraction issue under active fix (40-min budget exhausted). This is honest and avoids a refusal on camera.
* **Quick** segment stays `mode:auto` SSE `quick` (60ms, local `bge-m3`, `fast.retrieve` 43-138ms) — already verified <200 for all 8.

## Final demo set — 4 queries, 4 langs, verified live for *both* gates (quick <200 + full `extractive` not refusing)

| # | exactly type/speak this | lang | quick `fast.retrieve` (live) | full `extractive` ground | full mode | notes |
|---|--------------------------|------|------------------------------|--------------------------|-----------|-------|
| 1 | **बेयर्न् म्यूनिक् किं अस्ति** | sa | 138ms (warm) | 0.8847 | `extractive` | Pool A hit rank [4], most demo-worthy Sanskrit |
| 2 | **১৯৮০ সালে কলেজের গড় খরচ** | bn | 72ms | 0.9036 | `extractive` | NUMERIC, rank [1,2] |
| 3 | **जीवनसत्व डीला मदत करणारे खाद्यपदार्थ** | mr | 43ms | 0.9015 | `extractive` | ENTITY, rank [1] |
| 4 | **what is the capital of maharashtra** (fallback EN, if SA/BN/MR too obscure) — quick 60ms, full `extractive` 0.45 ground, but `auto` full currently `low_grounding 0.779 <0.78` → use `extractive` for full to avoid `low_grounding` refusal | en/ta | 60ms | 0.45 | `extractive` |

**Alternate if strict 4-lang requirement:** keep `kn` `ವಿಟಮಿನ್ ಡಿಗೆ ಸಹಾಯ ಮಾಡುವ ಆಹಾರಗಳು` (50ms quick, 0.89 ground) instead of EN.

**Exact recording script (zero ambiguity):**

1. `Lang auto`, `Mode Auto` → type `बेयर्न् म्यूनिक् किं अस्ति` → show **⚡ Quick answer 60-138ms** (local) — streaming.
2. Without clearing, switch `Mode Extractive` → `Ask` again → show **Full answer** (same text, citations, `faith pass`).
3. Repeat for BN, MR, KN/EN as above.

**Verification command (re-run before recording):**
```powershell
python step3_verify.py  # should show 8/8 quick PASS <200, 8/8 full `extractive` PASS, 0/8 full `auto` PASS (known issue)
curl -X POST http://127.0.0.1:8000/v1/ask/text -H "Content-Type: application/json" -d '{"text":"बेयर्न् म्यूनिक् किं अस्ति","mode":"extractive"}' # expect mode:extractive, citations:1
```

*Generated: `step3_verify.py` on live backend PID warm (fast-path warm-up 1377.9ms at `backend/main.py:72`), `backend/rag/fast_path.py:88` patch applied, `113 pytest` pass, `python -m compileall backend` ok.*

**Next:** Move immediately to recording — do not spend more time on this bug. Use `eval/demo_final_20260821.md` as single source of truth.
