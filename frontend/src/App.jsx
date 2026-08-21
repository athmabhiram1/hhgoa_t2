import React, { useCallback, useEffect, useRef, useState } from "react";
import { askSSE, getGraph, getHealth, getTelemetry } from "./api.js";

const MODE_LABEL = { auto: "Auto", extractive: "Extractive" };

export default function App() {
  const [text, setText] = useState("");
  const [lang, setLang] = useState("");
  const [mode, setMode] = useState("auto");
  const [stages, setStages] = useState([]);
  const [answer, setAnswer] = useState(null);
  const [quickAnswer, setQuickAnswer] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [requestId, setRequestId] = useState(null);
  const [totalMs, setTotalMs] = useState(null);
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [micError, setMicError] = useState("");
  const [telemetry, setTelemetry] = useState(null);
  const [graph, setGraph] = useState(null);
  const [health, setHealth] = useState({ status: "checking" });
  const [tab, setTab] = useState("ask");

  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const abortRef = useRef(null);

  // ---- telemetry + health poll -------------------------------------------
  const refresh = useCallback(() => {
    getTelemetry().then(setTelemetry).catch(() => {});
    getHealth().then(setHealth).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 2500);
    return () => clearInterval(id);
  }, [refresh]);

  // ---- request lifecycle --------------------------------------------------
  const handleEvent = useCallback((event, data) => {
    setStages((prev) => [...prev, { event, data, t: Date.now() }]);
    if (event === "quick") setQuickAnswer(data);
    if (event === "retrieval") setCandidates(data.candidates || []);
    if (event === "answer") setAnswer(data);
    if (event === "done") {
      setRequestId(data.request_id);
      setTotalMs(data.total_ms);
      setStages((prev) => [...prev, { event: "done", data: data.spans, t: Date.now() }]);
    }
    if (event === "error") setMicError(data.message || "server error");
  }, []);

  const submit = useCallback(
    async ({ textValue, audioB64 }) => {
      setBusy(true);
      setStages([]);
      setAnswer(null);
      setQuickAnswer(null);
      setCandidates([]);
      setRequestId(null);
      setTotalMs(null);
      setMicError("");
      abortRef.current = new AbortController();
      try {
        await askSSE({
          text: textValue,
          audioB64,
          lang: lang || undefined,
          mode,
          onEvent: handleEvent,
          signal: abortRef.current.signal,
        });
      } catch (err) {
        if (err.name !== "AbortError") {
          setMicError(String(err.message || err));
          setStages((prev) => [...prev, { event: "error", data: { message: String(err.message || err) }, t: Date.now() }]);
        }
      } finally {
        setBusy(false);
      }
    },
    [lang, mode, handleEvent]
  );

  // ---- mic capture --------------------------------------------------------
  const startRecording = useCallback(async () => {
    setMicError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: mime || "audio/webm" });
        const buf = await blob.arrayBuffer();
        const bytes = new Uint8Array(buf);
        let bin = "";
        for (let i = 0; i < bytes.length; i += 0x8000) {
          bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        }
        const audioB64 = btoa(bin);
        setRecording(false);
        await submit({ audioB64 });
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (err) {
      setMicError("Microphone unavailable — type your question instead.");
      setRecording(false);
    }
  }, [submit]);

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
  }, []);

  const cancel = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.onstop = null;
      recorderRef.current.stop();
      recorderRef.current.stream?.getTracks().forEach((t) => t.stop());
    }
    setRecording(false);
    setBusy(false);
  }, []);

  // ---- graph --------------------------------------------------------------
  const loadGraph = useCallback(() => {
    getGraph().then(setGraph).catch(() => setGraph(null));
  }, []);
  useEffect(() => {
    if (tab === "graph") loadGraph();
  }, [tab, loadGraph]);

  // voice flow labels for progressive stages
  const flowSteps = [
    { key: "transcript", label: "transcribing" },
    { key: "intent", label: "routing" },
    { key: "retrieval", label: "retrieving" },
    { key: "grounding", label: "grounding" },
    { key: "answer", label: "generating" },
  ];
  const activeStage = stages.length ? stages[stages.length - 1].event : null;

  return (
    <div className="app">
      <header>
        <h1>VakRAG</h1>
        <span className="sub">Voice-enabled Indic-language RAG · MSMARCO-XI</span>
        <span className={`health health-${health.status}`}>{health.status}</span>
      </header>

      <section className="hero">
        <div className="hero-left">
          <h2>Speak in any of 14 Indic languages — get a grounded answer in &lt;200ms</h2>
          <p>Local bge-m3 ANN (31,259 passages, 42–55ms P50) streams the extractive answer first; Vertex + Neo4j + RRF + LLM streams the full grounded answer after. Guardrails know when <em>not</em> to answer.</p>
          <div className="hero-badges">
            <span className="badge">14 languages</span>
            <span className="badge">Sarvam STT → Whisper fallback</span>
            <span className="badge">P50 45ms · P70 50ms · P100 80ms (70-query bench)</span>
            <span className="badge">31,259 local passages</span>
          </div>
        </div>
        <div className="hero-right">
          <div className="hero-card">
            <b>How it works</b>
            <ol>
              <li>Voice → Sarvam STT</li>
              <li>Local ANN → extractive span (200ms-compliant)</li>
              <li>Vertex/Neo4j RRF → grounding → LLM (progressive)</li>
            </ol>
            <span className="muted">Try: “किस राज्य की राजधानी मुंबई है” · “what is the capital of maharashtra”</span>
          </div>
        </div>
      </section>

      <nav>
        <button className={tab === "ask" ? "active" : ""} onClick={() => setTab("ask")}>Ask</button>
        <button className={tab === "latency" ? "active" : ""} onClick={() => setTab("latency")}>Latency</button>
        <button className={tab === "graph" ? "active" : ""} onClick={() => setTab("graph")}>Knowledge Graph</button>
      </nav>

      {tab === "ask" && (
        <section className="ask">
          <div className="controls">
            <input placeholder="Type a question in any Indic language…" value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit({ textValue: text })} />
            <select value={lang} onChange={(e) => setLang(e.target.value)}>
              <option value="">Lang auto</option>
              <option value="hi-IN">हिन्दी</option>
              <option value="bn-IN">বাংলা</option>
              <option value="ta-IN">தமிழ்</option>
              <option value="te-IN">తెలుగు</option>
              <option value="kn-IN">ಕನ್ನಡ</option>
              <option value="ml-IN">മലയാളം</option>
              <option value="mr-IN">मराठी</option>
              <option value="gu-IN">ગુજરાતી</option>
              <option value="pa-IN">ਪੰਜਾਬੀ</option>
              <option value="ur-IN">اردو</option>
              <option value="en">English</option>
            </select>
            <select value={mode} onChange={(e) => setMode(e.target.value)} title="Answer mode">
              {Object.entries(MODE_LABEL).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
          </div>

          <div className="actions">
            <button className="ask-btn" disabled={busy || !text.trim()} onClick={() => submit({ textValue: text })}>
              Ask
            </button>
            <button className={recording ? "mic-btn rec" : "mic-btn"} onClick={recording ? stopRecording : startRecording} disabled={busy && !recording}>
              {recording ? "■ Stop" : "🎙 Ask by voice"}
            </button>
            <button className="ghost" onClick={cancel} disabled={!busy && !recording}>Cancel</button>
          </div>
          {micError && <p className="error">{micError}</p>}

          {busy && (
            <div className="voice-flow">
              {flowSteps.map((s, i) => {
                const done = stages.some((x) => x.event === s.key || x.event === "quick" || x.event === "retrieval");
                const isActive = activeStage === s.key;
                return (
                  <span key={s.key} className={`flow-step ${done ? "done" : ""} ${isActive ? "active" : ""}`}>
                    {isActive ? "● " : done ? "✓ " : "○ "}{s.label}
                  </span>
                );
              })}
              <span className="flow-busy">● streaming…</span>
            </div>
          )}

          {quickAnswer && (
            <div className="answer-card quick-card">
              <div className="quick-head"><span className="badge-fast">⚡ Quick answer — {quickAnswer.latency_ms}ms (200ms-compliant, local bge-m3)</span> <span className="muted">grounding {quickAnswer.grounding_score?.toFixed(2)}</span></div>
              <p className="answer-text">{quickAnswer.text}</p>
              {quickAnswer.refusal_reason && <span className="badge-refusal">refusal: {quickAnswer.refusal_reason}</span>}
              <p className="muted">Streaming — full grounded answer follows…</p>
            </div>
          )}

          {answer && (
            <div className={`answer-card ${quickAnswer ? "full-card" : ""}`}>
              <div className="answer-head"><b>{quickAnswer ? "Full answer (progressive)" : "Answer"}</b> <span className="muted">· {totalMs != null ? `${totalMs}ms total` : ""}</span></div>
              <p className="answer-text">{answer.text}</p>
              <div className="answer-meta">
                <span>mode: {answer.mode}</span>
                {answer.mode === "extractive" && <span className="badge-fast">extractive = 200ms-compliant output</span>}
                <span>grounding: {answer.grounding_score?.toFixed(2)}</span>
                <span>confidence: {answer.confidence?.toFixed(2)}</span>
                {answer.refusal_reason && <span className="badge-refusal">refusal: {answer.refusal_reason}</span>}
              </div>
              {answer.citations?.length > 0 && (
                <details className="citations">
                  <summary>Citations ({answer.citations.length})</summary>
                  {answer.citations.map((c, i) => (
                    <blockquote key={i}>
                      <b>{c.passage_id.slice(0, 10)}</b> · {c.language_code} · {c.score?.toFixed(3)}
                      <br />{c.text.slice(0, 220)}
                    </blockquote>
                  ))}
                </details>
              )}
            </div>
          )}

          {candidates.length > 0 && (
            <div className="candidates">
              <h3>Retrieved passages</h3>
              {candidates.map((c, i) => (
                <div className="cand" key={c.id + i}>
                  <span className="cand-rank">#{i + 1}</span>
                  <span className="cand-src">{c.source}</span>
                  <span className="cand-lang">{c.lang || "?"}</span>
                  <span className="cand-score">{c.score.toFixed(3)}</span>
                  <p>{c.text}</p>
                </div>
              ))}
            </div>
          )}

          {stages.length > 0 && (
            <div className="stage-log">
              <h3>Stage stream {requestId && <span className="reqid">· {requestId} · {totalMs != null && `${totalMs} ms`}</span>}</h3>
              {stages.map((s, i) => (
                <div key={i} className={`stage stage-${s.event}`}>
                  <b>{s.event}</b>
                  <pre>{JSON.stringify(s.data, null, 0).slice(0, 400)}</pre>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {tab === "latency" && <LatencyPanel telemetry={telemetry} />}
      {tab === "graph" && <GraphPanel graph={graph} onLoad={loadGraph} />}

      <footer>
        HH Goa 2026 · Task 2 · SARVAM-GRP-… <a href="https://github.com/athmabhiram1/hhgoa_t2">repo</a>
      </footer>
    </div>
  );
}

function LatencyPanel({ telemetry }) {
  const totals = telemetry?.total_ms;
  const stages = telemetry?.stages || {};
  return (
    <section className="panel">
      <h2>Live latency</h2>
      <p className="muted">Request-path percentiles (ms), measured from the live /v1/ask traffic.</p>
      {!telemetry ? (
        <p className="muted">No traffic yet — ask a question first.</p>
      ) : (
        <table>
          <thead><tr><th>Metric</th><th>P50</th><th>P70</th><th>P100</th><th>n</th></tr></thead>
          <tbody>
            <tr className="total-row">
              <td>total</td>
              <td>{totals?.p50}</td><td>{totals?.p70}</td><td>{totals?.p100}</td><td>{totals?.n}</td>
            </tr>
            {Object.entries(stages).map(([name, s]) => (
              <tr key={name}>
                <td>{name}</td><td>{s.p50}</td><td>{s.p70}</td><td>{s.p100}</td><td>{s.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="muted">Budget: retrieval + extractive span answer ≤ 200 ms P50 — the extractive answer is THE 200ms-compliant output; the LLM answer streams as progressive enhancement, reported separately. STT reported separately (Sarvam &lt; 250 ms median).</p>
    </section>
  );
}

function GraphPanel({ graph, onLoad }) {
  const [layout, setLayout] = useState([]);
  useEffect(() => {
    if (!graph?.nodes?.length) return;
    const nodes = graph.nodes.map((n, i) => ({ ...n, x: 100 + Math.cos((i / graph.nodes.length) * Math.PI * 2) * 180, y: 100 + Math.sin((i / graph.nodes.length) * Math.PI * 2) * 180 }));
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const edges = graph.edges.filter((e) => byId.has(e.source) && byId.has(e.target));
    setLayout({ nodes, edges });
  }, [graph]);

  if (!graph || !graph.nodes?.length) {
    return (
      <section className="panel">
        <h2>Knowledge graph</h2>
        <p className="muted">No graph data — is the index built?</p>
        <button onClick={onLoad}>Refresh</button>
      </section>
    );
  }
  return (
    <section className="panel">
      <h2>Knowledge graph <span className="muted">({graph.nodes.length} nodes, {graph.edges.length} edges)</span></h2>
      <div className="graph-wrap">
        <svg viewBox="0 0 460 260" width="100%" height="320">
          {layout.edges.map((e, i) => (
            <line key={i} x1={layout.nodes.find((n) => n.id === e.source)?.x} y1={layout.nodes.find((n) => n.id === e.source)?.y} x2={layout.nodes.find((n) => n.id === e.target)?.x} y2={layout.nodes.find((n) => n.id === e.target)?.y} className="edge" />
          ))}
          {layout.nodes.map((n, i) => (
            <g key={i} transform={`translate(${n.x},${n.y})`}>
              <circle r={n.type === "query" ? 7 : 5} className={`node-${n.type}`} />
              <text y={-10} textAnchor="middle" className="node-label">{String(n.label).slice(0, 14)}</text>
            </g>
          ))}
        </svg>
      </div>
      <p className="muted">Query (blue) → query_type (green) → language (orange). Blue→Chunk edges power sibling expansion.</p>
    </section>
  );
}