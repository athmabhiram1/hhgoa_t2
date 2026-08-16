// API client: SSE streaming for /v1/ask + JSON endpoints.

export async function askSSE({ text, audioB64, lang, mode, onEvent, signal }) {
  const body = {};
  if (text) body.text = text;
  if (audioB64) body.audio_b64 = audioB64;
  if (lang) body.lang = lang;
  if (mode) body.mode = mode;

  const res = await fetch("/v1/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const events = buf.split("\n\n");
    buf = events.pop();
    for (const chunk of events) {
      let event = "message";
      let data = "";
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) {
        try {
          onEvent(event, JSON.parse(data));
        } catch {
          /* ignore malformed frame */
        }
      }
    }
  }
}

export async function getTelemetry() {
  const res = await fetch("/v1/telemetry");
  if (!res.ok) throw new Error("telemetry unavailable");
  return res.json();
}

export async function getGraph() {
  const res = await fetch("/v1/graph");
  if (!res.ok) throw new Error("graph unavailable");
  return res.json();
}

export async function getHealth() {
  try {
    const res = await fetch("/v1/health");
    return res.ok ? await res.json() : { status: "degraded" };
  } catch {
    return { status: "down" };
  }
}