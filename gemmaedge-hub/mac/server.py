"""
Mac Mini Core Reasoning Server — GemmaEdge Hub

Runs Gemma 4 26B (MoE) via Ollama. Handles escalated requests from any edge device
(Raspberry Pi or MacBook Air) and serves a live dashboard at http://localhost:8000

Why Gemma 4 26B here:
  - Mac Mini (24 GB unified RAM) fits the model comfortably (~18 GB)
  - Apple Metal gives ~15–20 tok/s — fast enough for real-time escalation
  - 256K context window for detailed multi-object scene analysis
"""

import base64
import logging
import time
from collections import deque
from datetime import datetime

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from shared.protocol import EdgeRequest, EdgeResponse

# ── config ────────────────────────────────────────────────────────────────────
CORE_MODEL = "gemma4:26b"
HOST = "0.0.0.0"
PORT = 8000
MAX_HISTORY = 20            # events shown on dashboard

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="GemmaEdge Mac Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory event log (no persistence needed for the demo)
event_log: deque[dict] = deque(maxlen=MAX_HISTORY)


# ── reasoning endpoint ────────────────────────────────────────────────────────

@app.post("/reason", response_model=EdgeResponse)
def reason(req: EdgeRequest) -> EdgeResponse:
    log.info("[%s] %s request from edge device (edge_conf=%.2f)", req.session_id, req.modality, req.local_confidence)

    user_content = req.prompt
    if req.text:
        user_content = (
            f"The edge device gave this initial answer (confidence {req.local_confidence:.0%}):\n"
            f'"{req.text}"\n\n'
            f"Provide a more detailed and accurate answer:\n{req.prompt}"
        )

    message: dict = {"role": "user", "content": user_content}
    if req.modality == "vision" and req.image_b64:
        message["images"] = [base64.b64decode(req.image_b64)]

    t0 = time.perf_counter()
    try:
        response = ollama.chat(
            model=CORE_MODEL,
            messages=[message],
            options={"num_ctx": 128_000},
        )
    except Exception as exc:
        log.error("[%s] Ollama error: %s", req.session_id, exc)
        raise HTTPException(status_code=503, detail=str(exc))

    elapsed = time.perf_counter() - t0
    answer = response["message"]["content"]
    tokens = response.get("eval_count", 0)
    confidence = min(0.95, 0.6 + (tokens / 500) * 0.3)

    log.info("[%s] Done in %.1fs, %d tokens", req.session_id, elapsed, tokens)

    # Record for dashboard
    event_log.appendleft({
        "time": datetime.now().strftime("%H:%M:%S"),
        "session_id": req.session_id,
        "edge_answer": req.text or "—",
        "edge_conf": round(req.local_confidence, 2),
        "mac_answer": answer[:600],
        "mac_conf": round(confidence, 2),
        "elapsed_s": round(elapsed, 1),
        "image_b64": req.image_b64,   # pass through so dashboard can show thumbnail
    })

    return EdgeResponse(
        session_id=req.session_id,
        answer=answer,
        confidence=confidence,
        tokens_used=tokens,
        model=CORE_MODEL,
    )


# ── status endpoint (polled by dashboard) ────────────────────────────────────

@app.get("/status")
def status() -> dict:
    return {"model": CORE_MODEL, "events": list(event_log)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": CORE_MODEL}


# ── dashboard ─────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GemmaEdge Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 24px; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .subtitle { color: #94a3b8; font-size: 0.85rem; margin-bottom: 24px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 0.75rem; font-weight: 600; }
  .live { background: #16a34a; color: #fff; }
  .waiting { background: #475569; color: #cbd5e1; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; padding: 8px 12px; background: #1e293b; color: #94a3b8;
       font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: .05em; }
  td { padding: 10px 12px; border-bottom: 1px solid #1e293b; vertical-align: top; }
  tr:hover td { background: #1e293b; }
  .conf { font-weight: 700; }
  .high { color: #4ade80; }
  .mid  { color: #facc15; }
  .low  { color: #f87171; }
  .answer { max-width: 320px; white-space: pre-wrap; word-break: break-word; }
  .thumb { width: 80px; height: 60px; object-fit: cover; border-radius: 4px; }
  .empty { text-align: center; padding: 48px; color: #475569; }
  #status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block;
                background: #4ade80; margin-right: 6px; }
</style>
</head>
<body>
<h1><span id="status-dot"></span>GemmaEdge Hub</h1>
<p class="subtitle">Edge device → Mac Mini escalation log &nbsp;·&nbsp; refreshes every 2s</p>
<div id="model-label" style="margin-bottom:16px; color:#64748b; font-size:0.8rem;"></div>
<table>
  <thead>
    <tr>
      <th>Time</th>
      <th>ID</th>
      <th>Frame</th>
      <th>Edge answer</th>
      <th>Edge conf</th>
      <th>Mac answer</th>
      <th>Mac conf</th>
      <th>Latency</th>
    </tr>
  </thead>
  <tbody id="rows"><tr><td colspan="8" class="empty">Waiting for events from edge device…</td></tr></tbody>
</table>

<script>
function confClass(v) {
  return v >= 0.75 ? 'high' : v >= 0.55 ? 'mid' : 'low';
}

async function refresh() {
  try {
    const data = await fetch('/status').then(r => r.json());
    document.getElementById('model-label').textContent = 'Core model: ' + data.model;

    const tbody = document.getElementById('rows');
    if (!data.events || data.events.length === 0) return;

    tbody.innerHTML = data.events.map(e => {
      const img = e.image_b64
        ? `<img class="thumb" src="data:image/jpeg;base64,${e.image_b64}" alt="frame">`
        : '—';
      const edgeConf = `<span class="conf ${confClass(e.edge_conf)}">${e.edge_conf}</span>`;
      const macConf  = `<span class="conf ${confClass(e.mac_conf)}">${e.mac_conf}</span>`;
      return `<tr>
        <td>${e.time}</td>
        <td style="font-family:monospace;color:#64748b">${e.session_id}</td>
        <td>${img}</td>
        <td class="answer">${e.edge_answer}</td>
        <td>${edgeConf}</td>
        <td class="answer">${e.mac_answer}</td>
        <td>${macConf}</td>
        <td>${e.elapsed_s}s</td>
      </tr>`;
    }).join('');
  } catch(err) {
    document.getElementById('status-dot').style.background = '#f87171';
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    log.info("Dashboard → http://localhost:%d", PORT)
    uvicorn.run(app, host=HOST, port=PORT)
