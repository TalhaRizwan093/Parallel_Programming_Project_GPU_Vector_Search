from __future__ import annotations

import argparse
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string, request

HTML = r"""
<!doctype html>
<html><head>
<title>GPU Search Demo — Wikipedia 1M</title>
<meta charset="utf-8">
<style>
:root { --primary: #1F3A68; --accent: #4caf50; --bg: #f7f9fc; --card: #ffffff; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: var(--bg); color: #1c1c1c; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
.header { background: var(--primary); color: white; padding: 20px 24px; border-radius: 12px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }
.header h1 { margin: 0 0 4px 0; font-size: 24px; }
.header .sub { font-size: 13px; opacity: 0.85; }
.search-box { background: var(--card); padding: 20px; border-radius: 12px; margin-bottom: 16px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
textarea { width: 100%; min-height: 70px; font-size: 15px; padding: 10px; border: 1px solid #ccd; border-radius: 6px; font-family: inherit; resize: vertical; }
.row { display: flex; align-items: center; gap: 12px; margin-top: 10px; flex-wrap: wrap; }
button { padding: 9px 22px; font-size: 14px; background: var(--primary); color: white; border: 0; border-radius: 6px; cursor: pointer; font-weight: 600; }
button:hover { background: #15294a; }
button:disabled { background: #aaa; cursor: not-allowed; }
select { padding: 8px; font-size: 14px; border: 1px solid #ccd; border-radius: 6px; }
.stats { display: flex; gap: 24px; font-size: 13px; color: #555; padding: 12px 20px; background: white; border-radius: 8px; margin-bottom: 16px; }
.stats span b { color: var(--primary); font-size: 16px; }
.result { background: var(--card); border-left: 4px solid var(--primary); padding: 14px 18px; margin: 10px 0; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.result .meta { color: #777; font-size: 12px; margin-bottom: 6px; display: flex; gap: 16px; }
.result .meta .score { color: var(--accent); font-weight: 600; }
.result .text { font-size: 14px; line-height: 1.55; }
.loading { text-align: center; padding: 40px; color: #888; }
.examples { font-size: 13px; color: #555; margin-top: 6px; }
.examples a { color: var(--primary); text-decoration: none; margin: 0 6px; }
.examples a:hover { text-decoration: underline; }
</style>
</head><body>
<div class="container">
  <div class="header">
    <h1>GPU-Powered Wikipedia Search</h1>
    <div class="sub">1 000 000 paragraphs · 384-d MiniLM embeddings · custom CUDA kernels on RTX 3070</div>
  </div>

  <div class="search-box">
    <form id="f" onsubmit="event.preventDefault(); search();">
      <textarea id="q" placeholder="Type your question, e.g. 'how do birds know where to fly in winter?'"></textarea>
      <div class="row">
        <button id="go">🔍 Search</button>
        <span>Engine version:</span>
        <select id="v">
          <option value="v1">V1 — naive (5.5 ms/q)</option>
          <option value="v2">V2 — coalesced (1.4 ms/q)</option>
          <option value="v3" selected>V3.1 — PQ fat-chunks (0.48 ms/q)</option>
        </select>
      </div>
      <div class="examples">
        Try: <a href="#" onclick="setQuery('how do birds know where to fly in winter')">bird migration</a>
        <a href="#" onclick="setQuery('first programmable computer in history')">computing history</a>
        <a href="#" onclick="setQuery('photosynthesis in plants')">plant biology</a>
        <a href="#" onclick="setQuery('quantum entanglement explained simply')">quantum physics</a>
      </div>
    </form>
  </div>

  <div id="stats" class="stats" style="display:none">
    <span>Search latency: <b id="lat">-</b> ms</span>
    <span>Embed latency: <b id="emb">-</b> ms</span>
    <span>Total: <b id="tot">-</b> ms</span>
    <span>Engine: <b id="vshow">-</b></span>
  </div>
  <div id="results"></div>
</div>

<script>
function setQuery(t) {
  document.getElementById('q').value = t;
  search();
}
async function search() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const v = document.getElementById('v').value;
  document.getElementById('go').disabled = true;
  document.getElementById('results').innerHTML = '<div class="loading">Searching ' + (v === 'v3' ? '32 MB compressed' : '1.54 GB') + ' index on GPU…</div>';
  document.getElementById('stats').style.display = 'none';
  try {
    const res = await fetch('/search', {method:'POST', headers:{'Content-Type':'application/json'},
                                         body: JSON.stringify({query:q, version:v})});
    const j = await res.json();
    document.getElementById('lat').innerText = j.search_ms.toFixed(2);
    document.getElementById('emb').innerText = j.embed_ms.toFixed(2);
    document.getElementById('tot').innerText = (j.search_ms + j.embed_ms).toFixed(2);
    document.getElementById('vshow').innerText = v.toUpperCase();
    document.getElementById('stats').style.display = 'flex';
    const html = j.results.map((r,i) =>
      `<div class="result">
         <div class="meta">
           <span>#${i+1} · id ${r.id}</span>
           <span class="score">distance ${r.score.toFixed(4)}</span>
         </div>
         <div class="text">${escapeHTML(r.text)}</div>
       </div>`).join('');
    document.getElementById('results').innerHTML = html || '<div class="loading">No results.</div>';
  } catch (e) {
    document.getElementById('results').innerHTML = '<div class="loading">Error: ' + e + '</div>';
  } finally {
    document.getElementById('go').disabled = false;
  }
}
function escapeHTML(s) { return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
</script>
</body></html>
"""

def load_text(path: Path) -> list[str]:
    if not path.exists():
        return [f"<paragraph {i}>" for i in range(1_000_000)]
    print(f"[demo] loading paragraph text from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        texts = f.read().splitlines()
    print(f"[demo] loaded {len(texts):,} paragraphs")
    return texts

def write_query_bin(emb: np.ndarray, path: Path) -> None:

    n, d = 1, emb.shape[-1]
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, d))
        emb.reshape(-1).astype(np.float32).tofile(f)

def read_topk_bin(path: Path):

    with open(path, "rb") as f:
        raw = f.read()
    n_q = int.from_bytes(raw[0:4], "little", signed=True)
    K = int.from_bytes(raw[4:8], "little", signed=True)
    payload = raw[8:]
    per_q = K * 4 + K * 4
    ids = np.frombuffer(payload[0:K*4], dtype=np.int32)
    dists = np.frombuffer(payload[K*4:per_q], dtype=np.float32)
    return ids.tolist(), dists.tolist()

def make_app(root: Path) -> Flask:
    app = Flask(__name__)
    texts = load_text(root / "data" / "wiki.txt")

    print("[demo] loading MiniLM-L6 on GPU for query embedding...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    print(f"[demo] embedder on {device}")

    engine = root / "build" / "search"
    wiki = root / "data" / "wiki.bin"
    pq = root / "data" / "wiki_pq.bin"
    tmpdir = Path(tempfile.gettempdir())

    @app.get("/")
    def index():
        return render_template_string(HTML)

    @app.post("/search")
    def search():
        body = request.get_json(force=True)
        q = body["query"]
        version = body.get("version", "v3")

        t_emb_0 = time.perf_counter()
        emb = model.encode([q], normalize_embeddings=True, show_progress_bar=False)
        embed_ms = (time.perf_counter() - t_emb_0) * 1000.0

        qbin = tmpdir / f"demo_q_{os.getpid()}.bin"
        outbin = tmpdir / f"demo_out_{os.getpid()}.bin"
        write_query_bin(emb, qbin)

        args = [
            str(engine),
            "--db", str(wiki),
            "--version", version,
            "--queries", str(qbin),
            "--out", str(outbin),
            "--warmup", "0",
            "--runs", "1",
            "--bench",
        ]
        if version == "v3":
            args.extend(["--pq", str(pq)])
        t_search_0 = time.perf_counter()
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        search_ms = (time.perf_counter() - t_search_0) * 1000.0
        if proc.returncode != 0:
            return jsonify({"error": "engine failed",
                            "stdout": proc.stdout, "stderr": proc.stderr}), 500

        ids, dists = read_topk_bin(outbin)

        kernel_ms = None
        for line in proc.stdout.splitlines():
            if line.strip().startswith("median ="):
                try:
                    kernel_ms = float(line.split("=")[1].split("ms")[0].strip())
                except Exception:
                    pass

        return jsonify({
            "search_ms": kernel_ms if kernel_ms is not None else search_ms,
            "embed_ms": embed_ms,
            "results": [
                {"id": int(i), "score": float(d), "text": texts[int(i)][:400]}
                for i, d in zip(ids, dists)
            ],
        })

    return app

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    app = make_app(args.root.resolve())
    print(f"[demo] visit http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
