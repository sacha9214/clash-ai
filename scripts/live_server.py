"""Page de suivi à distance : ouvrir http://<ip du PC>:8080 depuis un autre ordi du même réseau Wi-Fi.

Montre en direct ce que voit l'IA (image annotée, 1 fois/s), l'état de la boucle (en match, en pause, bloquée)
et le score des matchs du jour. Lecture seule : rien ne peut être commandé depuis la page.

  .venv\\Scripts\\python scripts/live_server.py            (port 8080)
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "runs/live.jpg"
LOG = Path(os.environ.get("TEMP", ".")) / "pl.log"
JOURNAL = ROOT / "runs/games/journal.jsonl"

PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Clash AI en direct</title>
<style>
 :root { --bg:#10131a; --fg:#e8ecf3; --mut:#8a93a6; --ok:#3fb950; --bad:#f85149; --card:#1a1f2b; }
 body { margin:0; background:var(--bg); color:var(--fg); font:15px system-ui, sans-serif; }
 main { display:flex; gap:20px; padding:16px; flex-wrap:wrap; justify-content:center; }
 img { max-height:88vh; max-width:100%; min-width:280px; min-height:400px; border-radius:10px; background:#000; }
 img:not([src]) { visibility:hidden; }
 .side { min-width:260px; max-width:420px; flex:1; }
 .card { background:var(--card); border-radius:10px; padding:12px 14px; margin-bottom:12px; }
 h1 { font-size:18px; margin:0 0 8px; } .mut { color:var(--mut); font-size:13px; }
 .w { color:var(--ok); } .l { color:var(--bad); } pre { white-space:pre-wrap; margin:0; font-size:13px; }
</style></head><body><main>
<img id="live" alt="écran de l'IA">
<div class="side">
 <div class="card"><h1>État</h1><div id="state">…</div><div class="mut" id="age"></div></div>
 <div class="card"><h1>Aujourd'hui</h1><div id="score">…</div></div>
 <div class="card"><h1>Derniers événements</h1><pre id="log">…</pre></div>
 <div class="card"><h1>Revues des matchs</h1><div id="reviews" class="mut">…</div></div>
</div></main>
<script>
let lastAge = null, loading = false;
function refreshImage(s) {
  // image chargée en arrière-plan puis échangée d'un coup : pas de clignotement ; rien tant qu'il n'y a pas d'image
  if (s.image_age == null || loading || s.image_age === lastAge && s.image_age > 3) return;
  loading = true;
  const img = new Image();
  img.onload = () => { document.getElementById('live').src = img.src; loading = false; };
  img.onerror = () => { loading = false; };
  img.src = '/live.jpg?t=' + Date.now();
  lastAge = s.image_age;
}
async function tick() {
  try {
    const s = await (await fetch('/status')).json();
    refreshImage(s);
    document.getElementById('state').textContent = s.state;
    document.getElementById('age').textContent = s.image_age == null ? 'pas encore d\\'image'
        : 'image d\\'il y a ' + s.image_age + ' s';
    document.getElementById('score').innerHTML = '<span class="w">' + s.wins + ' victoires</span> · <span class="l">'
        + s.losses + ' défaites</span> · ' + s.other + ' autres';
    document.getElementById('log').textContent = s.log.join('\\n');
    const esc = v => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    const html = s.reviews.map(r => {
      const res = r.result === 'win' ? '<span class="w">victoire</span>' : r.result === 'loss'
          ? '<span class="l">défaite</span>' : esc(r.result || '?');
      const notes = r.notes.map(n => '<li>' + esc(n) + '</li>').join('');
      return '<div style="margin-bottom:10px"><b>' + r.game.slice(9, 11) + 'h' + r.game.slice(11, 13) + '</b> · '
          + res + ' · adversaire ' + esc(r.ctx || '?') + '<div>' + esc(r.facts) + '</div>'
          + (notes ? '<ul style="margin:4px 0 0 18px;padding:0;color:var(--fg)">' + notes + '</ul>' : '') + '</div>';
    }).join('');
    const box = document.getElementById('reviews');
    if (box.dataset.html !== html) { box.innerHTML = html || 'aucune revue'; box.dataset.html = html; }
  } catch (e) { document.getElementById('state').textContent = 'PC injoignable'; }
}
tick(); setInterval(tick, 1000);
</script></body></html>"""


def status() -> dict:
    lines = []
    if LOG.exists():
        text = LOG.read_bytes()[-200_000:].decode("utf-8", "replace").replace("\r", "\n")
        keep = ("=== ", "adversaire de type", "[scanner]", "autoplay en échec", "inconnu", "Traceback", "combat 1/1")
        lines = [l.strip() for l in text.splitlines() if any(k in l for k in keep)][-12:]
    last = lines[-1] if lines else ""
    age = round(time.time() - LIVE.stat().st_mtime) if LIVE.exists() else None
    if "pause revue" in last:
        state = "⏸ pause revue (correction des erreurs du dernier match)"
    elif "échec" in last or "inconnu" in last or "Traceback" in last:
        state = "⚠ boucle arrêtée ou bloquée : " + last
    elif age is not None and age < 10:
        state = "▶ en match"
    elif "combat" in last or "=== match" in last:
        state = "▶ recherche d'adversaire / menu"
    else:
        state = "… en attente"
    today = time.strftime("%Y%m%d")
    w = l = o = 0
    if JOURNAL.exists():
        for line in JOURNAL.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if str(j.get("game", "")).startswith(today):
                r = j.get("result")
                w, l, o = w + (r == "win"), l + (r == "loss"), o + (r not in ("win", "loss"))
    return {"state": state, "image_age": age, "wins": w, "losses": l, "other": o, "log": lines, "reviews": reviews()}


def reviews(n: int = 8) -> list[dict]:
    """Revue de chaque match (la plus récente d'abord) : bilan automatique (runs/lessons.jsonl) + ce que Claude a
    constaté et corrigé (runs/reviews.jsonl : {"game", "note"})."""
    def rows(f):
        if not f.exists():
            return []
        out = []
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out
    notes = {}
    for r in rows(ROOT / "runs/reviews.jsonl"):
        if r.get("game") and r.get("note"):
            notes.setdefault(r["game"], []).append(r["note"])
    out = []
    for les in [x for x in rows(ROOT / "runs/lessons.jsonl") if x.get("game")][-n:][::-1]:
        g = les["game"]
        out.append({"game": g, "result": les.get("result"), "ctx": les.get("ctx"),
                    "facts": f"{les.get('giants', 0)} Géants · {les.get('slowed_only', 0)} défenses qui ralentissent "
                             f"seulement · {les.get('ghost_defenses', 0)} défenses contre des fantômes",
                    "notes": notes.get(g, [])})
    return out


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            body, ctype = PAGE.encode("utf-8"), "text/html; charset=utf-8"
        elif path == "/live.jpg" and LIVE.exists():
            body, ctype = LIVE.read_bytes(), "image/jpeg"
        elif path == "/status":
            body, ctype = json.dumps(status(), ensure_ascii=False).encode("utf-8"), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.1.1", 80))          # aucun paquet envoyé : sert seulement à connaître l'IP locale
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"suivi à distance : http://{lan_ip()}:{port}  (même réseau Wi-Fi)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
