"""Page locale pour corriger les pré-étiquettes (phase C du détecteur).

  .venv-yolo\\Scripts\\python scripts/label_tool.py      puis ouvrir http://localhost:8765

Clic sur une boîte : la sélectionner. Suppr : l'effacer. C : changer de camp (bleu = nous, rouge = eux).
Glisser sur une zone vide : dessiner une boîte (carte = celle choisie dans la liste).
Entrée : valider l'image et passer à la suivante. Échap : ignorer l'image (floue, menu…).
Les images validées vont dans D:/clash-ai-dataset/real/{images,labels} au format YOLO (classes unité_camp).
Une carte inconnue du modèle (sortie après avril 2024) est gardée : elle devient une nouvelle classe, ajoutée à la
fin de D:/clash-ai-dataset/real/classes.json (les numéros existants ne bougent pas).
"""
from __future__ import annotations

import json
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

DATA = Path("D:/clash-ai-dataset")
REAL = DATA / "real"
# Liste des classes des images réelles : celles du modèle (data.yaml), puis les NOUVELLES cartes ajoutées ici
# (sorties après les sprites de KataCR, avril 2024). Les numéros ne changent jamais : on ajoute à la fin.
CLASSES_FILE = REAL / "classes.json"


def load_classes() -> list[str]:
    if CLASSES_FILE.exists():
        return json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
    names = yaml.safe_load(open(DATA / "data.yaml"))["names"]
    return [names[i] for i in sorted(names)]


CLASSES = load_classes()


def class_id(key: str) -> int:
    """Numéro de la classe « unité_camp » ; une carte inconnue devient une nouvelle classe."""
    if key not in CLASSES:
        CLASSES.append(key)
        REAL.mkdir(parents=True, exist_ok=True)
        CLASSES_FILE.write_text(json.dumps(CLASSES, ensure_ascii=False, indent=0), encoding="utf-8")
        print("nouvelle carte ajoutée :", key)
    return CLASSES.index(key)


def units() -> list[str]:
    return sorted({n.rsplit("_", 1)[0] for n in CLASSES})

PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Étiquetage</title>
<style>
body{font-family:system-ui;background:#1b1d22;color:#eee;margin:0;display:flex;gap:16px;padding:12px}
canvas{background:#000;cursor:crosshair}
#side{width:300px}#side button{margin:3px 0;width:100%;padding:8px;font-size:14px}
input{width:100%;padding:6px;font-size:14px}.k{color:#9ab}#list div{padding:2px 4px;cursor:pointer}
#list div.sel{background:#345}
</style></head><body>
<canvas id="c" width="568" height="896"></canvas>
<div id="side">
<div id="stat"></div><div id="hint" style="background:#553;padding:6px;margin:6px 0;display:none"></div>
<p>Carte pour les nouvelles boîtes / la boîte sélectionnée<br><span class="k">(une carte absente de la liste :
tape son nom, ex. « goblin-machine » — elle devient une nouvelle classe)</span></p>
<input id="cls" list="units" placeholder="ex. knight"><datalist id="units"></datalist>
<button id="apply">Renommer la sélection</button>
<button id="flip">Changer de camp (C)</button>
<button id="del">Supprimer (Suppr)</button>
<button id="save">Valider et suivante (Entrée)</button>
<button id="skip">Ignorer l'image (Échap)</button>
<p class="k">Bleu = nos unités, rouge = ennemies. Glisser pour dessiner une boîte.</p>
<div id="list"></div></div>
<script>
const c=document.getElementById('c'),g=c.getContext('2d');let img=new Image(),cur=null,boxes=[],sel=-1,drag=null;
const W=568,H=896;
async function next(){const r=await (await fetch('/api/next')).json();document.getElementById('stat').textContent=
 r.name?`${r.left} image(s) à corriger — ${r.done} validée(s)`:`Terminé — ${r.done} validée(s)`;
 const h=document.getElementById('hint');if(r.hint){h.style.display='block';h.innerHTML=`Nouvelle carte à chercher : <b>${r.hint}</b><br>Encadre-la (tape « ${r.hint} » dans la liste), sinon Échap si elle n'est pas à l'écran.`;document.getElementById('cls').value=r.hint}else h.style.display='none';
 if(!r.name){g.clearRect(0,0,W,H);return}cur=r.name;boxes=r.boxes;sel=-1;img.onload=draw;img.src='/img/'+r.name+'.jpg'}
function draw(){g.drawImage(img,0,0,W,H);boxes.forEach((b,i)=>{const[x0,y0,x1,y1]=b.box;
 g.strokeStyle=b.side?'#f44':'#4af';g.lineWidth=i==sel?3:1.5;g.strokeRect(x0*W,y0*H,(x1-x0)*W,(y1-y0)*H);
 g.fillStyle=g.strokeStyle;g.font='11px system-ui';g.fillText(b.name,x0*W+2,y0*H-3)});
 if(drag&&drag.b){const[x0,y0,x1,y1]=drag.b;g.strokeStyle='#ff0';g.strokeRect(x0*W,y0*H,(x1-x0)*W,(y1-y0)*H)}
 document.getElementById('list').innerHTML=boxes.map((b,i)=>`<div class="${i==sel?'sel':''}" onclick="sel=${i};draw()">${b.side?'🔴':'🔵'} ${b.name}</div>`).join('')}
function pos(e){const r=c.getBoundingClientRect();return[(e.clientX-r.left)/W,(e.clientY-r.top)/H]}
c.onmousedown=e=>{const[x,y]=pos(e);const i=boxes.findIndex(b=>x>=b.box[0]&&x<=b.box[2]&&y>=b.box[1]&&y<=b.box[3]);
 if(i>=0){sel=i;document.getElementById('cls').value=boxes[i].name;draw()}else drag={x,y,b:null}};
c.onmousemove=e=>{if(!drag)return;const[x,y]=pos(e);drag.b=[Math.min(x,drag.x),Math.min(y,drag.y),Math.max(x,drag.x),Math.max(y,drag.y)];draw()};
c.onmouseup=e=>{if(drag&&drag.b&&(drag.b[2]-drag.b[0])*W>6){const n=document.getElementById('cls').value.trim();
 if(!n){alert('Choisis la carte dans la liste à droite');}else{boxes.push({name:n,side:drag.b[1]<0.5?1:0,box:drag.b});sel=boxes.length-1}}drag=null;draw()};
const act={del(){if(sel>=0){boxes.splice(sel,1);sel=-1;draw()}},flip(){if(sel>=0){boxes[sel].side^=1;draw()}},
 apply(){const n=document.getElementById('cls').value.trim();if(sel>=0&&n){boxes[sel].name=n;draw()}},
 async save(){await fetch('/api/save',{method:'POST',body:JSON.stringify({name:cur,boxes})});next()},
 async skip(){await fetch('/api/skip',{method:'POST',body:JSON.stringify({name:cur})});next()}};
for(const k in act)document.getElementById(k).onclick=act[k];
document.onkeydown=e=>{if(e.target.tagName=='INPUT')return;if(e.key=='Delete'||e.key=='Backspace')act.del();
 else if(e.key=='c'||e.key=='C')act.flip();else if(e.key=='Enter')act.save();else if(e.key=='Escape')act.skip()};
fetch('/api/units').then(r=>r.json()).then(u=>document.getElementById('units').innerHTML=u.map(n=>`<option value="${n}">`).join(''));
next();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/units":
            self._json(units())
        elif self.path == "/api/next":
            pending = sorted((REAL / "pending").glob("*.json"))
            done = len(list((REAL / "labels").glob("*.txt"))) if (REAL / "labels").exists() else 0
            if not pending:
                return self._json({"name": None, "left": 0, "done": done})
            p = pending[0]
            hint = p.with_suffix(".hint")
            self._json({"name": p.stem, "boxes": json.loads(p.read_text()), "left": len(pending), "done": done,
                        "hint": hint.read_text().strip() if hint.exists() else None})
        elif self.path.startswith("/img/"):
            f = REAL / "pending" / Path(self.path[5:]).name
            if not f.exists():
                return self._json({"error": "introuvable"}, 404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "introuvable"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = Path(req["name"]).name
        src_img, src_json = REAL / "pending" / f"{name}.jpg", REAL / "pending" / f"{name}.json"
        if self.path == "/api/save":
            (REAL / "images").mkdir(parents=True, exist_ok=True)
            (REAL / "labels").mkdir(parents=True, exist_ok=True)
            lines = []
            for b in req["boxes"]:
                name = b["name"].strip().lower().replace(" ", "-")
                if not name:
                    continue
                key = f"{name}_{int(b['side'])}"
                x0, y0, x1, y1 = b["box"]
                lines.append(f"{class_id(key)} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
            (REAL / "labels" / f"{name}.txt").write_text("\n".join(lines))
            shutil.move(src_img, REAL / "images" / f"{name}.jpg")
            src_json.unlink(missing_ok=True)
            src_json.with_suffix(".hint").unlink(missing_ok=True)
        elif self.path == "/api/skip":
            (REAL / "skipped").mkdir(parents=True, exist_ok=True)
            if src_img.exists():
                shutil.move(src_img, REAL / "skipped" / f"{name}.jpg")
            src_json.unlink(missing_ok=True)
            src_json.with_suffix(".hint").unlink(missing_ok=True)
        self._json({"ok": True})


if __name__ == "__main__":
    print("Étiquetage : http://localhost:8765  (Ctrl+C pour arrêter)")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
