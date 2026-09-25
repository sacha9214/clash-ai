"""Transcrit des vidéos d'explication (stratégie Clash Royale) avec faster-whisper sur GPU.

Écrit un .txt horodaté à côté de chaque fichier audio ; on en tire ensuite des règles pour brain.py.

  .venv\\Scripts\\python scripts/transcribe.py D:/clash-ai-videos/explain/*.m4a
"""
import glob
import os
import sys
import time
from pathlib import Path

# Windows : les DLL CUDA (cuBLAS, cuDNN) viennent des paquets pip nvidia-*
if sys.platform == "win32":
    import nvidia
    for bin_dir in glob.glob(os.path.join(list(nvidia.__path__)[0], "*", "bin")):
        os.add_dll_directory(bin_dir)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel  # noqa: E402

files = [f for a in sys.argv[1:] for f in glob.glob(a)]
if not files:
    sys.exit(__doc__)
model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
for f in files:
    out = Path(f).with_suffix(".txt")
    if out.exists():
        print("déjà fait :", out)
        continue
    t = time.time()
    segments, info = model.transcribe(f, language="en", vad_filter=True)
    with open(out, "w", encoding="utf-8") as fh:
        for s in segments:
            fh.write(f"[{int(s.start // 60):02d}:{int(s.start % 60):02d}] {s.text.strip()}\n")
    print(f"{Path(f).name} : {info.duration / 60:.1f} min transcrites en {time.time() - t:.0f} s -> {out.name}")
