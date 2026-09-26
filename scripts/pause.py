"""Met en pause / relance les calculs en cours (pipeline, entraînements, analyses), sans rien perdre.

Suspend les processus (comme un bouton pause) : ils reprennent exactement où ils étaient.
La mémoire du GPU reste réservée pendant la pause.

  .venv\\Scripts\\python scripts/pause.py pause
  .venv\\Scripts\\python scripts/pause.py resume
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys

PATTERNS = ("pipeline.py", "finetune_detector.py", "train_yolo.py", "train_v2.py", "train_placement.py",
            "extract_videos.py", "pseudo_label.py", "night_videos.py", "multiprocessing", "yt-dlp")


def targets() -> list[tuple[int, str]]:
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','yt-dlp.exe' } | "
          "Select-Object ProcessId, CommandLine | ConvertTo-Json")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    rows = json.loads(out) if out.strip() else []
    rows = rows if isinstance(rows, list) else [rows]
    return [(r["ProcessId"], r["CommandLine"] or "") for r in rows
            if any(p in (r["CommandLine"] or "") for p in PATTERNS) and "label_tool" not in (r["CommandLine"] or "")]


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "pause"
    ntdll, k32 = ctypes.windll.ntdll, ctypes.windll.kernel32
    fn = ntdll.NtSuspendProcess if action == "pause" else ntdll.NtResumeProcess
    n = 0
    for pid, cmd in targets():
        h = k32.OpenProcess(0x0800, False, pid)       # PROCESS_SUSPEND_RESUME
        if h:
            fn(h)
            k32.CloseHandle(h)
            n += 1
    print(f"{'en pause' if action == 'pause' else 'relancés'} : {n} processus")


if __name__ == "__main__":
    main()
