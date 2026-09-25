"""Yeux et mains : flux vidéo scrcpy (H.264 brut) + injection de touches.

On lance nous-mêmes le serveur scrcpy sur le téléphone plutôt que l'appli
scrcpy : on récupère ainsi les images décodées en Python sans fenêtre
intermédiaire, et on envoie les taps par la socket de contrôle déjà ouverte
(≈ quelques ms) au lieu de `adb shell input` (≈ 50 ms par tap).
"""
from __future__ import annotations

import glob

import os
import random
import shutil
import socket
import struct
import subprocess
import threading
import time

import av
import numpy as np



def _find_tool(env: str, names: tuple[str, ...], globs: tuple[str, ...]) -> str:
    """Chemin d'un outil scrcpy/adb : variable d'env, puis PATH, puis emplacements connus (Mac/Linux/Windows)."""
    if os.environ.get(env):
        return os.environ[env]
    for n in names:
        w = shutil.which(n)
        if w:
            return w
    for g in globs:
        hits = sorted(glob.glob(os.path.expandvars(os.path.expanduser(g))))
        if hits:
            return hits[-1]
    return names[0]


_WINGET_SCRCPY = r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Genymobile.scrcpy_*\scrcpy-win64-v*"
ADB = _find_tool("ADB", ("adb",), (_WINGET_SCRCPY + r"\adb.exe", r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"))


def _scrcpy_server() -> str:
    scrcpy = shutil.which("scrcpy")
    near = (os.path.join(os.path.dirname(os.path.realpath(scrcpy)), "scrcpy-server"),) if scrcpy else ()
    return _find_tool("SCRCPY_SERVER_PATH", (), (*near, _WINGET_SCRCPY + r"\scrcpy-server",
                      "/opt/homebrew/share/scrcpy/scrcpy-server", "/usr/local/share/scrcpy/scrcpy-server",
                      "/usr/share/scrcpy/scrcpy-server"))


SERVER_LOCAL = _scrcpy_server()
SERVER_REMOTE = "/data/local/tmp/scrcpy-server.jar"
SERVER_VERSION = "4.1"

# Types d'action Android MotionEvent
ACTION_DOWN, ACTION_UP, ACTION_MOVE = 0, 1, 2
MSG_INJECT_TOUCH = 2


def _adb(serial: str, *args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([ADB, "-s", serial, *args], capture_output=True, text=True, **kw)


def first_usb_device() -> str:
    out = subprocess.run([ADB, "devices"], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "device" and ":" not in parts[0] and not parts[0].startswith("emulator"):
            return parts[0]
    raise RuntimeError("Aucun téléphone USB détecté (adb devices)")


class Device:
    """Connexion scrcpy : `frame()` renvoie la dernière image BGR, `tap(x, y)` touche l'écran.

    Les coordonnées sont celles de l'image vidéo (après max_size).
    """

    def __init__(self, serial: str | None = None, max_size: int = 1280, max_fps: int = 60,
                 bit_rate: int = 8_000_000):
        self.serial = serial or first_usb_device()
        self.max_size, self.max_fps, self.bit_rate = max_size, max_fps, bit_rate
        self.scid = random.randrange(1, 2**31)
        self.port = 27183 + random.randrange(0, 500)
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0          # time.perf_counter() à la fin du décodage
        self.frame_count = 0
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._running = False
        self._proc: subprocess.Popen | None = None
        self._video: socket.socket | None = None
        self._control: socket.socket | None = None
        self._ctrl_lock = threading.Lock()

    # ---------- démarrage ----------
    def start(self) -> "Device":
        _adb(self.serial, "push", SERVER_LOCAL, SERVER_REMOTE, check=True)
        name = f"localabstract:scrcpy_{self.scid:08x}"
        _adb(self.serial, "forward", f"tcp:{self.port}", name, check=True)
        args = [
            f"scid={self.scid:08x}", "log_level=warn", "tunnel_forward=true",
            "audio=false", "control=true", "video_codec=h264",
            # flux H.264 nu, sauf l'octet factice qui confirme que le serveur écoute
            "send_device_meta=false", "send_frame_meta=false", "send_codec_meta=false", "send_dummy_byte=true",
            f"max_size={self.max_size}", f"max_fps={self.max_fps}",
            f"video_bit_rate={self.bit_rate}", "stay_awake=true", "clipboard_autosync=false",
            "cleanup=true",
        ]
        cmd = [ADB, "-s", self.serial, "shell", f"CLASSPATH={SERVER_REMOTE}", "app_process", "/",
               "com.genymobile.scrcpy.Server", SERVER_VERSION, *args]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        self._video = self._connect(expect_dummy=True)
        self._control = self._connect()
        self._control.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._running = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        if not self.wait_frame(timeout=10):
            log = self._proc.stdout.read() if self._proc.poll() is not None else ""
            self.stop()
            raise RuntimeError(f"Pas d'image reçue du téléphone. Log serveur : {log}")
        return self

    def _connect(self, expect_dummy: bool = False) -> socket.socket:
        deadline = time.time() + 10
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError("Le serveur scrcpy s'est arrêté : " + self._proc.stdout.read())
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=2)
                # adb forward accepte la connexion même si le serveur n'écoute pas encore :
                # seul l'octet factice prouve qu'on parle bien au serveur.
                if expect_dummy and s.recv(1) != b"\x00":
                    s.close()
                    time.sleep(0.05)
                    continue
                return s
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("Impossible de se connecter au serveur scrcpy")

    # ---------- vidéo ----------
    def _video_loop(self):
        codec = av.CodecContext.create("h264", "r")
        codec.options = {"flags": "low_delay"}
        codec.thread_type = "NONE"  # pas de file de threads = pas d'images de retard
        sock = self._video
        sock.settimeout(None)
        while self._running:
            try:
                data = sock.recv(1 << 16)
            except OSError:
                break
            if not data:
                break
            for packet in codec.parse(data):
                try:
                    frames = codec.decode(packet)
                except av.error.InvalidDataError:
                    continue
                for f in frames:
                    img = f.to_ndarray(format="bgr24")
                    with self._new_frame:
                        self._frame = img
                        self._frame_time = time.perf_counter()
                        self.frame_count += 1
                        self._new_frame.notify_all()
        self._running = False

    def wait_frame(self, timeout: float = 1.0, after: int | None = None) -> bool:
        """Attend une image (plus récente que le compteur `after` si donné)."""
        with self._new_frame:
            target = (after if after is not None else self.frame_count if self._frame is not None else -1)
            return self._new_frame.wait_for(lambda: self._frame is not None and self.frame_count > target,
                                            timeout=timeout)

    def frame(self) -> tuple[np.ndarray, float, int]:
        """(image BGR, instant de réception, numéro) — sans copie."""
        with self._lock:
            return self._frame, self._frame_time, self.frame_count

    @property
    def size(self) -> tuple[int, int]:
        img = self._frame
        return (img.shape[1], img.shape[0]) if img is not None else (0, 0)

    # ---------- touches ----------
    def _touch(self, action: int, x: float, y: float, pointer: int = 0x1234):
        w, h = self.size
        x, y = int(round(min(max(x, 0), w - 1))), int(round(min(max(y, 0), h - 1)))
        pressure = 0xFFFF if action != ACTION_UP else 0
        msg = struct.pack(">BBqiiHHHII", MSG_INJECT_TOUCH, action, pointer, x, y, w, h, pressure, 1 if action != ACTION_MOVE else 0, 1 if action != ACTION_UP else 0)
        with self._ctrl_lock:
            self._control.sendall(msg)

    def tap(self, x: float, y: float, hold: float = 0.03):
        self._touch(ACTION_DOWN, x, y)
        time.sleep(hold)
        self._touch(ACTION_UP, x, y)

    def drag(self, x0: float, y0: float, x1: float, y1: float, duration: float = 0.12, steps: int = 6):
        self._touch(ACTION_DOWN, x0, y0)
        for i in range(1, steps + 1):
            t = i / steps
            time.sleep(duration / steps)
            self._touch(ACTION_MOVE, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        self._touch(ACTION_UP, x1, y1)

    # ---------- arrêt ----------
    def stop(self):
        self._running = False
        for s in (self._video, self._control):
            try:
                s and s.close()
            except OSError:
                pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        _adb(self.serial, "forward", "--remove", f"tcp:{self.port}")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
