# clash-ai

An AI that learns to play Clash Royale through friendly battles, running on a real Android phone over USB.

**Status: phase 1 (eyes + hands) working.** The agent currently plays random cards; detection and learning come next.

## How it works

- `clashai/device.py` launches the scrcpy server on the phone directly, decodes the raw H.264 stream in Python (PyAV, low-delay) and injects touches through scrcpy's control socket.
- `clashai/battle.py` reads the battle screen: in-battle detection, elixir, playable cards.
- `clashai/actions.py` plays cards and *verifies* each tap by watching the card lift.
- `clashai/viewer.py` shows the live stream with the AI's overlay.

Measured on a REDMAGIC 11 Air: ~58 fps stream, **83 ms median tap-to-screen latency** (70–90 ms).

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv torch ultralytics opencv-python av numpy
brew install scrcpy android-platform-tools
.venv/bin/python -m clashai.viewer              # live window (q to quit)
.venv/bin/python scripts/test_phase1.py         # during a Training Camp match
```

## Roadmap

1. ✅ Stream + verified taps + elixir/hand reading
2. Unit/tower detection (YOLO) + tracking (ByteTrack), drawn live in the viewer
3. Imitation learning from recorded human games
4. Reinforcement learning after each friendly battle

Automating Clash Royale is against Supercell's Terms of Service; use a secondary account, friendly battles and Training Camp only.
