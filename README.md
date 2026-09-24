# clash-ai

An AI that learns to play Clash Royale through friendly battles, running on a real Android phone over USB.

**Status: phase 2a — live unit detection and tracking working** (with KataCR's pre-trained detectors). The agent still plays random cards.

## How it works

- `clashai/device.py` launches the scrcpy server on the phone directly, decodes the raw H.264 stream in Python (PyAV, low-delay) and injects touches through scrcpy's control socket.
- `clashai/battle.py` reads the battle screen: in-battle detection, elixir, playable cards.
- `clashai/actions.py` plays cards and *verifies* each tap by watching the card lift.
- `clashai/detect.py` detects and tracks units (YOLOv8 + ByteTrack) using the pre-trained detectors from [KataCR](https://github.com/wty-yy/KataCR) (MIT).
- `clashai/viewer.py` shows the live stream with the AI's overlay: boxes, track IDs, trails, side (blue = ours, red = enemy), elixir, playable cards.

Measured on a REDMAGIC 11 Air (M3 MacBook Air host): ~58 fps stream, **83 ms median tap-to-screen latency** (70–90 ms). Detection with the two KataCR detectors runs at ~110 ms/frame alone on MPS (fp16), 160–340 ms while the agent is also playing — too slow, hence phase 2b.

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv torch ultralytics opencv-python av numpy
brew install scrcpy android-platform-tools
.venv/bin/python -m clashai.viewer              # live window (q to quit)

# Detection (needs KataCR's old ultralytics 8.1 in a separate venv)
git clone --depth 1 https://github.com/wty-yy/KataCR third_party/KataCR
git clone --depth 1 https://github.com/wty-yy/Clash-Royale-Detection-Dataset data/Clash-Royale-Dataset
# set path_dataset in third_party/KataCR/katacr/build_dataset/constant.py to data/Clash-Royale-Dataset
# download detector1/2_v0.7.13.pt (links in KataCR's README) into models/katacr/
uv venv --python 3.11 .venv-katacr && uv pip install --python .venv-katacr ultralytics==8.1.24 torch==2.2.2 torchvision==0.17.2 "numpy<2" opencv-python==4.9.0.80 Pillow==10.3.0 lap jax==0.4.26 jaxlib==0.4.26 flax==0.8.1 optax==0.2.2 orbax-checkpoint==0.5.3 matplotlib==3.8.2 scipy tqdm einops moviepy==1.0.3 tensorboardX pandas "av<13"
.venv-katacr/bin/python -m clashai.viewer --detect
.venv/bin/python scripts/test_phase1.py         # during a Training Camp match
```

## Roadmap

1. ✅ Stream + verified taps + elixir/hand reading
2. ✅ (2a) Unit/tower detection + tracking drawn live, using KataCR's detectors
   ⏳ (2b) Train our own small, fast detector (target < 20 ms/frame) on KataCR's generative sprite dataset, with cards released since 2024
3. Imitation learning from recorded human games
4. Reinforcement learning after each friendly battle

Automating Clash Royale is against Supercell's Terms of Service; use a secondary account, friendly battles and Training Camp only.
