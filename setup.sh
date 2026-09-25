#!/usr/bin/env bash
# Récupère tout ce qui n'est pas dans le dépôt (trop lourd ou pas à nous) et prépare les environnements.
# macOS / Linux / Windows (Git Bash). Sur PC NVIDIA, installer ensuite torch CUDA (voir README).
set -e
cd "$(dirname "$0")"
[ -d third_party/KataCR ] || git clone --depth 1 https://github.com/wty-yy/KataCR third_party/KataCR
(cd third_party/KataCR && git apply --check ../../katacr_patches.diff 2>/dev/null && git apply ../../katacr_patches.diff) || true
[ -d data/Clash-Royale-Dataset ] || git clone --depth 1 https://github.com/wty-yy/Clash-Royale-Detection-Dataset data/Clash-Royale-Dataset
python -m pip install -q gdown uv 2>/dev/null || true
mkdir -p models/katacr
[ -f models/katacr/detector1_v0.7.13.pt ] || gdown -O models/katacr/detector1_v0.7.13.pt 1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_
[ -f models/katacr/detector2_v0.7.13.pt ] || gdown -O models/katacr/detector2_v0.7.13.pt 1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD
mkdir -p third_party/KataCR/runs && cp models/katacr/*.pt third_party/KataCR/runs/
uv venv --python 3.11 .venv-katacr
uv pip install --python .venv-katacr ultralytics==8.1.24 torchvision "numpy<2" opencv-python==4.9.0.80 Pillow==10.3.0 lap \
  jax==0.4.26 jaxlib==0.4.26 flax==0.8.1 optax==0.2.2 orbax-checkpoint==0.5.3 matplotlib==3.8.2 scipy tqdm einops \
  moviepy==1.0.3 tensorboardX pandas "av<13"
mkdir -p runs/games && cp -n learning/journal.jsonl runs/games/ 2>/dev/null || true
echo "OK. Sur RTX 50xx : uv pip install --python .venv-katacr torch torchvision --index-url https://download.pytorch.org/whl/cu128"
