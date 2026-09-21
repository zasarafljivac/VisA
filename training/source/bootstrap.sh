#!/usr/bin/env bash
# This runs INSIDE the isolated SageMaker container, never on the user's host.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export YOLO_CONFIG_DIR=/tmp/visa-yolo-settings
export YOLO_AUTOINSTALL=false
export PIP_DISABLE_PIP_VERSION_CHECK=1
SOURCE=/opt/ml/input/data/source
mkdir -p /opt/ml/model /opt/ml/output/data
trap 'code=$?; printf "Training bootstrap failed (exit %s). See CloudWatch logs.\n" "$code" > /opt/ml/output/failure; exit "$code"' ERR
python - <<'PY'
import torch
assert torch.__version__.split('+')[0] == '2.2.0', f'Unexpected PyTorch: {torch.__version__}'
assert torch.cuda.is_available(), 'CUDA is unavailable; refusing CPU fallback'
print('Base container:', torch.__version__, torch.version.cuda, flush=True)
PY
# opencv-python is an explicit Ultralytics dependency. Install its OS libraries
# only if absent, rather than installing conflicting GUI/headless wheels.
if ! python - <<'PY'
import ctypes
ctypes.CDLL('libGL.so.1')
ctypes.CDLL('libglib-2.0.so.0')
PY
then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgl1 libglib2.0-0
  rm -rf /var/lib/apt/lists/*
fi
python -m pip install --no-cache-dir -c "$SOURCE/constraints.txt" -r "$SOURCE/requirements.txt"
if python - "$SOURCE/run-config.json" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))['training']['export_onnx'] else 1)
PY
then
  python -m pip install --no-cache-dir -c "$SOURCE/constraints.txt" -r "$SOURCE/requirements-export.txt"
fi
cd /opt/ml/output/data
# exec forwards SageMaker termination signals directly to the Python process.
exec python "$SOURCE/train.py" --config "$SOURCE/run-config.json"
