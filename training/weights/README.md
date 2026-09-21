Put trusted pretrained `yolo26n.pt` here, or run from the repository root:

    python training/run.py download-weights

This command downloads only the official Ultralytics YOLO26n release asset,
records its SHA-256, and does not install PyTorch or run training.
Alternatively set paths.weights in config.local.yaml to an existing local file.
Never load an untrusted .pt checkpoint. No model weights are included in this repository.
