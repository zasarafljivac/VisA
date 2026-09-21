#!/usr/bin/env python3
"""YOLO26n training entry point shared by SageMaker and optional local execution.

No weights/model substitution. Imports the GPU stack only when actually run.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from common import MODEL_NAME, ULTRALYTICS_VERSION, portable_yaml, sha256, validate_dataset, write_json


def training_kwargs(cfg: dict, yaml_path: Path, run_dir: Path) -> dict:
    tr = cfg['training']
    keys = ('epochs', 'batch', 'workers', 'imgsz', 'seed', 'patience', 'optimizer', 'amp', 'deterministic')
    return {
        **{k: tr[k] for k in keys},
        'data': str(yaml_path), 'device': cfg.get('device', '0'),
        'project': str(run_dir), 'name': 'train', 'exist_ok': False,
        'task': 'detect', 'pretrained': True, 'save': True, 'val': True,
        'plots': True, 'cache': False, 'verbose': True,
    }


def metric_dict(metrics) -> dict:
    values = getattr(metrics, 'results_dict', {}) or {}
    return {str(k): float(v) for k, v in values.items()}


def check_model(model) -> None:
    if model.task != 'detect':
        raise ValueError('The checkpoint is not a detection model.')
    model_yaml = model.model.yaml
    stem = Path(str(model_yaml.get('yaml_file', ''))).stem
    scale = model_yaml.get('scale')
    if stem not in {'yolo26', 'yolo26n'} or (scale not in (None, 'n')):
        raise ValueError(f'Expected YOLO26 nano, got yaml_file={stem!r}, scale={scale!r}. No fallback allowed.')
    if stem == 'yolo26' and scale != 'n':
        raise ValueError('Cannot verify nano scale in checkpoint metadata. No model substitution allowed.')


def execute_training(cfg: dict, yolo_class, environment: dict) -> dict:
    """Dependency-injected only for offline unit testing; production uses YOLO."""
    started = time.time()
    tr = cfg['training']
    if tr['model'] != MODEL_NAME or str(tr['ultralytics_version']) != ULTRALYTICS_VERSION:
        raise ValueError('Expected yolo26n.pt and Ultralytics 8.4.60.')
    weights = Path(cfg['weights']).resolve()
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError(f'Missing local checkpoint: {weights}. Automatic fallback/download is disabled.')
    if cfg.get('weights_sha256') and sha256(weights) != cfg['weights_sha256']:
        raise ValueError('Checkpoint SHA-256 changed after job preparation.')
    dataset = Path(cfg['dataset']).resolve()
    run_dir, model_dir = Path(cfg['run_dir']).resolve(), Path(cfg['model_dir']).resolve()
    if (run_dir / 'train').exists() or (model_dir / 'best.pt').exists():
        raise ValueError('Output already contains a training run; refusing to overwrite.')
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    report = validate_dataset(dataset, include_test=tr['evaluate_test'])
    expected = cfg.get('dataset_fingerprint')
    if expected and expected != report['dataset_fingerprint']:
        raise ValueError('Uploaded dataset does not match the validated local snapshot.')
    write_json(model_dir / 'dataset_validation.json', report)
    write_json(model_dir / 'training_config.json', cfg)
    write_json(model_dir / 'environment.json', environment)
    data_yaml = run_dir / 'data.yaml'
    portable_yaml(data_yaml, str(dataset), tr['evaluate_test'])
    # A portable artifact YAML is metadata; images are not bundled in model.tar.gz.
    portable_yaml(model_dir / 'data.yaml', '.', tr['evaluate_test'])
    (model_dir / 'classes.txt').write_text('defect\n', encoding='utf-8')
    for name in ('SOURCE_ATTRIBUTION.md', 'conversion_report.json', 'manifest.csv'):
        origin = dataset / name
        if origin.is_file():
            shutil.copy2(origin, model_dir / name)
    model = yolo_class(str(weights), task='detect')
    check_model(model)

    def emit_epoch(trainer):
        for key, value in (getattr(trainer, 'metrics', {}) or {}).items():
            try:
                print(f'VISA_METRIC {key}={float(value):.8f}', flush=True)
            except (ValueError, TypeError):
                pass

    model.add_callback('on_fit_epoch_end', emit_epoch)
    kwargs = training_kwargs(cfg, data_yaml, run_dir)
    write_json(model_dir / 'resolved_training_args.json', kwargs)
    train_result = model.train(**kwargs)
    actual_run = Path(model.trainer.save_dir)
    for name in ('best.pt', 'last.pt'):
        origin = actual_run / 'weights' / name
        if not origin.is_file():
            raise RuntimeError(f'Training did not produce {name}: {actual_run}')
        shutil.copy2(origin, model_dir / name)
    for origin in actual_run.iterdir():
        if origin.is_file() and origin.suffix.lower() in {'.csv', '.yaml', '.png', '.jpg', '.json'}:
            shutil.copy2(origin, model_dir / origin.name)
    metrics = {
        'protocol': 'Custom supervised VisA split, NOT the original anomaly benchmark',
        'validation': metric_dict(train_result),
        'test_evaluated': False,
        'initial_weights_sha256': sha256(weights),
        'best_weights_sha256': sha256(model_dir / 'best.pt'),
        'dataset_fingerprint': report['dataset_fingerprint'],
        'train_seconds': time.time() - started,
    }
    write_json(model_dir / 'metrics.json', metrics)
    if tr['evaluate_test']:
        best = yolo_class(str(model_dir / 'best.pt'), task='detect')
        test_result = best.val(data=str(data_yaml), split='test', imgsz=tr['imgsz'],
                               batch=tr['batch'], workers=tr['workers'], device=cfg.get('device', '0'),
                               project=str(run_dir), name='test', exist_ok=False, plots=True)
        metrics['test'] = metric_dict(test_result)
        metrics['test_evaluated'] = True
        # The test is only used after best.pt has been selected using validation.
        write_json(model_dir / 'metrics.json', metrics)
    if tr['export_onnx']:
        best = yolo_class(str(model_dir / 'best.pt'), task='detect')
        # Do not guess an output-head/shape contract. Record actual graph I/O.
        exported = Path(best.export(format='onnx', imgsz=tr['imgsz'], batch=1,
                                    device='cpu', half=False, dynamic=False,
                                    simplify=False, opset=17))
        target = model_dir / 'best.onnx'
        if exported.resolve() != target.resolve():
            shutil.copy2(exported, target)
        import onnx
        graph = onnx.load(str(target))
        onnx.checker.check_model(graph)
        def spec(value):
            return {'name': value.name, 'shape': [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]}
        write_json(model_dir / 'onnx_metadata.json', {
            'inputs': [spec(x) for x in graph.graph.input],
            'outputs': [spec(x) for x in graph.graph.output],
            'sha256': sha256(target), 'ultralytics': ULTRALYTICS_VERSION,
            'note': 'Generic ONNX export; not a verified deployment contract for any specific edge runtime.'})
    metrics['total_seconds'] = time.time() - started
    write_json(model_dir / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/visa-yolo-settings')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f'Expected ultralytics {ULTRALYTICS_VERSION}, got {ultralytics.__version__}')
    if torch.__version__.split('+')[0] != '2.2.0':
        raise RuntimeError(f'Expected PyTorch 2.2.0 for this pinned baseline, got {torch.__version__}')
    if cfg.get('device', '0') != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available. Refusing silent CPU fallback.')
    for key in ('sync', 'wandb', 'mlflow', 'clearml', 'neptune', 'comet', 'dvc', 'dvclive', 'tensorboard'):
        if key in settings:
            settings.update({key: False})
    environment = {
        'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
        'ultralytics': ultralytics.__version__, 'cuda_build': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'packages': sorted({d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}.items()),
    }
    execute_training(cfg, YOLO, environment)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        # SageMaker includes this text as the training job failure reason.
        if Path('/opt/ml/output').is_dir():
            Path('/opt/ml/output/failure').write_text(str(exc)[:1000], encoding='utf-8')
        raise SystemExit(1)
