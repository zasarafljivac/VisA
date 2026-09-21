"""CPU-only validation and paths shared by the launcher and training entry point."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
ULTRALYTICS_VERSION = '8.4.60'
MODEL_NAME = 'yolo26n.pt'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a YAML mapping: {path}')
    return data


def ensure_regular(path: Path, root: Path) -> None:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Not a regular file inside the selected dataset: {path}')


def validate_dataset(root: Path, include_test: bool = True) -> dict:
    """Accept only our flat YOLO layout; never execute YAML download hooks.

    Validate empty negative labels, nonempty positives, bounding boxes and exact
    decoded-image duplicates across splits. The original data.yaml 'path' is
    deliberately ignored, because it may refer to another machine.
    """
    root = root.resolve()
    meta = read_yaml(root / 'data.yaml')
    names = meta.get('names')
    if names not in ({0: 'defect'}, {'0': 'defect'}, ['defect']):
        raise ValueError('This package requires exactly one class, 0: defect.')
    if 'download' in meta:
        raise ValueError('A dataset YAML download hook is not allowed.')
    splits = ['train', 'val'] + (['test'] if include_test else [])
    result: dict[str, Any] = {'root': str(root), 'names': {'0': 'defect'}, 'splits': {}, 'files': []}
    hashes: dict[str, str] = {}
    pixels_seen: dict[str, str] = {}
    for split in splits:
        if meta.get(split) != f'images/{split}':
            raise ValueError(f'data.yaml must use {split}: images/{split}')
        image_dir, label_dir = root / 'images' / split, root / 'labels' / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f'Missing image or label directory for {split}.')
        if image_dir.is_symlink() or label_dir.is_symlink():
            raise ValueError('Dataset directories must not be symlinks.')
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f'No images in {image_dir}')
        stats = {'images': 0, 'positive': 0, 'negative': 0, 'boxes': 0}
        stems: set[str] = set()
        for image in images:
            ensure_regular(image, root)
            if image.stem.lower() in stems:
                raise ValueError(f'Duplicate image stem: {image}')
            stems.add(image.stem.lower())
            label = label_dir / (image.stem + '.txt')
            ensure_regular(label, root)  # negatives are explicit empty files in our converter
            lines = [s.strip() for s in label.read_text(encoding='utf-8').splitlines() if s.strip()]
            for number, line in enumerate(lines, 1):
                cols = line.split()
                if len(cols) != 5 or cols[0] != '0':
                    raise ValueError(f'Invalid YOLO detection row: {label}:{number}')
                x, y, w, h = map(float, cols[1:])
                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError(f'Non-finite bounding box: {label}:{number}')
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f'Invalid normalized box: {label}:{number}')
                if min(x - w / 2, y - h / 2) < -1e-8 or max(x + w / 2, y + h / 2) > 1 + 1e-8:
                    raise ValueError(f'Box extends beyond the image: {label}:{number}')
            digest = sha256(image)
            with Image.open(image) as im:
                im.load()
                if im.getexif().get(274, 1) != 1:
                    raise ValueError(f'EXIF-rotated image needs explicit handling: {image}')
                pix = im.convert('RGB')
                pixel_hash = hashlib.sha256(str(pix.size).encode() + pix.tobytes()).hexdigest()
            for seen, value in ((hashes, digest), (pixels_seen, pixel_hash)):
                prior = seen.get(value)
                if prior is not None and prior != split:
                    raise ValueError(f'Duplicate image leaks across {prior}/{split}: {image}')
                seen[value] = split
            stats['images'] += 1
            stats['positive' if lines else 'negative'] += 1
            stats['boxes'] += len(lines)
            for item, item_hash in ((image, digest), (label, sha256(label))):
                result['files'].append({'path': item.relative_to(root).as_posix(), 'sha256': item_hash, 'bytes': item.stat().st_size})
        orphans = [p.name for p in label_dir.glob('*.txt') if p.stem.lower() not in stems]
        if orphans:
            raise ValueError(f'Orphan labels in {split}: {orphans[:5]}')
        if not stats['positive'] or not stats['negative']:
            raise ValueError(f'{split} must contain both positive and negative images.')
        result['splits'][split] = stats
    result['dataset_fingerprint'] = hashlib.sha256(json.dumps(result['files'], sort_keys=True).encode()).hexdigest()
    return result


def portable_yaml(destination: Path, root: str, include_test: bool) -> None:
    data = {'path': root, 'train': 'images/train', 'val': 'images/val', 'names': {0: 'defect'}}
    if include_test:
        data['test'] = 'images/test'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
