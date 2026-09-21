#!/usr/bin/env python3
"""Convert ORIGINAL VisA class-index masks into a supervised YOLO detection dataset.

Default category: chewinggum. One output class: 0 = defect.
Original images/masks are read-only; output must be a new directory.
Python 3.10+. See README_EN.md / README_BS.md for usage, scope and limitations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import platform
import random
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    import cv2
    import numpy as np
    import PIL
    from PIL import Image, ImageDraw
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Run: python -m pip install -r requirements.txt\n"
        f"Details: {exc}"
    ) from exc

VERSION = "1.0.0"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}
DATASET_URL = "https://github.com/amazon-science/spot-diff"
FORMAT_URL = "https://docs.ultralytics.com/datasets/detect/"
DATASET_LICENSE = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = """# Source and attribution

Source: Visual Anomaly (VisA) dataset, Amazon Science.
Authors: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer.
Paper: SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
and Segmentation (2022).
Repository: https://github.com/amazon-science/spot-diff
Dataset license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Changes: original nonzero mask labels were collapsed to `defect`; connected
regions were converted to axis-aligned bounding boxes (or one union box if
explicitly requested). A NEW supervised train/val/test split was created.
The images in images/ are byte-for-byte copies of the original files.
Images in previews/ additionally contain visualization overlays.
This is NOT the original VisA anomaly-detection benchmark protocol.
Retain source attribution, the license reference and this modification notice
when publishing examples or redistributing the derived data.
"""


class DatasetError(ValueError):
    """An input would produce unsafe, incomplete or misleading annotations."""


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    def yolo_line(self, width: int, height: int) -> str:
        """Pixel cells occupy [x, x+w) and [y, y+h), including 1-pixel regions."""
        if not (width > 0 and height > 0 and self.w > 0 and self.h > 0):
            raise DatasetError("Image and box dimensions must be positive.")
        if not (0 <= self.x < self.x + self.w <= width
                and 0 <= self.y < self.y + self.h <= height):
            raise DatasetError(f"Out-of-bounds box: {self}, image {width}x{height}")
        values = ((self.x + self.w / 2) / width,
                  (self.y + self.h / 2) / height,
                  self.w / width, self.h / height)
        return "0 " + " ".join(f"{v:.10f}" for v in values)


@dataclass
class Sample:
    image: Path
    mask: Path | None
    condition: str
    width: int = 0
    height: int = 0
    boxes: list[Box] = field(default_factory=list)
    mask_values: list[int] = field(default_factory=list)
    foreground_pixels: int = 0
    removed_components: int = 0
    removed_pixels: int = 0
    components: int = 0
    image_sha256: str = ""
    pixels_sha256: str = ""
    mask_sha256: str = ""
    binary_mask_sha256: str = ""
    split: str = ""
    output_name: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def child_directory(parent: Path, name: str) -> Path | None:
    if not parent.is_dir():
        return None
    matches = [p for p in parent.iterdir()
               if p.is_dir() and p.name.casefold() == name.casefold()]
    if len(matches) > 1:
        raise DatasetError(f"Ambiguous directory name {name!r} inside {parent}")
    return matches[0] if matches else None


def walk_directory(root: Path, *parts: str) -> Path | None:
    current = root
    for part in parts:
        found = child_directory(current, part)
        if found is None:
            return None
        current = found
    return current


def locate_category(source: Path, category: str) -> Path:
    """Accept a category folder, the VisA root, or its direct parent."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise DatasetError(f"Source directory does not exist: {source}")
    candidates: list[Path] = []
    if source.name.casefold() == category.casefold():
        candidates.append(source)
    direct = child_directory(source, category)
    if direct:
        candidates.append(direct)
    wrapper = child_directory(source, "VisA")
    if wrapper:
        nested = child_directory(wrapper, category)
        if nested:
            candidates.append(nested)
    found = [p.resolve() for p in candidates
             if walk_directory(p, "Data", "Images", "Normal")
             and walk_directory(p, "Data", "Images", "Anomaly")
             and walk_directory(p, "Data", "Masks", "Anomaly")]
    found = list(dict.fromkeys(found))
    if len(found) > 1:
        raise DatasetError("Multiple category folders found. Pass the exact folder with --source.")
    if not found:
        raise DatasetError(
            f"Cannot find the ORIGINAL VisA layout for {category!r} under {source}.\n"
            f"Expected: {category}/Data/Images/Normal, Data/Images/Anomaly, "
            "Data/Masks/Anomaly.\nUse the extracted original archive, not VisA_pytorch/1cls. "
            "--source may also point directly to the category folder."
        )
    return found[0]


def indexed_files(folder: Path, extensions: set[str]) -> dict[str, Path]:
    """Match stems case-insensitively; refuse ambiguous stems and nested input."""
    found: dict[str, Path] = {}
    for path in sorted(folder.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
        if path.name.startswith(".") or path.name.lower() in {"thumbs.db", "desktop.ini"}:
            continue
        if path.is_dir():
            raise DatasetError(f"Unexpected subdirectory in a flat VisA data folder: {path}")
        if path.suffix.lower() not in extensions:
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                raise DatasetError(f"Unsupported/lossy mask format: {path}. Use original PNG masks.")
            continue
        if not path.is_file() or not inside(path.resolve(), folder.resolve()):
            raise DatasetError(f"Input file is not a regular file within its data folder: {path}")
        stem = path.stem.casefold()
        if stem in found:
            raise DatasetError(f"Ambiguous image/mask stem: {found[stem]} and {path}")
        found[stem] = path
    return found


def discover_samples(category_root: Path, category: str) -> list[Sample]:
    directories = [walk_directory(category_root, "Data", "Images", "Normal"),
                   walk_directory(category_root, "Data", "Images", "Anomaly"),
                   walk_directory(category_root, "Data", "Masks", "Anomaly")]
    if any(p is None for p in directories):
        raise DatasetError("Required source directories are missing.")
    normal_dir, anomaly_dir, mask_dir = directories
    assert normal_dir and anomaly_dir and mask_dir
    for folder in directories:
        assert folder is not None
        if not inside(folder.resolve(), category_root.resolve()):
            raise DatasetError(f"Source directory resolves outside the category: {folder}")
    normal = indexed_files(normal_dir, IMAGE_EXTENSIONS)
    anomalous = indexed_files(anomaly_dir, IMAGE_EXTENSIONS)
    masks = indexed_files(mask_dir, MASK_EXTENSIONS)
    if not normal or not anomalous:
        raise DatasetError("Both Normal and Anomaly folders must contain images.")
    missing = sorted(set(anomalous) - set(masks))
    extra = sorted(set(masks) - set(anomalous))
    if missing or extra:
        raise DatasetError(
            f"Images and masks do not match by filename stem.\n"
            f"Missing masks ({len(missing)}): {', '.join(missing[:10]) or '-'}\n"
            f"Masks without an image ({len(extra)}): {', '.join(extra[:10]) or '-'}"
        )
    samples: list[Sample] = []
    names: set[str] = set()
    for condition, files in (("normal", normal), ("anomaly", anomalous)):
        for stem, image in sorted(files.items()):
            safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", image.stem)
            name = f"{category}_{condition}_{safe_stem}{image.suffix.lower()}"
            # Labels have .txt extensions, so check the stem as well as the image name.
            output_stem = Path(name).stem.casefold()
            if output_stem in names:
                raise DatasetError(f"Output filename collision after normalization: {name}")
            names.add(output_stem)
            samples.append(Sample(image, masks[stem] if condition == "anomaly" else None,
                                  condition, output_name=name))
    return samples


def read_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise DatasetError(f"Multi-frame images are unsupported: {path}")
            orientation = image.getexif().get(274, 1)
            if orientation not in (None, 1):
                raise DatasetError(
                    f"EXIF orientation {orientation} in {path}. Export image and mask into "
                    "the same explicit pixel orientation first; refusing to guess alignment."
                )
            image.load()  # Detect truncated/corrupt files before producing output.
            return np.array(image.convert("RGB"), dtype=np.uint8)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Cannot decode image {path}: {exc}") from exc


def read_mask(path: Path) -> tuple[np.ndarray, list[int]]:
    """Preserve P-mode palette INDICES. Never convert index masks to grayscale."""
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise DatasetError(f"Multi-frame masks are unsupported: {path}")
            if image.getexif().get(274, 1) not in (None, 1):
                raise DatasetError(f"EXIF-rotated mask is unsupported: {path}")
            image.load()
            array = np.array(image)
            if array.ndim == 3:
                # Grayscale stored as RGB/RGBA/LA is fine; arbitrary color maps are not.
                channels = array.shape[2]
                if channels in (2, 4):
                    alpha = array[:, :, -1]
                    if not np.all(alpha == 255):
                        raise DatasetError(f"Non-opaque alpha in mask: {path}")
                if channels == 2:
                    array = array[:, :, 0]
                elif channels in (3, 4) and np.array_equal(array[:, :, 0], array[:, :, 1]) \
                        and np.array_equal(array[:, :, 0], array[:, :, 2]):
                    array = array[:, :, 0]
                else:
                    raise DatasetError(
                        f"Colorized RGB mask is not a class-index mask: {path}. "
                        "Use the original VisA PNG mask, not a visualization."
                    )
            if array.ndim != 2 or array.dtype.kind not in "bui":
                raise DatasetError(f"Mask must be a single-channel integer index image: {path}")
            values = [int(v) for v in np.unique(array)]
            if values and values[0] < 0:
                raise DatasetError(f"Negative mask labels are unsupported: {path}")
            return np.ascontiguousarray(array > 0, dtype=np.uint8), values
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Cannot decode mask {path}: {exc}") from exc


def boxes_from_binary(binary: np.ndarray, *, mode: str = "components",
                      min_area: int = 1, padding: int = 0,
                      connectivity: int = 8) -> tuple[list[Box], dict[str, int]]:
    if binary.ndim != 2 or not binary.size:
        raise DatasetError("Mask must be a nonempty 2D array.")
    if mode not in {"components", "union"} or min_area < 1 or padding < 0 \
            or connectivity not in {4, 8}:
        raise DatasetError("Invalid bounding-box options.")
    binary = np.ascontiguousarray(binary > 0, dtype=np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity, ltype=cv2.CV_32S
    )
    accepted: list[Box] = []
    removed_components = removed_pixels = 0
    for row in stats[1:]:  # Index 0 is the background, never a defect.
        x, y, w, h, area = (int(v) for v in row)
        if area < min_area:
            removed_components += 1
            removed_pixels += area
        else:
            accepted.append(Box(x, y, w, h))
    if mode == "union" and accepted:
        x = min(b.x for b in accepted)
        y = min(b.y for b in accepted)
        right = max(b.x + b.w for b in accepted)
        bottom = max(b.y + b.h for b in accepted)
        accepted = [Box(x, y, right - x, bottom - y)]
    height, width = binary.shape
    padded: set[Box] = set()
    for box in accepted:
        x, y = max(0, box.x - padding), max(0, box.y - padding)
        right = min(width, box.x + box.w + padding)
        bottom = min(height, box.y + box.h + padding)
        padded.add(Box(x, y, right - x, bottom - y))
    # Different components can yield identical enclosing boxes; do not duplicate YOLO rows.
    result = sorted(padded, key=lambda b: (b.y, b.x, b.h, b.w))
    return result, {"components": count - 1,
                    "removed_components": removed_components,
                    "removed_pixels": removed_pixels,
                    "duplicate_boxes_removed": len(accepted) - len(result)}


def analyse_sample(sample: Sample, args: argparse.Namespace) -> None:
    pixels = read_image(sample.image)
    sample.height, sample.width = pixels.shape[:2]
    sample.image_sha256 = sha256_file(sample.image)
    digest = hashlib.sha256(f"{sample.width}x{sample.height}:RGB:".encode())
    digest.update(pixels.tobytes())
    sample.pixels_sha256 = digest.hexdigest()
    if sample.mask is None:
        return
    binary, sample.mask_values = read_mask(sample.mask)
    if binary.shape != (sample.height, sample.width):
        raise DatasetError(
            f"Image/mask dimension mismatch for {sample.image.name}: image "
            f"{sample.width}x{sample.height}, mask {binary.shape[1]}x{binary.shape[0]}. "
            "No automatic resizing is performed."
        )
    sample.foreground_pixels = int(binary.sum())
    if sample.foreground_pixels == 0:
        raise DatasetError(f"Anomaly has an EMPTY mask: {sample.mask}. It cannot become a normal sample.")
    sample.mask_sha256 = sha256_file(sample.mask)
    sample.binary_mask_sha256 = hashlib.sha256(binary.tobytes()).hexdigest()
    sample.boxes, info = boxes_from_binary(
        binary, mode=args.box_mode, min_area=args.min_area,
        padding=args.padding, connectivity=args.connectivity
    )
    sample.components = info["components"]
    sample.removed_components = info["removed_components"]
    sample.removed_pixels = info["removed_pixels"]
    if not sample.boxes:
        raise DatasetError(
            f"All defects were removed by --min-area for {sample.mask}. "
            "Use --min-area 1; refusing to export an anomaly with an empty label."
        )
    for box in sample.boxes:
        box.yolo_line(sample.width, sample.height)  # Validate bounds before output.


def split_counts(total: int, ratios: Sequence[float]) -> list[int]:
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios) \
            or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9, rel_tol=0):
        raise DatasetError("--split must contain three positive finite fractions summing to 1.")
    if total < 3:
        raise DatasetError("Each condition needs at least 3 unique image groups for train/val/test.")
    ideal = [total * r for r in ratios]
    counts = [math.floor(v) for v in ideal]
    order = sorted(range(3), key=lambda i: (-(ideal[i] - counts[i]), i))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    for i in range(3):
        if counts[i] == 0:
            donor = max(range(3), key=lambda j: counts[j])
            counts[donor] -= 1
            counts[i] += 1
    return counts


def assign_splits(samples: list[Sample], ratios: Sequence[float], seed: int) -> list[dict]:
    """Split normal/anomaly separately; identical decoded pixels stay in ONE split."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in sorted(samples, key=lambda s: s.output_name):
        groups[sample.pixels_sha256].append(sample)
    duplicates: list[dict] = []
    for pixel_hash, group in groups.items():
        annotations = {(s.condition, s.binary_mask_sha256) for s in group}
        if len(annotations) != 1:
            raise DatasetError(
                "Identical image pixels have conflicting labels/masks: "
                + ", ".join(str(s.image) for s in group)
            )
        if len(group) > 1:
            duplicates.append({"pixels_sha256": pixel_hash,
                               "images": [str(s.image) for s in group]})
    for condition in ("normal", "anomaly"):
        bucket = [g for g in groups.values() if g[0].condition == condition]
        bucket.sort(key=lambda g: g[0].output_name)
        random.Random(f"visa-yolo-v1:{seed}:{condition}").shuffle(bucket)
        counts = split_counts(len(bucket), ratios)
        offset = 0
        for split, count in zip(SPLITS, counts):
            for group in bucket[offset:offset + count]:
                for sample in group:
                    sample.split = split
            offset += count
    for item in duplicates:
        item["split"] = groups[item["pixels_sha256"]][0].split
    return duplicates


def create_preview(sample: Sample, destination: Path) -> None:
    pixels = read_image(sample.image)
    if sample.mask:
        mask, _ = read_mask(sample.mask)
        visible = mask.astype(bool)
        # Original mask in translucent red; exported boxes in green.
        pixels[visible] = (pixels[visible].astype(np.float32) * 0.60
                           + np.array([255, 45, 60]) * 0.40).astype(np.uint8)
    canvas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(canvas)
    stroke = max(2, sample.width // 500)
    for index, box in enumerate(sample.boxes, 1):
        draw.rectangle((box.x, box.y, box.x + box.w - 1, box.y + box.h - 1),
                       outline=(30, 255, 90), width=stroke)
        text = f"defect {index}"
        tx, ty = box.x, max(0, box.y - 14)
        rectangle = draw.textbbox((tx, ty), text)
        draw.rectangle(rectangle, fill=(0, 0, 0))
        draw.text((tx, ty), text, fill=(30, 255, 90))
    # A separate header does not obscure the source image.
    header = Image.new("RGB", (sample.width, sample.height + 34), (20, 24, 32))
    header.paste(canvas, (0, 34))
    ImageDraw.Draw(header).text((8, 10),
        f"{sample.output_name} | {sample.split} | {sample.condition} | boxes: {len(sample.boxes)}",
        fill=(255, 255, 255))
    header.save(destination, quality=95)


def write_previews(samples: list[Sample], destination: Path, count: int, seed: int) -> int:
    if count == 0:
        return 0
    # Review annotations using train/val previews; keep the test set aside.
    buckets: list[list[Sample]] = []
    rng = random.Random(f"visa-yolo-previews:{seed}")
    for split in ("train", "val"):
        for condition in ("anomaly", "normal"):
            group = sorted([s for s in samples if s.split == split and s.condition == condition],
                           key=lambda s: s.output_name)
            rng.shuffle(group)
            buckets.append(group)
    selected: list[Sample] = []
    while len(selected) < count and any(buckets):
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
    destination.mkdir()
    cards: list[str] = []
    for sample in selected:
        filename = f"{sample.split}_{Path(sample.output_name).stem}.jpg"
        create_preview(sample, destination / filename)
        name = html.escape(filename, quote=True)
        caption = html.escape(f"{sample.split} / {sample.condition} / {len(sample.boxes)} boxes")
        cards.append(f'<article><a href="{name}"><img loading="lazy" src="{name}" '
                     f'alt="{caption}"></a><p>{caption}</p></article>')
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisA mask-to-box review</title><style>
body{font:16px system-ui,sans-serif;margin:32px;background:#161a21;color:#eee}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}
article{margin:0;padding:12px;background:#232b36;border-radius:8px}img{width:100%;height:auto}
p{line-height:1.5}a{color:inherit}</style>
<h1>Mask-to-box review</h1><p>Red: original mask. Green: exported bounding boxes.
Click an image for the full-resolution preview. These are annotations, NOT model predictions.
Only train/val examples are shown. Do not use previews as training images.</p><main>"""
    (destination / "index.html").write_text(document + "\n".join(cards) + "</main></html>\n",
                                            encoding="utf-8")
    return len(selected)


def sample_record(sample: Sample, category_root: Path) -> dict:
    return {
        "source_image": sample.image.relative_to(category_root).as_posix(),
        "source_mask": sample.mask.relative_to(category_root).as_posix() if sample.mask else "",
        "image": f"images/{sample.split}/{sample.output_name}",
        "label": f"labels/{sample.split}/{Path(sample.output_name).stem}.txt",
        "condition": sample.condition, "split": sample.split,
        "width": sample.width, "height": sample.height, "box_count": len(sample.boxes),
        "mask_values": sample.mask_values, "foreground_pixels": sample.foreground_pixels,
        "components": sample.components, "removed_components": sample.removed_components,
        "removed_pixels": sample.removed_pixels, "image_sha256": sample.image_sha256,
        "pixels_sha256": sample.pixels_sha256, "mask_sha256": sample.mask_sha256,
        "binary_mask_sha256": sample.binary_mask_sha256,
    }


def summarize(samples: list[Sample]) -> dict:
    result: dict = {}
    for split in SPLITS:
        members = [s for s in samples if s.split == split]
        result[split] = {"images": len(members),
                         "normal": sum(s.condition == "normal" for s in members),
                         "anomaly": sum(s.condition == "anomaly" for s in members),
                         "boxes": sum(len(s.boxes) for s in members)}
    return result


def write_dataset(samples: list[Sample], root: Path, output: Path,
                  args: argparse.Namespace, duplicates: list[dict], notices: list[str]) -> dict:
    """Write to a temporary sibling, verify copies/labels, then publish the new directory."""
    if output.exists() or output.is_symlink():
        raise DatasetError(f"Output already exists: {output}. Choose a NEW --output directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        for split in SPLITS:
            (temp / "images" / split).mkdir(parents=True)
            (temp / "labels" / split).mkdir(parents=True)
        records = []
        for sample in sorted(samples, key=lambda s: (s.split, s.output_name)):
            record = sample_record(sample, root)
            image_target = temp / record["image"]
            label_target = temp / record["label"]
            shutil.copy2(sample.image, image_target)
            if sha256_file(image_target) != sample.image_sha256:
                raise DatasetError(f"Source image changed during conversion: {sample.image}")
            if sample.mask and sha256_file(sample.mask) != sample.mask_sha256:
                raise DatasetError(f"Source mask changed during conversion: {sample.mask}")
            lines = [box.yolo_line(sample.width, sample.height) for box in sample.boxes]
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            records.append(record)
        # Absolute path avoids dependence on the caller's Ultralytics datasets_dir setting.
        yaml_text = ("# New supervised split; not the original VisA benchmark.\n"
                     "# After moving this directory, update path below to its new absolute path.\n"
                     f"path: {json.dumps(output.as_posix(), ensure_ascii=True)}\n"
                     "train: images/train\nval: images/val\ntest: images/test\n\n"
                     "names:\n  0: defect\n")
        (temp / "data.yaml").write_text(yaml_text, encoding="utf-8")
        (temp / "classes.txt").write_text("defect\n", encoding="utf-8")
        (temp / "SOURCE_ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
        with (temp / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            for record in records:
                writer.writerow({**record, "mask_values": json.dumps(record["mask_values"])})
        with (temp / "annotations.jsonl").open("w", encoding="utf-8") as handle:
            by_image = {s.output_name: s for s in samples}
            for record in records:
                sample = by_image[Path(record["image"]).name]
                extended = {**record, "boxes_xywh_pixels": [
                    {"class_id": 0, "x": b.x, "y": b.y, "width": b.w, "height": b.h}
                    for b in sample.boxes]}
                handle.write(json.dumps(extended, ensure_ascii=True) + "\n")
        preview_count = write_previews(samples, temp / "previews", args.previews, args.seed)
        report = {
            "converter_version": VERSION,
            "converter_sha256": sha256_file(Path(__file__)),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_category_root": str(root), "output_root": str(output),
            "category": args.category, "class_names": {"0": "defect"},
            "dataset_source": DATASET_URL, "dataset_license": DATASET_LICENSE,
            "yolo_format": FORMAT_URL,
            "parameters": {"split": args.split, "seed": args.seed, "box_mode": args.box_mode,
                           "connectivity": args.connectivity, "min_area": args.min_area,
                           "padding": args.padding, "previews_requested": args.previews},
            "split_method": "normal/anomaly stratification over identical-RGB-pixel groups",
            "official_benchmark_split": False,
            "totals": {"images": len(samples), "boxes": sum(len(s.boxes) for s in samples),
                       "removed_components": sum(s.removed_components for s in samples),
                       "removed_pixels": sum(s.removed_pixels for s in samples),
                       "previews": preview_count},
            "splits": summarize(samples), "duplicate_image_groups": duplicates,
            "warnings": notices,
            "limitations": ["Only exact decoded-pixel duplicates are grouped; near-duplicates, "
                            "capture sessions and product identities are not inferred.",
                            "Connected regions are not semantic defect instances.",
                            "Mask foreground represents defects, not whole products.",
                            "Copied images are not resized or re-encoded."],
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "opencv": cv2.__version__, "pillow": PIL.__version__},
        }
        (temp / "conversion_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        validate_export(temp, samples)
        if output.exists():
            raise DatasetError(f"Output appeared during conversion: {output}; refusing to replace it.")
        temp.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)  # Only our newly-created temporary directory.
        raise


def validate_export(directory: Path, samples: list[Sample]) -> None:
    expected_images = {f"images/{s.split}/{s.output_name}" for s in samples}
    actual_images = {p.relative_to(directory).as_posix()
                     for p in (directory / "images").rglob("*") if p.is_file()}
    if expected_images != actual_images:
        raise DatasetError("Export validation failed: image inventory mismatch.")
    expected_labels = {f"labels/{s.split}/{Path(s.output_name).stem}.txt" for s in samples}
    actual_labels = {p.relative_to(directory).as_posix()
                     for p in (directory / "labels").rglob("*") if p.is_file()}
    if expected_labels != actual_labels:
        raise DatasetError("Export validation failed: label inventory mismatch.")
    memberships: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        label = directory / "labels" / sample.split / f"{Path(sample.output_name).stem}.txt"
        expected = [box.yolo_line(sample.width, sample.height) for box in sample.boxes]
        lines = label.read_text(encoding="utf-8").splitlines()
        if lines != expected or (sample.condition == "anomaly" and not lines):
            raise DatasetError(f"Export validation failed: invalid label {label}")
        memberships[sample.pixels_sha256].add(sample.split)
    if any(len(splits) > 1 for splits in memberships.values()):
        raise DatasetError("Export validation failed: identical image pixels span multiple splits.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Original VisA masks -> YOLO bounding boxes, class 0=defect (no training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=Path("."),
                        help="Extracted VisA root, its parent, or the category directory")
    parser.add_argument("--category", default="chewinggum", help="One original VisA category")
    parser.add_argument("--output", type=Path, help="NEW output directory (default: ./yolo_CATEGORY)")
    parser.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"), help="Positive fractions summing to 1")
    parser.add_argument("--seed", type=int, default=42, help="Repeatable split and preview selection")
    parser.add_argument("--box-mode", choices=["components", "union"], default="components",
                        help="One box per connected region, or one enclosing box per anomaly image")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8,
                        help="Pixel connectivity; 8 includes diagonal adjacency")
    parser.add_argument("--min-area", type=int, default=1,
                        help="Minimum component area in original-mask pixels (1 keeps all)")
    parser.add_argument("--padding", type=int, default=0,
                        help="Additional pixels on each box side, clipped to image bounds")
    parser.add_argument("--previews", type=int, default=24,
                        help="Maximum annotated train/val previews; 0 disables")
    parser.add_argument("--check-only", action="store_true", help="Validate/plan only; write no dataset")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser


def run(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.category):
        raise DatasetError("Category must contain only letters, digits, '_' and '-'.")
    args.category = args.category.lower()
    split_counts(100, args.split)  # Validate fractions up front.
    if args.min_area < 1 or args.padding < 0 or args.previews < 0:
        raise DatasetError("--min-area must be >= 1; --padding and --previews must be >= 0.")
    root = locate_category(args.source, args.category)
    output_arg = (args.output or Path(f"yolo_{args.category}")).expanduser()
    if output_arg.is_symlink():
        raise DatasetError("Output must not be a symbolic link.")
    output = output_arg.resolve()
    data_root = walk_directory(root, "Data")
    assert data_root
    if inside(root, output) or inside(output, data_root.resolve()):
        raise DatasetError("Output must not replace source/ancestors or be inside source Data/.")
    if not args.check_only and output.exists():
        raise DatasetError(f"Output already exists: {output}. Choose a NEW --output directory.")
    samples = discover_samples(root, args.category)
    print(f"Source: {root}\nFound {len(samples)} images. Validating images and masks...", flush=True)
    for index, sample in enumerate(samples, 1):
        analyse_sample(sample, args)
        if index % 50 == 0 or index == len(samples):
            print(f"  Validated {index}/{len(samples)}", flush=True)
    duplicates = assign_splits(samples, args.split, args.seed)
    notices: list[str] = []
    if duplicates:
        notices.append(f"{len(duplicates)} exact-pixel duplicate groups kept within single splits. "
                       "Ratios are approximate when a group contains multiple images.")
    removed = sum(s.removed_components for s in samples)
    if removed:
        notices.append(f"--min-area discarded {removed} components. Review their annotations carefully.")
    fragmented = sum(len(s.boxes) > 20 for s in samples)
    if fragmented:
        notices.append(f"{fragmented} images have more than 20 boxes; check previews for fragmented masks.")
    high_coverage = sum(s.foreground_pixels / (s.width * s.height) > 0.8 for s in samples)
    if high_coverage:
        notices.append(f"{high_coverage} masks cover over 80% of the image; confirm foreground meaning.")
    counts = Counter(s.condition for s in samples)
    if args.category == "chewinggum" and (counts["normal"], counts["anomaly"]) != (503, 100):
        notices.append(f"Found {counts['normal']} normal / {counts['anomaly']} anomaly images; "
                       "the published complete chewinggum subset has 503 / 100. "
                       "A deliberately selected subset is acceptable.")
    print("\nSplit       Images    Normal   Anomaly     Boxes")
    for split, info in summarize(samples).items():
        print(f"{split:<10}{info['images']:>8}{info['normal']:>10}"
              f"{info['anomaly']:>10}{info['boxes']:>10}")
    for notice in notices:
        print(f"WARNING: {notice}", file=sys.stderr)
    if args.check_only:
        print("\nValidation complete. No dataset was written (--check-only).")
        return
    print(f"\nWriting to: {output}", flush=True)
    report = write_dataset(samples, root, output, args, duplicates, notices)
    print(f"\nDONE: {report['totals']['images']} images, {report['totals']['boxes']} boxes.")
    print(f"YOLO config: {output / 'data.yaml'}")
    print(f"Audit report: {output / 'conversion_report.json'}")
    if report["totals"]["previews"]:
        print(f"Visual review: {output / 'previews' / 'index.html'}")
    print("Original data unchanged. No training, model downloads or inference were performed.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (DatasetError, OSError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled. Original data unchanged.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
