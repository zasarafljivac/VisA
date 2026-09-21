"""Synthetic regression tests. No VisA data, network, GPU or model is used."""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visa_yolo_tools.convert_visa_to_yolo as converter


def make_dataset(parent: Path, normal: int = 20, anomaly: int = 10) -> Path:
    root = parent / "VisA" / "chewinggum"
    for sub in ("Images/Normal", "Images/Anomaly", "Masks/Anomaly"):
        (root / "Data" / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(761)
    for condition, count in (("Normal", normal), ("Anomaly", anomaly)):
        for index in range(count):
            pixels = rng.integers(0, 256, size=(80, 100, 3), dtype=np.uint8)
            # Same stems across Normal/Anomaly exercise output collision prevention.
            image = root / "Data" / "Images" / condition / f"{index:03d}.JPG"
            Image.fromarray(pixels).save(image)
            if condition == "Anomaly":
                mask = np.zeros((80, 100), dtype=np.uint8)
                mask[10:20, 20:40] = index % 6 + 1
                mask[40:45, 60:70] = (index + 2) % 6 + 1
                Image.fromarray(mask).save(root / "Data/Masks/Anomaly" / f"{index:03d}.png")
    return root


class MaskAndBoxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_low_mask_values_are_foreground(self):
        raw = np.arange(7, dtype=np.uint8).reshape(1, 7)
        path = self.directory / "mask.png"
        Image.fromarray(raw).save(path)
        binary, values = converter.read_mask(path)
        self.assertEqual(values, list(range(7)))
        self.assertEqual(binary.tolist(), [[0, 1, 1, 1, 1, 1, 1]])

    def test_palette_indices_not_palette_brightness(self):
        image = Image.new("P", (12, 10))
        # White background (index 0), BLACK defect (index 1).
        # Converting this to L would invert the annotation.
        palette = [0] * 768
        palette[:3] = [255, 255, 255]
        image.putpalette(palette)
        image.putpixel((6, 7), 1)
        path = self.directory / "palette.png"
        image.save(path)
        binary, values = converter.read_mask(path)
        self.assertEqual(values, [0, 1])
        self.assertEqual(int(binary.sum()), 1)
        self.assertEqual(int(binary[7, 6]), 1)

    def test_binary_255_mask(self):
        raw = np.zeros((12, 10), dtype=np.uint8)
        raw[2:4, 5:8] = 255
        path = self.directory / "binary.png"
        Image.fromarray(raw).save(path)
        binary, values = converter.read_mask(path)
        self.assertEqual(values, [0, 255])
        self.assertEqual(int(binary.sum()), 6)

    def test_grayscale_rgb_mask_accepted(self):
        raw = np.zeros((10, 10, 3), dtype=np.uint8)
        raw[2:4, 3:7, :] = 3
        path = self.directory / "rgb.png"
        Image.fromarray(raw).save(path)
        binary, _ = converter.read_mask(path)
        self.assertEqual(int(binary.sum()), 8)

    def test_color_visualization_mask_rejected(self):
        raw = np.zeros((10, 10, 3), dtype=np.uint8)
        raw[2:4, 3:7, 0] = 255
        path = self.directory / "color.png"
        Image.fromarray(raw).save(path)
        with self.assertRaisesRegex(converter.DatasetError, "Colorized"):
            converter.read_mask(path)

    def test_nonopaque_alpha_rejected(self):
        raw = np.zeros((10, 10, 4), dtype=np.uint8)
        path = self.directory / "alpha.png"
        Image.fromarray(raw).save(path)
        with self.assertRaisesRegex(converter.DatasetError, "alpha"):
            converter.read_mask(path)

    def test_16bit_index_mask(self):
        raw = np.zeros((10, 10), dtype=np.uint16)
        raw[2:4, 3:7] = 6
        path = self.directory / "16bit.png"
        Image.fromarray(raw).save(path)
        binary, values = converter.read_mask(path)
        self.assertEqual(values, [0, 6])
        self.assertEqual(int(binary.sum()), 8)

    def test_components_and_normalization(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[10:30, 20:60] = 1
        mask[80:90, 150:160] = 6
        boxes, info = converter.boxes_from_binary(mask)
        self.assertEqual(boxes, [converter.Box(20, 10, 40, 20), converter.Box(150, 80, 10, 10)])
        self.assertEqual(info["components"], 2)
        self.assertEqual(boxes[0].yolo_line(200, 100), "0 0.2000000000 0.2000000000 0.2000000000 0.2000000000")

    def test_single_pixel_bottom_right_not_zero_area(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[99, 199] = 1
        boxes, _ = converter.boxes_from_binary(mask)
        self.assertEqual(boxes, [converter.Box(199, 99, 1, 1)])
        self.assertEqual(boxes[0].yolo_line(200, 100), "0 0.9975000000 0.9950000000 0.0050000000 0.0100000000")

    def test_full_frame_box(self):
        mask = np.ones((100, 200), dtype=np.uint8)
        boxes, _ = converter.boxes_from_binary(mask)
        self.assertEqual(boxes[0].yolo_line(200, 100), "0 0.5000000000 0.5000000000 1.0000000000 1.0000000000")

    def test_union_mode(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[10:30, 20:60] = 1
        mask[80:90, 150:160] = 1
        boxes, _ = converter.boxes_from_binary(mask, mode="union")
        self.assertEqual(boxes, [converter.Box(20, 10, 140, 80)])

    def test_padding_clipped_to_edges(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        boxes, _ = converter.boxes_from_binary(mask, padding=100)
        self.assertEqual(boxes, [converter.Box(0, 0, 10, 10)])

    def test_padding_deduplicates_identical_boxes(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = mask[9, 9] = 1
        boxes, info = converter.boxes_from_binary(mask, padding=100)
        self.assertEqual(boxes, [converter.Box(0, 0, 10, 10)])
        self.assertEqual(info["duplicate_boxes_removed"], 1)

    def test_min_area_uses_foreground_count(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[1, 1] = 1
        mask[10:13, 10:13] = 1
        boxes, info = converter.boxes_from_binary(mask, min_area=2)
        self.assertEqual(boxes, [converter.Box(10, 10, 3, 3)])
        self.assertEqual(info["removed_components"], 1)
        self.assertEqual(info["removed_pixels"], 1)

    def test_connectivity_and_touching_different_classes(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1, 1], mask[2, 2] = 1, 6
        self.assertEqual(len(converter.boxes_from_binary(mask, connectivity=8)[0]), 1)
        self.assertEqual(len(converter.boxes_from_binary(mask, connectivity=4)[0]), 2)

    def test_invalid_box_rejected(self):
        with self.assertRaises(converter.DatasetError):
            converter.Box(99, 0, 2, 3).yolo_line(100, 100)

    def test_empty_binary_returns_no_boxes(self):
        boxes, info = converter.boxes_from_binary(np.zeros((10, 10), dtype=np.uint8))
        self.assertEqual(boxes, [])
        self.assertEqual(info["components"], 0)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="visa converter test ")
        self.parent = Path(self.temp.name)
        self.root = make_dataset(self.parent)
        self.output = self.parent / "YOLO output"

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, *extra: str, output: Path | None = None) -> str:
        args = converter.build_parser().parse_args([
            "--source", str(self.parent / "VisA"),
            "--output", str(output or self.output), "--previews", "0", *extra
        ])
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            converter.run(args)
        return captured.getvalue()

    def mask_path(self, stem: str = "000") -> Path:
        return self.root / "Data/Masks/Anomaly" / f"{stem}.png"

    def test_source_root_parent_category_and_case_insensitive_folders(self):
        for path in (self.parent, self.parent / "VisA", self.root):
            self.assertEqual(converter.locate_category(path, "chewinggum"), self.root)
        source = self.root / "Data/Images/Anomaly"
        source.rename(source.with_name("ANOMALY"))
        self.assertEqual(len(converter.discover_samples(self.root, "chewinggum")), 30)

    def test_full_conversion_and_byte_identical_copies(self):
        original_hashes = {p: converter.sha256_file(p) for p in self.root.rglob("*") if p.is_file()}
        self.execute("--previews", "8")
        report = json.loads((self.output / "conversion_report.json").read_text())
        self.assertEqual(report["totals"]["images"], 30)
        self.assertEqual(report["totals"]["boxes"], 20)
        self.assertEqual(report["totals"]["previews"], 8)
        self.assertEqual(report["splits"]["train"], {"images": 21, "normal": 14, "anomaly": 7, "boxes": 14})
        with (self.output / "manifest.csv").open() as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 30)
        for record in records:
            source = self.root / record["source_image"]
            target = self.output / record["image"]
            self.assertEqual(source.read_bytes(), target.read_bytes())
            lines = (self.output / record["label"]).read_text().splitlines()
            self.assertEqual(len(lines), 0 if record["condition"] == "normal" else 2)
            for line in lines:
                parts = line.split()
                self.assertEqual(len(parts), 5)
                self.assertEqual(parts[0], "0")
                self.assertTrue(all(0 < float(v) <= 1 for v in parts[1:]))
        self.assertEqual({p: converter.sha256_file(p) for p in original_hashes}, original_hashes)
        previews = list((self.output / "previews").glob("*.jpg"))
        self.assertEqual(len(previews), 8)
        self.assertTrue(all(not p.name.startswith("test_") for p in previews))
        self.assertTrue((self.output / "previews/index.html").exists())
        config = (self.output / "data.yaml").read_text()
        path_line = next(s for s in config.splitlines() if s.startswith("path: "))
        self.assertEqual(json.loads(path_line[len("path: "):]), self.output.as_posix())
        self.assertIn("  0: defect", config)
        for line in (self.output / "annotations.jsonl").read_text().splitlines():
            record = json.loads(line)
            self.assertEqual(record["box_count"], len(record["boxes_xywh_pixels"]))

    def test_repeatability_of_splits_and_labels(self):
        other = self.parent / "repeat"
        self.execute()
        self.execute(output=other)
        self.assertEqual((self.output / "manifest.csv").read_bytes(), (other / "manifest.csv").read_bytes())
        for label in (self.output / "labels").rglob("*.txt"):
            self.assertEqual(label.read_bytes(), (other / label.relative_to(self.output)).read_bytes())

    def test_check_only_does_not_create_output(self):
        log = self.execute("--check-only")
        self.assertFalse(self.output.exists())
        self.assertIn("No dataset was written", log)

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(converter.DatasetError, "already exists"):
            self.execute()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_missing_mask_is_fatal(self):
        self.mask_path().unlink()
        with self.assertRaisesRegex(converter.DatasetError, "Missing masks"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_orphan_mask_is_fatal(self):
        shutil.copy2(self.mask_path(), self.mask_path("orphan"))
        with self.assertRaisesRegex(converter.DatasetError, "Masks without an image"):
            self.execute()

    def test_ambiguous_mask_stem_is_fatal(self):
        Image.open(self.mask_path()).save(self.root / "Data/Masks/Anomaly/000.bmp")
        with self.assertRaisesRegex(converter.DatasetError, "Ambiguous"):
            self.execute()

    def test_dimension_mismatch_is_fatal(self):
        Image.fromarray(np.ones((40, 50), dtype=np.uint8)).save(self.mask_path())
        with self.assertRaisesRegex(converter.DatasetError, "dimension mismatch"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_empty_anomaly_mask_is_fatal(self):
        Image.fromarray(np.zeros((80, 100), dtype=np.uint8)).save(self.mask_path())
        with self.assertRaisesRegex(converter.DatasetError, "EMPTY mask"):
            self.execute()

    def test_filter_cannot_silently_make_positive_negative(self):
        with self.assertRaisesRegex(converter.DatasetError, "All defects were removed"):
            self.execute("--min-area", "100000")

    def test_corrupt_image_is_fatal(self):
        (self.root / "Data/Images/Normal/000.JPG").write_bytes(b"not a jpeg")
        with self.assertRaisesRegex(converter.DatasetError, "Cannot decode image"):
            self.execute()

    def test_rotated_exif_image_is_rejected(self):
        path = self.root / "Data/Images/Normal/000.JPG"
        with Image.open(path) as source:
            pixels = source.copy()
        exif = Image.Exif()
        exif[274] = 6
        pixels.save(path, exif=exif)
        with self.assertRaisesRegex(converter.DatasetError, "EXIF orientation"):
            self.execute()

    def test_exact_duplicates_stay_together(self):
        source = self.root / "Data/Images/Normal/000.JPG"
        shutil.copy2(source, source.with_name("duplicate.JPG"))
        self.execute()
        report = json.loads((self.output / "conversion_report.json").read_text())
        self.assertEqual(len(report["duplicate_image_groups"]), 1)
        with (self.output / "manifest.csv").open() as handle:
            records = list(csv.DictReader(handle))
        corresponding = [r for r in records if r["source_image"].endswith(("Normal/000.JPG", "Normal/duplicate.JPG"))]
        self.assertEqual(len(corresponding), 2)
        self.assertEqual(corresponding[0]["split"], corresponding[1]["split"])

    def test_conflicting_labels_on_identical_pixels_rejected(self):
        shutil.copy2(self.root / "Data/Images/Normal/000.JPG", self.root / "Data/Images/Anomaly/000.JPG")
        with self.assertRaisesRegex(converter.DatasetError, "conflicting"):
            self.execute()

    def test_conflicting_masks_on_identical_pixels_rejected(self):
        shutil.copy2(self.root / "Data/Images/Anomaly/000.JPG", self.root / "Data/Images/Anomaly/001.JPG")
        raw = np.zeros((80, 100), dtype=np.uint8)
        raw[50:60, 20:30] = 1
        Image.fromarray(raw).save(self.mask_path("001"))
        with self.assertRaisesRegex(converter.DatasetError, "conflicting"):
            self.execute()

    def test_input_data_output_path_rejected(self):
        with self.assertRaisesRegex(converter.DatasetError, "inside source Data"):
            self.execute(output=self.root / "Data/output")
        self.assertFalse((self.root / "Data/output").exists())

    def test_bad_split_fractions_rejected(self):
        for fractions in ((0.8, 0.2, 0.2), (float("nan"), 0.2, 0.8), (0, 0.5, 0.5), (-1, 1, 1)):
            with self.subTest(fractions=fractions):
                with self.assertRaises(converter.DatasetError):
                    converter.split_counts(100, fractions)

    def test_split_counts_and_minimum_sizes(self):
        self.assertEqual(converter.split_counts(503, [.7, .15, .15]), [352, 76, 75])
        self.assertEqual(converter.split_counts(100, [.7, .15, .15]), [70, 15, 15])
        self.assertEqual(converter.split_counts(3, [.7, .15, .15]), [1, 1, 1])
        with self.assertRaises(converter.DatasetError):
            converter.split_counts(2, [.7, .15, .15])

    def test_failed_write_is_cleaned_and_not_published(self):
        with mock.patch.object(converter, "write_previews", side_effect=OSError("Simulated disk failure")):
            with self.assertRaisesRegex(OSError, "Simulated"):
                self.execute()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.parent.glob(".YOLO output.partial-*")), [])
        self.assertTrue(self.root.exists())

    def test_bad_category_path_is_rejected(self):
        with self.assertRaisesRegex(converter.DatasetError, "Category"):
            self.execute("--category", "../chewinggum")


if __name__ == "__main__":
    unittest.main(verbosity=2)
