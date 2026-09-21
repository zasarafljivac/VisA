# Test report — VisA YOLO tools 1.0.0

Date: 21 September 2026.

## Result

**38/38 automated regression tests: PASS.**

Command, from the `visa_yolo_tools` folder:

```bash
python -m unittest discover -s tests -v
```

Execution was verified on Linux with Python 3.13.5, NumPy 2.3.5, OpenCV 4.13.0 and Pillow 12.3.0. Support for Python 3.10+ is based on the language/API subset used and the declared dependencies; a test matrix across all Python versions was not run separately. The PowerShell instructions were not executed on Windows.

## What the regression suite covers

- Masks with small indices 1–6, binary 0/255 masks, 16-bit indices and palette PNGs where the defect colour is black and the background is white. Palette indices are not converted to brightness.
- Separate components, a box covering the full image, a single pixel at the bottom-right border, diagonal 4/8 connectivity, merging into a union box, padding and its clamping to the image dimensions.
- YOLO coordinates and normalization, rejection of invalid boxes, filtering by the number of masked pixels and removal of identical output boxes.
- Acceptable grayscale-as-RGB masks and rejection of arbitrary colorized RGB/transparent masks.
- Full conversion of a synthetic structure in the original VisA layout, generation of the YAML/CSV/JSONL files, preview images and the HTML overview.
- Good images with empty annotations, source names that are identical in Normal and Anomaly, copies byte-for-byte identical to the originals, and an untouched source.
- Reproducibility of the split and annotations, planning without writing (`--check-only`), exact counts for 503/100 samples and the minimum set size.
- Missing, empty, ambiguous and orphan masks; differing dimensions; corrupted images and EXIF rotation.
- Identification of exactly identical RGB pixels, keeping duplicates in a single split and rejecting conflicting annotations of identical images.
- Protection of an existing output and of the source Data folder; removal of only the new temporary output after a simulated write error.

## Additional checks performed

**CLI smoke test: PASS.** The script was run as a standalone program over **603 synthetic images**, arranged as 503 normal and 100 defective. The output had 422 train, 91 val and 90 test images, with correct image/label pairs. Every synthetic defective image had two deliberately drawn regions, so this test produced 200 boxes. **This is not the number of boxes in the real VisA dataset.**

**Independent geometry check: PASS.** An additional one-off audit of 300 random masks was run for both connectivities, 600 cases in total. The results were compared with an independent Python flood-fill algorithm, and the round-trip conversion of normalized YOLO coordinates back to pixels was verified. This additional audit is not part of the 38-test regression command above.

**Syntax/bytecode check: PASS.** `python -m compileall` was run over the code.

## Limits of the verification

**The full original VisA archive was not downloaded or processed as part of this verification.** The original folder structure, the name-pairing rule and the meaning of non-zero mask values were checked against the official Amazon repository; all executed tests used synthetic data.

**Ultralytics training and inference were not installed or run.** The output files were checked against the documented YOLO detection format, without performing actual training.

The verification does not measure the quality of a trained model, the semantically optimal grouping of cracks into boxes, near-duplicate leakage or the usefulness of an experiment in production. Before training, review the generated `previews/index.html` and `conversion_report.json` on your actual dataset.
