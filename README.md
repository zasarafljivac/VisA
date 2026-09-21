# VisA → YOLO converter

A local, CPU-only Python script that converts the segmentation masks of the
[Visual Anomaly (VisA)](https://github.com/amazon-science/spot-diff) dataset into an
**Ultralytics YOLO detection** dataset with a single class, `0: defect`, and a new
reproducible train/val/test split.

- No GPU, PyTorch, Ultralytics or network access needed — only NumPy, OpenCV and Pillow.
- The original dataset is never modified; the output is always a new folder.
- One VisA category per run (default: `chewinggum`).
- The VisA images and masks are **not** included; download them from the source repository.

## Quick start

```bash
git clone https://github.com/zasarafljivac/VisA.git
cd VisA
# extract the VisA category you need (e.g. chewinggum/) here

python3 -m venv .venv-visa
source .venv-visa/bin/activate
python -m pip install -r visa_yolo_tools/requirements.txt

python visa_yolo_tools/convert_visa_to_yolo.py --source . --output ./yolo_chewinggum
```

Then open `yolo_chewinggum/previews/index.html` to check the boxes against the masks.

## Optional: training on SageMaker

[training/](training/) contains a small controller that validates the converted dataset and
submits a single YOLO26n training job to Amazon SageMaker using your own AWS credentials,
bucket and execution role. Nothing is uploaded and no job is created without an explicit
`--execute`. See the training README in [English](training/README_EN.md) or [Bosnian](training/README_BS.md).

## Documentation

- [README in English](visa_yolo_tools/README_EN.md)
- [README na bosanskom](visa_yolo_tools/README_BS.md)
- [Test report in English](visa_yolo_tools/TEST_REPORT_EN.md)
- [Izvještaj o provjeri na bosanskom](visa_yolo_tools/TEST_REPORT_BS.md)
- Training README in [English](training/README_EN.md) / [na bosanskom](training/README_BS.md), and the [training test report](training/TEST_REPORT.md)
- [Dataset source and attribution](visa_yolo_tools/SOURCE_ATTRIBUTION.md)

## License

The code in this repository is released under the [MIT License](LICENSE).

The VisA dataset itself is published by Amazon Science under CC BY 4.0 and is not covered by
this license. This is an independently written conversion script; it is not an official Amazon
or Ultralytics tool.
