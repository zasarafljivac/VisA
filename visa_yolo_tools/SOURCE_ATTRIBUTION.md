# Source and attribution

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
