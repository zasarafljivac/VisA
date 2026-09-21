# VisA chewinggum → YOLO bounding box annotations

A local Python script that converts the existing VisA masks into an **Ultralytics YOLO detection** dataset. The default category is `chewinggum`, and the output has a single class: **`0: defect`**.

The script does not train a model, does not run inference, does not download images, and does not require a GPU, PyTorch, CUDA, an AWS account or an Ultralytics installation. Once the dependencies are installed it runs locally, with no network requests. The original dataset is never modified.

**This repository contains no VisA images or masks.** You need the extracted original VisA archive, downloaded separately [1]. `data.yaml` is generated after conversion, once the actual output path is known.

## 1. Folder layout

Clone this repository, or copy just the `visa_yolo_tools` folder, so that `visa_yolo_tools` sits next to the `chewinggum` folder. If you cloned the repository, you can extract the VisA categories straight into its root: they are excluded by `.gitignore`.

```text
VisA/
├── chewinggum/
│   └── Data/
│       ├── Images/
│       │   ├── Normal/
│       │   │   ├── 000.JPG
│       │   │   └── ...
│       │   └── Anomaly/
│       │       ├── 000.JPG
│       │       └── ...
│       └── Masks/
│           └── Anomaly/
│               ├── 000.png
│               └── ...
├── visa_yolo_tools/
│   ├── convert_visa_to_yolo.py
│   ├── requirements.txt
│   ├── README_BS.md
│   ├── README_EN.md
│   ├── TEST_REPORT_BS.md
│   ├── TEST_REPORT_EN.md
│   ├── SHA256SUMS.txt
│   ├── SOURCE_ATTRIBUTION.md
│   └── tests/
│       └── test_converter.py
└── ... other categories can stay here
```

The file names are illustrative. What matters is that an image and its mask inside `Anomaly` match by name without the extension: e.g. `043.JPG` and `043.png`. Names in `Normal` and `Anomaly` may be identical: the output adds different prefixes so nothing gets overwritten.

The script reads the **original** `Data/Images/...` and `Data/Masks/...` layout confirmed in Amazon's repository [1, 3]. It does not use `image_anno.csv` or the original `split_csv` splits: a new split is made from all images of the selected category. It does not support the reorganized `VisA_pytorch/1cls` layout, a ZIP/TAR archive that has not been extracted, or arbitrary RGB visualizations of masks.

## 2. Running — Linux / WSL

You need Python **3.10 or newer**, `venv` and `pip`. Open a terminal in your `VisA` folder:

```bash
cd "/path/to/VisA"

python3 -m venv .venv-visa
source .venv-visa/bin/activate
python -m pip install -r visa_yolo_tools/requirements.txt

python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --output ./yolo_chewinggum
```

This gives you a new local copy of the dataset in `VisA/yolo_chewinggum/`. Extra disk space is needed for the copies of the selected category's images and for the preview images; the other VisA categories are not copied.

The next time you use the same virtual environment, just activate `.venv-visa`; you do not need to recreate it or reinstall the packages.

### Windows PowerShell

From the `VisA` folder:

```powershell
py -3 -m venv .venv-visa
.\.venv-visa\Scripts\python.exe -m pip install -r .\visa_yolo_tools\requirements.txt
.\.venv-visa\Scripts\python.exe .\visa_yolo_tools\convert_visa_to_yolo.py --source . --output .\yolo_chewinggum
```

Activating the environment in PowerShell is not needed, so there is no need to change the execution policy. `py -3` must point to Python 3.10 or newer. The PowerShell commands are provided as instructions; the script was tested on Linux.

### A different source location

`--source` can point to `VisA`, to its direct parent folder, or directly to `chewinggum`:

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source "/mnt/datasets/VisA/chewinggum" \
  --output "/mnt/datasets/yolo_chewinggum"
```

The output is relative to the current working directory unless given as an absolute path. The script processes only one category per run.

## 3. Defaults

| Setting | Value |
|---|---|
| Category | `chewinggum` |
| Classes | One: `0 = defect` |
| Mask foreground | All pixels with a value greater than zero |
| Boxes | One per connected foreground region |
| Connectivity | 8 neighbours, including diagonals |
| Minimum area | 1 pixel — nothing is discarded by size |
| Extra padding | 0 pixels |
| Train / validation / test | Approximately 70% / 15% / 15%, separately for normal and defective images |
| Seed | `42` |
| Visual preview | Up to 24 examples from train/val, with masks and boxes |
| Training images | Original images copied byte-for-byte |
| Normal images | Empty `.txt` files, no `good` class |

For the complete published `chewinggum` set of 503 good and 100 defective images [1], **if there are no groups of identical images**, the default split gives:

| Set | Good | Defective | Total |
|---|---:|---:|---:|
| train | 352 | 70 | 422 |
| val | 76 | 15 | 91 |
| test | 75 | 15 | 90 |
| Total | 503 | 100 | 603 |

The number of bounding boxes is not equal to the number of defective images: a single image can have several boxes. You get the exact number after processing your masks.

The script can also work on a deliberately selected smaller set. At least three distinct image groups are required for each of the two conditions, normal/anomaly, so that every output set contains both conditions. For very small sets the ratios are adjusted to that minimum.

## 4. Output files

```text
yolo_chewinggum/
├── data.yaml
├── classes.txt
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── previews/
│   ├── index.html
│   └── ... .jpg preview images
├── manifest.csv
├── annotations.jsonl
├── conversion_report.json
└── SOURCE_ATTRIBUTION.md
```

`data.yaml` defines the class and the folders in the Ultralytics format [4]. Its `path` field contains the **absolute path of the output folder**. After moving the generated dataset to another computer/server, change `path` to the new absolute path; train/val/test remain relative to that path. Write a YAML path on a Windows drive with `/`, e.g. `D:/datasets/yolo_chewinggum`.

`manifest.csv` links every source image and mask to the output files, the condition and split, the dimensions, the number of boxes, the original mask values and the SHA-256 hashes. The `source_image` and `source_mask` paths are relative to the source category folder (`chewinggum`), not to the top of the `VisA` archive.

`annotations.jsonl` additionally stores the pixel coordinates of the boxes for verification. `conversion_report.json` stores the parameters, seed, statistics, warnings, library versions, script hash and groups of identical images.

### Visual check

Open **`yolo_chewinggum/previews/index.html`** in a browser, without starting a server. Clicking an image opens its full-resolution preview.

The red semi-transparent layer is the existing mask. The green rectangles are the computed bounding boxes. These are **annotations, not model predictions**. Good samples have neither a mask nor boxes.

Preview images are never placed into `images/` and must not be used as training input. Test images are deliberately not shown in the automatic preview: use train/val to review the conversion rules, and keep test for the final evaluation.

## 5. Additional options

### Checking the source without creating a dataset

```bash
python visa_yolo_tools/convert_visa_to_yolo.py --source . --check-only
```

This loads and validates the images/masks, computes the boxes and plans the split, but does not write an output dataset.

### A different split or more preview images

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --output ./yolo_chewinggum_80_10_10 \
  --split 0.80 0.10 0.10 \
  --seed 42 \
  --previews 48
```

All three ratios must be positive and must sum to 1. Repeating the same set, script version, parameters and seed gives the same split. After adding or removing source images, the split of the remaining images may change as well; for a reproducible experiment keep the generated dataset and the manifest.

### One box around all defects in an image

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --output ./yolo_chewinggum_union \
  --box-mode union
```

`union` is optional: it can merge distant defects into a very wide rectangle that also covers good parts. The default `components` is more precise in terms of localizing separate regions, but fragmented masks can produce many boxes.

### Padding or filtering tiny components

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --output ./yolo_chewinggum_padded \
  --padding 2
```

`--min-area 5` discards connected components with fewer than five marked pixels **at the original mask resolution**. The measure is the number of foreground pixels, not the area of the rectangle. This is not the default: legitimate tiny cracks could disappear. Every discard is recorded. If a defective image would lose all of its boxes, the script aborts instead of wrongly declaring it good.

`--connectivity 4` does not treat diagonal touching as connectivity. For this example keep the default value of 8.

`--previews 0` turns off the preview images. You can see all options like this:

```bash
python visa_yolo_tools/convert_visa_to_yolo.py --help
```

### Another original VisA category

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --category cashew \
  --output ./yolo_cashew
```

Here too, all non-zero defect types become the same `defect` class. The script does not generate separate YOLO classes for individual defect types.

## 6. Safeguards and limitations

**Masks.** Raw index values are read, including palette PNGs (`P` mode), without converting the palette to grayscale. This matters because index 1 does not have to have a brightness of 1 [6]. Binary 0/255 masks inside the original layout are also supported. Arbitrary colorized RGB masks are rejected; grayscale stored through identical RGB channels is acceptable. For RGBA/LA masks an opaque alpha is mandatory.

**Matching.** A missing mask, a mask without an image, an ambiguous file name, a corrupted image, an empty mask for a defective image, and differing image and mask dimensions all stop processing. There is no silent skipping of samples and no automatic resize. Images/masks with EXIF rotation are rejected so that the coordinate systems cannot end up misaligned.

**Pixel coordinates.** A box covers the full marked pixels. Because of that, even a one-pixel component has a positive width and height. The YOLO record is `class_id x_center y_center width height`, normalized to the image dimensions [4]. The algorithm uses OpenCV connected components [5]. Touching classes from the mask can end up in the same box because they are deliberately merged into a single foreground class.

**Duplicates and leakage.** Images with exactly identical decoded RGB pixels always stay in the same split, including files that differ only in metadata. If identical images have conflicting annotations, processing stops. The script does **not** recognize similar images, different shots of the same physical product, the same batch or the same capture session. This is therefore not a guarantee that all forms of data leakage are absent. The split is stratified by normal/anomaly status, **not** by all six defect types.

**Writing.** An existing output folder is never overwritten or deleted. For another attempt specify a new `--output`. The output cannot replace the source or its parent folders, nor be inside the source `Data/`. Processing first validates the input, then writes into a new temporary folder, and only after verification publishes the final folder. On error, only the temporary folder created by the script is removed. Image copies are verified with SHA-256. Do not modify the input files while the conversion is running.

**Meaning of the class.** This is detection of the defect location, not detection of the whole product, good/bad product classification, or measurement of the precise defect boundary. A bounding box does not necessarily represent one physical defect: connected components are a geometric, not a semantic, partition. The later absence of a model detection is not proof that the product is good.

## 7. Training and dataset license

To train in your existing Ultralytics environment, use the generated **`yolo_chewinggum/data.yaml`** as `data`. This tool does not install or run training.

Use only `train/val` for choosing parameters and keep the `test` split for the final evaluation. The split is a **new supervised experiment**: results on it are not comparable with the original VisA anomaly-detection benchmark, whose protocols use different splits [1, 2].

The VisA dataset is marked as CC BY 4.0 in the source repository [1, 7], and this tool does not change that license. `SOURCE_ATTRIBUTION.md` is automatically copied alongside the derived dataset; if you publish or redistribute images or derived data, keep the attribution, the license reference and the description of changes.

## 8. Verifying the script

From the `VisA` folder, after installing the dependencies:

```bash
python -m unittest discover -s visa_yolo_tools/tests -v
```

The tests create small synthetic inputs in the system temporary folder, process them and then remove them. They do not touch your dataset. Details of the verification performed and its limitations are in `TEST_REPORT_EN.md`.

## Sources

[1] Amazon Science — VisA description, folder layout, sample counts, license:
https://github.com/amazon-science/spot-diff

[2] Amazon Science — original mask reading/binarization and reorganization:
https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py

[3] Amazon Science — original image/mask pairing; index class definitions:
https://github.com/amazon-science/spot-diff/blob/main/split_csv/1cls.csv
https://github.com/amazon-science/spot-diff/blob/main/utils/id2class.py

[4] Ultralytics — YOLO detection format, normalized coordinates, dataset YAML:
https://docs.ultralytics.com/datasets/detect/

[5] OpenCV — connectedComponentsWithStats:
https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html

[6] Pillow — image modes, palette/index pixels:
https://pillow.readthedocs.io/en/stable/handbook/concepts.html

[7] Dataset license:
https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET
https://creativecommons.org/licenses/by/4.0/

The source layout and format were verified on 21 September 2026. This is a separately written conversion script; it is not an official Amazon or Ultralytics tool.
