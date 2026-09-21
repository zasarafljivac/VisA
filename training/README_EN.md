# VisA chewinggum → YOLO26n training on SageMaker

The training tools live in the **`training`** folder of this repository, next to the
converter in `visa_yolo_tools`. All commands are run from the **repository root**.

The main flow is: local preparation and validation → S3 dataset/source channels → one
SageMaker GPU training job → `model.tar.gz` with `best.pt`, `last.pt` and metadata.
You do not need to install PyTorch, CUDA or Ultralytics on your laptop to run this flow.

This is a standalone demonstration training example. It is not a production pipeline
and does not include model deployment.

## Quick start — Linux / WSL

No command starts an AWS job without an explicit `--execute`.

### 1. Folder layout

```text
VisA/
├── chewinggum/Data/Images/Normal/...
├── chewinggum/Data/Images/Anomaly/...
├── chewinggum/Data/Masks/Anomaly/...
├── yolo_chewinggum/                 # may already exist from the previous step
├── visa_yolo_tools/                 # mask → YOLO format converter
└── training/
    ├── run.py
    ├── config.yaml
    ├── config.local.example.yaml
    ├── requirements.txt             # local preparation and AWS controller only
    ├── source/                      # small package that is sent to SageMaker
    ├── tests/
    ├── weights/                     # yolo26n.pt goes here
    └── work/                        # created later; not in the repository
```

The original VisA archive must be extracted. A GitHub clone without the downloaded
images is not a dataset by itself. The original images/masks are not included in the
repository.

All relative paths in the configuration are resolved from the **repository root**,
regardless of the terminal's current working directory. If your source is e.g.
`datasets/VisA`, you can use `prepare --source datasets/VisA`.

### 2. Install only the local dependencies

The local controller needs Python 3.10 or newer. The SageMaker image uses its own
separate Python 3.10 and GPU libraries.

```bash
python3 -m venv training/.venv
source training/.venv/bin/activate
python -m pip install -r training/requirements.txt
```

Do not install the old requirements of the original `spot-diff` research code into
this environment. There is no need to run its model or the original anomaly-detection
training.

### 3. Prepare or reuse the YOLO dataset

```bash
python training/run.py prepare
```

If `yolo_chewinggum` already exists, it is validated and **the same split** is used,
with no new splitting or overwriting. If it does not exist, the converter from
`visa_yolo_tools` builds it from the original images and masks.

The conversion uses all non-zero mask pixels, a single class `0: defect`, connected
components and an approximately 70/15/15 train/val/test split, seed 42. Good images
have empty `.txt` annotations. Existing images are not redrawn. `previews/index.html`
is only a visual check of the annotations, and its images are not used for training.

For a new split or a different conversion choose a **new output dataset folder**; the
existing dataset is deliberately not overwritten. For the converter's special
parameters see `python visa_yolo_tools/convert_visa_to_yolo.py --help`.

### 4. Add the pretrained weights

```bash
python training/run.py download-weights
```

This command downloads only the official Ultralytics release asset `yolo26n.pt` and
stores its SHA-256. It does not use PyTorch, does not access AWS and does not start
training.

Alternative: copy your trusted `yolo26n.pt` into `training/weights/`, or change
`paths.weights` to an existing local file. An existing weights file is not
overwritten. Do not load unknown/untrusted `.pt` files; a model checkpoint is not
plain text. The recorded hash lets you track the identity of the file, but it is not
an independent verification of the publisher's signature. Weights are not included in
the repository.

### 5. Enter an existing S3 bucket

```bash
cp training/config.local.example.yaml training/config.local.yaml
```

In `config.local.yaml` change:

```yaml
aws:
  bucket: name-of-your-existing-bucket
```

The bucket must be in the region from `aws.region` (default **eu-central-1**),
private, and accessible to your AWS credentials and to the SageMaker execution role.
The tool does not create a bucket, IAM role, endpoint or deployment. By default the
standard AWS credential chain is used (e.g. `AWS_PROFILE` or the default profile);
you can set a named profile and a different region through `aws.profile` and
`aws.region` in `config.local.yaml`.

By default it looks for an existing IAM role `SageMakerExecutionRole` in your AWS
account. If you do not have the `iam:GetRole` permission, enter the full known ARN:

```yaml
aws:
  bucket: name-of-your-existing-bucket
  execution_role_arn: arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
```

Do not put an AWS access key/secret/session token into files. Your existing AWS
credential provider and profile are used. No AWS account ID is embedded in the tool.

### 6. Review the plan without AWS calls

```bash
python training/run.py plan --preset smoke
```

This checks the dataset, all three splits, possible leakage of identical images, the
weights and the configuration, and prints the request. **No AWS calls, uploads or GPU
costs.** If the role is given only by name, the plan shows a placeholder; the ARN is
resolved only on an explicit submit. The plan does not confirm IAM permissions, the
existence of the GPU image, quota or available capacity.

### 7. Run a small paid smoke training

```bash
python training/run.py submit --preset smoke --execute
```

`--execute` is required for the S3 upload and for creating the training job. Without
it `submit` performs only a local dry run. Before the upload, the AWS identity, the
bucket region and the execution role are checked. The job name, ARN and the location
of the local evidence are printed.

Smoke settings: **1 epoch, batch 1, workers 0, imgsz 640, one ml.g4dn.xlarge**.
This verifies the mechanics of the pipeline; it is not evidence of the quality of the
trained model. The maximum job runtime in this preset is 1,800 seconds. That is not
an estimate of the duration or total cost of the job.

### 8. Full training

After verifying the smoke job:

```bash
python training/run.py submit --preset full --execute
```

This preset uses **60 epochs**, batch 1, workers 0 and a runtime limit of 14,400
seconds. The number of epochs is an initial demonstration setting, not a guaranteed
optimal value. You can explicitly change e.g. the batch size or resolution:

```bash
python training/run.py plan --preset full --batch 4 --imgsz 960
python training/run.py submit --preset full --batch 4 --imgsz 960 --execute
```

The val set is used for validation and for choosing `best.pt`. **By default the test
images are not uploaded at all.** For the final experiment, once the choices and
parameters are frozen, you can explicitly include evaluation on the test set once:

```bash
python training/run.py submit --preset full --evaluate-test --execute
```

The test set is then sent to the job, but it is used only **after training**, to
evaluate the selected `best.pt`. Do not tune parameters afterwards based on the test
results and then present them as an unbiased final evaluation.

## Monitoring, stopping and downloading

Replace `JOB_NAME` with the actual name from the submit output:

```bash
python training/run.py status --job-name JOB_NAME
python training/run.py status --job-name JOB_NAME --wait
python training/run.py download --job-name JOB_NAME
```

`download` works once the job is `Completed`. The archive is checked before
extraction: paths outside the output, symbolic links and overwriting existing results
are not allowed. Models and reports end up here:

```text
training/work/jobs/JOB_NAME/
├── create-training-job.json
├── config-snapshot.json
├── input-manifest.json
├── dataset-validation.json
├── submission.json
├── model.tar.gz
└── artifacts/
    ├── best.pt
    ├── last.pt
    ├── metrics.json
    ├── results.csv
    ├── environment.json
    ├── training_config.json
    ├── resolved_training_args.json
    ├── dataset_validation.json
    ├── data.yaml
    ├── classes.txt
    └── ... available plots and source metadata
```

The `data.yaml` in the model artifact is a portable description; it **does not
contain the dataset images**. For local inference/evaluation use the real
`yolo_chewinggum/data.yaml`, or set its `path` to the location of the downloaded
dataset.

CloudWatch logs, via the AWS CLI:

```bash
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix "JOB_NAME/" \
  --follow --region eu-central-1
```

Stopping a job:

```bash
python training/run.py stop --job-name JOB_NAME --execute
```

**Closing the terminal or pressing Ctrl+C during a local wait does NOT stop the
SageMaker job.** The stop command sends a request; use status to check that the job
is really `Stopped`. The tool does not delete S3 inputs, models, logs or job evidence.
S3/log storage can remain after training ends. There is no automatic retry of a job
under a different name.

If the CreateTrainingJob call ends with a network error, first check the printed job
name through `status`/the AWS console: the job may have been created even though the
response did not arrive.

## Versions and runtime settings

| Item | Setting |
|---|---|
| Task | `detect`, single class `defect` |
| Starting checkpoint | `yolo26n.pt`; no substitution with another model |
| Ultralytics | `8.4.60`, deliberately frozen reference version |
| GPU container | AWS PyTorch 2.2.0 / Python 3.10 / CUDA 12.1 |
| AWS profile / region | default credential chain / `eu-central-1` |
| Instance | 1 × `ml.g4dn.xlarge` |
| Seed | 42 |
| Optimizer | `auto`, the choice is left to the pinned Ultralytics version |
| AMP | `false`, explicit FP32 for this example |
| Train/val/test | New supervised split, not the original VisA benchmark |

`amp=false` is a choice of this example; it avoids the auxiliary AMP check, which
could download an additional model. The checkpoint is verified as a YOLO26 nano
detection model before training. An unavailable GPU aborts execution instead of a
silent CPU fallback.

The bootstrap installs the pinned Ultralytics/NumPy/OpenCV packages **inside the GPU
container**. Constraints keep PyTorch 2.2.0 and torchvision 0.17.0; a conflict aborts
the job instead of a hidden upgrade of the CUDA/PyTorch environment. If needed it
installs `libgl1` and `libglib2.0-0` inside the container. Outbound access to the
package repositories is required. This is not a hermetic offline image nor a
security-audited production container; the actually installed versions are recorded
in `environment.json`.

## Optional ONNX export

```bash
python training/run.py submit --preset full --export-onnx --execute
```

Adds `best.onnx` and `onnx_metadata.json` after training. The export is FP32, batch 1,
static, opset 17, without simplification. The actual I/O names and dimensions are
recorded instead of inventing an output format for some version of YOLO26.

This is a generic export, **not a verified deployment contract for a specific edge
device**. It does not build a TensorRT engine. ONNX export was not executed/tested
here; run an initial smoke job with `--export-onnx` before a full job for which ONNX
is mandatory.

## Optional direct local training

The main workflow above remains SageMaker. For a local GPU use a **separate**
environment with Python 3.10 or 3.11, because the reference PyTorch 2.2.0 has no
wheel for newer Python versions. Example for Linux/WSL with a compatible NVIDIA/CUDA
driver:

```bash
python3.10 -m venv training/.venv-gpu
training/.venv-gpu/bin/python -m pip install \
  torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
training/.venv-gpu/bin/python -m pip install \
  -c training/source/constraints.txt -r training/source/requirements.txt
python training/run.py local --preset smoke --python training/.venv-gpu/bin/python
```

`local` uses the same `source/train.py`, but with local output paths. It sends nothing
to AWS. For a deliberate CPU test add `--device cpu`; there is no automatic CPU
fallback. This local GPU flow was not executed while verifying the tool.

## Permissions and expected errors

The local AWS profile needs the appropriate `sts:GetCallerIdentity`,
`s3:GetBucketLocation`, S3 upload/download, `sagemaker:CreateTrainingJob`,
`DescribeTrainingJob`, `StopTrainingJob`, `iam:PassRole` and, if needed, `iam:GetRole`.
Viewing CloudWatch logs requires additional read permissions. The execution role must
have a SageMaker trust policy and access to the chosen S3 prefix, the logs and the GPU
image. With KMS, the corresponding key policy/grants are needed as well. This is not
an automatic IAM setup.

`ResourceLimitExceeded` usually means you should check the SageMaker training quota
for the exact instance type and region; the EC2 quota is not the same thing. An
approved quota is no guarantee of free capacity. The script does not change quotas,
the region or the instance type as a fallback.

No private repo, `.env`, complete repository or production dataset is uploaded. The
upload is limited to the specific validated dataset files, a few metadata files, six
source files, the runtime configuration and the chosen checkpoint.

## Tests, limitations and sources

```bash
python -m unittest discover -s training/tests -v
```

See `TEST_REPORT.md` for the checks that were actually performed. The offline tests
include a mock/fake model to verify the orchestration code; they **do not represent
YOLO training**. No AWS job was run, model accuracy was not measured, the full VisA
data and the weights were not downloaded, and ONNX export was not executed. Report
only your own actual training/evaluation results as results.

Sources for the public technical contracts (verified on 21 September 2026):

- Ultralytics YOLO26, `yolo26n.pt`, train/val/export:
  https://docs.ultralytics.com/models/yolo26/
- Ultralytics training API:
  https://docs.ultralytics.com/modes/train/
- YOLO detection dataset format:
  https://docs.ultralytics.com/datasets/detect/
- Ultralytics package release history:
  https://pypi.org/project/ultralytics/
- Official pretrained release:
  https://github.com/ultralytics/assets/releases/tag/v8.4.0
- AWS AlgorithmSpecification / ContainerEntrypoint:
  https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AlgorithmSpecification.html
- SageMaker File-mode channels and their paths:
  https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html
- AWS PyTorch 2.2.0 container reference:
  https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-data-parallel-support.html
- VisA dataset and annotations:
  https://github.com/amazon-science/spot-diff

Credit the VisA data with the CC BY 4.0 attribution and a description of the
conversion/split; see `visa_yolo_tools/SOURCE_ATTRIBUTION.md`. The dataset license is
not a substitute for the separate terms of use of the Ultralytics software/weights,
which are stated on its official site. Do not publish private AWS identifiers or full
local paths from logs without reviewing them. The tool does not publish anything on
its own.
