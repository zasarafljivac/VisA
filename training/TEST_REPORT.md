# Verification report — VisA YOLO26n training toolkit

Date: 2026-09-21

## Executed

- **31 offline controller/training regression tests passed** using:
  `python -m unittest discover -s training/tests -v`.
  The 38 converter regressions live in `visa_yolo_tools/tests` and are reported in
  `visa_yolo_tools/TEST_REPORT_EN.md`.
- Python syntax compilation passed for every included `.py` file.
- `bash -n training/source/bootstrap.sh` passed.
- The generated SageMaker `CreateTrainingJob` request passed local validation
  against the installed botocore SageMaker service model.
- Five real CLI invocations passed from a **different current working directory**,
  with the tools placed inside a temporary VisA-shaped repository:
  `prepare`, `validate`, `plan --preset smoke`,
  `plan --preset full --evaluate-test`, and a second `prepare`.
- That CLI fixture had **503 synthetic normal images + 100 synthetic anomaly
  images**, with synthetic class-index masks. It generated 422 train, 91 val,
  and 90 test images. The second prepare reused the existing split.
- Dry-run CLI checks did not create a work/job directory or make AWS calls.

## Test coverage

Together, the 38 converter regressions and the 31 controller/training tests cover:

- Original VisA-style layout, index masks 1–6, connected components, normalized
  boxes, empty negative labels and image/mask consistency.
- Consistent split reuse and exact-image leakage checks across train/val/test.
- Rebased runtime data.yaml rather than accidentally retaining a host path.
- Default exclusion of test images, explicit opt-in test inclusion, upload
  allowlisting and exclusion of arbitrary `.env` / preview files.
- Fixed YOLO26 nano model/version guards and training parameter validation.
- Offline planning, no-execute guards, preflight failure handling, a single
  explicit paid-job submission path tested with fake AWS objects only.
- Artifact paths, best/last checkpoint copying, metadata writing and refusal
  to overwrite an existing run, tested with a **fake model object**.
- Safe model archive extraction and rejection of traversal/absolute/link entries.

## NOT executed / NOT established

- No real YOLO model was trained. The model object used in orchestration tests is
  a test double; its fake weights and placeholder metrics are not real results.
- No SageMaker job was created, no S3 upload was made, and no GPU charge incurred.
- No AWS identity, permission, quota, GPU capacity or ECR image availability was
  verified against a real AWS account.
- The pinned GPU dependencies were **not installed together** or exercised in
  the specified AWS PyTorch 2.2.0 container. The bootstrap syntax was checked,
  not its actual package installation or CUDA runtime behavior.
- The full original VisA archive and official yolo26n.pt were not downloaded.
  CLI fixtures use synthetic images and a clearly fake local checkpoint file
  solely for offline planning. These fixtures are not included in the repository.
- ONNX export and TensorRT/edge deployment were not executed.

The included smoke preset is intended to let you verify the actual cloud
runtime before a longer training run. Passing offline tests does not prove
GPU compatibility, model accuracy, production readiness or security certification.
