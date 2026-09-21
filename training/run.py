#!/usr/bin/env python3
"""VisA training controller. AWS mutations require an explicit `--execute`.

Run `python training/run.py --help` from any working directory.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / 'source'))
from common import MODEL_NAME, ULTRALYTICS_VERSION, read_yaml, sha256, validate_dataset, portable_yaml, write_json

WEIGHTS_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt'
SOURCE_FILES = ('train.py', 'common.py', 'bootstrap.sh', 'requirements.txt', 'constraints.txt', 'requirements-export.txt')
META_FILES = ('classes.txt', 'SOURCE_ATTRIBUTION.md', 'conversion_report.json', 'manifest.csv')
CONVERTER = ROOT / 'visa_yolo_tools/convert_visa_to_yolo.py'
TERMINAL_STATES = {'Completed', 'Failed', 'Stopped'}


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'Expected a mapping for {key}')
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None) -> dict:
    cfg = read_yaml(path or HERE / 'config.yaml')
    if path is None and (HERE / 'config.local.yaml').is_file():
        cfg = deep_merge(cfg, read_yaml(HERE / 'config.local.yaml'))
    if cfg.get('schema_version') != 1:
        raise ValueError('Unsupported config schema version.')
    if cfg['training']['model'] != MODEL_NAME or str(cfg['training']['ultralytics_version']) != ULTRALYTICS_VERSION:
        raise ValueError('This package is pinned to yolo26n.pt / Ultralytics 8.4.60. No fallback model.')
    if cfg['aws']['instance_count'] != 1:
        raise ValueError('Only one training instance is supported.')
    if not re.fullmatch(r'[a-z]{2}-[a-z]+-\d', cfg['aws']['region']):
        raise ValueError('Invalid AWS Region.')
    if not re.fullmatch(r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,29})', cfg['aws']['job_prefix']):
        raise ValueError('job_prefix must have 1-30 alphanumeric/hyphen characters.')
    return cfg


def local_path(value: str) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def settings_for(cfg: dict, args) -> tuple[dict, int]:
    tr = copy.deepcopy(cfg['training'])
    preset = cfg['presets'][args.preset]
    max_seconds = preset.get('max_run_seconds', cfg['aws']['max_run_seconds'])
    tr.update({k: v for k, v in preset.items() if k != 'max_run_seconds'})
    for key in ('epochs', 'batch', 'imgsz', 'workers'):
        value = getattr(args, key, None)
        if value is not None:
            tr[key] = value
    if getattr(args, 'evaluate_test', False):
        tr['evaluate_test'] = True
    if getattr(args, 'export_onnx', False):
        tr['export_onnx'] = True
    for key in ('epochs', 'batch', 'imgsz'):
        if isinstance(tr[key], bool) or not isinstance(tr[key], int) or tr[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ('workers', 'patience', 'seed'):
        if isinstance(tr[key], bool) or not isinstance(tr[key], int) or tr[key] < 0:
            raise ValueError(f'{key} must be a non-negative integer.')
    if tr['imgsz'] % 32:
        raise ValueError('imgsz must be divisible by 32, e.g. 640 or 960.')
    for key in ('amp', 'deterministic', 'evaluate_test', 'export_onnx'):
        if not isinstance(tr[key], bool):
            raise ValueError(f'{key} must be a YAML boolean.')
    if not isinstance(max_seconds, int) or max_seconds < 60:
        raise ValueError('max_run_seconds must be an integer >= 60.')
    return tr, max_seconds


def check_aws_config(cfg: dict) -> None:
    aws = cfg['aws']
    bucket = aws['bucket']
    if not isinstance(bucket, str) or not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', bucket):
        raise ValueError('Set aws.bucket to an existing S3 bucket in training/config.local.yaml.')
    prefix = aws['prefix']
    if not isinstance(prefix, str) or not prefix.strip('/') or any(x in {'.', '..'} for x in prefix.split('/')):
        raise ValueError('Use a nonempty dedicated aws.prefix without . or .. segments.')
    if not str(aws['instance_type']).startswith(('ml.g', 'ml.p')):
        raise ValueError('A GPU SageMaker instance (ml.g* or ml.p*) is required.')
    if not isinstance(aws['volume_size_gb'], int) or aws['volume_size_gb'] < 5:
        raise ValueError('volume_size_gb must be >= 5.')
    if aws['execution_role_arn'] and not re.fullmatch(r'arn:aws:iam::\d{12}:role/.+', aws['execution_role_arn']):
        raise ValueError('Invalid execution_role_arn.')


def new_job_name(cfg: dict) -> str:
    return cfg['aws']['job_prefix'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]


def validate_job_name(name: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', name):
        raise ValueError('Invalid SageMaker job name.')
    return name


def create_request(cfg: dict, name: str, training: dict, max_seconds: int, role_arn: str) -> dict:
    aws = cfg['aws']
    prefix = f"{aws['prefix'].strip('/')}/{validate_job_name(name)}"
    uri = f"s3://{aws['bucket']}/{prefix}"
    image = aws['image_uri'].format(region=aws['region'])
    request = {
        'TrainingJobName': name, 'RoleArn': role_arn,
        'AlgorithmSpecification': {
            'TrainingImage': image, 'TrainingInputMode': 'File',
            'ContainerEntrypoint': ['/bin/bash', '/opt/ml/input/data/source/bootstrap.sh'],
            'MetricDefinitions': [
                {'Name': 'validation:mAP50', 'Regex': r'VISA_METRIC metrics/mAP50\(B\)=([0-9.eE+\-]+)'},
                {'Name': 'validation:mAP50-95', 'Regex': r'VISA_METRIC metrics/mAP50-95\(B\)=([0-9.eE+\-]+)'},
            ],
        },
        'InputDataConfig': [
            {'ChannelName': channel, 'DataSource': {'S3DataSource': {
                'S3DataType': 'S3Prefix', 'S3Uri': f'{uri}/{channel}/', 'S3DataDistributionType': 'FullyReplicated'}},
             'InputMode': 'File', 'CompressionType': 'None'}
            for channel in ('dataset', 'source')
        ],
        'OutputDataConfig': {'S3OutputPath': f'{uri}/output/'},
        'ResourceConfig': {'InstanceType': aws['instance_type'], 'InstanceCount': 1, 'VolumeSizeInGB': aws['volume_size_gb']},
        'StoppingCondition': {'MaxRuntimeInSeconds': max_seconds},
        'EnableManagedSpotTraining': False,
        'EnableNetworkIsolation': False,  # package installation requires outbound network
        'HyperParameters': {k: str(training[k]) for k in ('model', 'epochs', 'batch', 'workers', 'imgsz', 'seed')},
        'Environment': {'PYTHONUNBUFFERED': '1', 'YOLO_CONFIG_DIR': '/tmp/visa-yolo-settings', 'YOLO_AUTOINSTALL': 'false'},
        'Tags': [{'Key': 'Project', 'Value': 'VisA-YOLO'}, {'Key': 'Dataset', 'Value': 'chewinggum'}],
    }
    if aws['kms_key_arn']:
        request['OutputDataConfig']['KmsKeyId'] = aws['kms_key_arn']
        request['ResourceConfig']['VolumeKmsKeyId'] = aws['kms_key_arn']
    return request


def prepare_dataset(cfg: dict, args) -> None:
    output = local_path(cfg['paths']['dataset'])
    if output.exists():
        summary = validate_dataset(output)
        print('Reusing the existing validated split; it has NOT been regenerated.')
        print(json.dumps({k: v for k, v in summary.items() if k != 'files'}, indent=2))
        return
    conv = cfg['conversion']
    source = local_path(args.source or cfg['paths']['visa_source'])
    subprocess.run([sys.executable, str(CONVERTER),
                    '--source', str(source), '--output', str(output), '--category', conv['category'],
                    '--split', *map(str, conv['split']), '--seed', str(conv['seed']),
                    '--previews', str(conv['previews'])], check=True)
    print(json.dumps({k: v for k, v in validate_dataset(output).items() if k != 'files'}, indent=2))


def download_weights(cfg: dict) -> None:
    output = local_path(cfg['paths']['weights'])
    if output.exists():
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError(f'Invalid existing weights file: {output}')
        print(f'Existing weights retained: {output}\nSHA-256: {sha256(output)}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + '.partial-' + uuid.uuid4().hex[:8])
    try:
        request = Request(WEIGHTS_URL, headers={'User-Agent': 'VisA-YOLO26-Training/1.0'})
        with urlopen(request, timeout=60) as response, tmp.open('xb') as stream:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 100 * 1024 * 1024:
                    raise ValueError('Unexpectedly large model download.')
                stream.write(chunk)
        if tmp.stat().st_size < 100_000:
            raise ValueError('Downloaded file is too small to be the requested checkpoint.')
        with tmp.open('rb') as stream:
            if stream.read(2) != b'PK':
                raise ValueError('Expected a PyTorch ZIP-format checkpoint, not HTML/error text.')
        # Do not overwrite a weights file created while this download was running.
        with output.open('xb') as stream, tmp.open('rb') as source:
            shutil.copyfileobj(source, stream)
        write_json(output.with_suffix('.download.json'), {'url': WEIGHTS_URL, 'sha256': sha256(output),
                   'bytes': output.stat().st_size, 'downloaded_at': datetime.now(timezone.utc).isoformat()})
        print(f'Downloaded {output}\nSHA-256: {sha256(output)}')
    finally:
        tmp.unlink(missing_ok=True)


def session_for(cfg: dict):
    import boto3
    from botocore.config import Config
    session = boto3.Session(profile_name=cfg['aws']['profile'] or None, region_name=cfg['aws']['region'])
    return session, Config(retries={'max_attempts': 3, 'mode': 'standard'}, connect_timeout=10, read_timeout=60)


def aws_preflight(cfg: dict):
    session, client_config = session_for(cfg)
    identity = session.client('sts', config=client_config).get_caller_identity()
    s3 = session.client('s3', config=client_config)
    bucket_region = s3.get_bucket_location(Bucket=cfg['aws']['bucket']).get('LocationConstraint') or 'us-east-1'
    if bucket_region == 'EU':
        bucket_region = 'eu-west-1'
    if bucket_region != cfg['aws']['region']:
        raise ValueError(f'Bucket region {bucket_region} differs from training region {cfg["aws"]["region"]}.')
    role = cfg['aws']['execution_role_arn']
    if not role:
        role = session.client('iam', config=client_config).get_role(RoleName=cfg['aws']['role_name'])['Role']['Arn']
    # Fail closed on an unexpected cross-account execution role.
    if role.split(':')[4] != identity['Account']:
        raise ValueError('Execution role and selected AWS profile belong to different accounts.')
    print(f"AWS identity: {identity['Arn']}\nExecution role: {role}\nBucket region: {bucket_region}")
    return session, client_config, s3, role, identity


def stage_job(cfg: dict, name: str, tr: dict, report: dict) -> Path:
    dataset = local_path(cfg['paths']['dataset'])
    weights = local_path(cfg['paths']['weights'])
    job_dir = local_path(cfg['paths']['work']) / 'jobs' / name
    job_dir.mkdir(parents=True, exist_ok=False)
    stage = job_dir / 'input'
    (stage / 'source/weights').mkdir(parents=True)
    for entry in report['files']:
        origin, dest = dataset / entry['path'], stage / 'dataset' / entry['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)
        if sha256(dest) != entry['sha256']:
            raise ValueError(f'Dataset changed during staging: {origin}')
    portable_yaml(stage / 'dataset/data.yaml', '.', tr['evaluate_test'])
    for name in META_FILES:
        origin = dataset / name
        if origin.is_file() and not origin.is_symlink():
            shutil.copy2(origin, stage / 'dataset' / name)
    for name in SOURCE_FILES:
        shutil.copy2(HERE / 'source' / name, stage / 'source' / name)
    snapshot_weights = stage / 'source/weights' / MODEL_NAME
    before = sha256(weights)
    shutil.copy2(weights, snapshot_weights)
    if sha256(snapshot_weights) != before:
        raise ValueError('Weights changed during staging.')
    runtime = {'training': tr, 'dataset': '/opt/ml/input/data/dataset',
               'weights': '/opt/ml/input/data/source/weights/yolo26n.pt',
               'weights_sha256': sha256(snapshot_weights),
               'dataset_fingerprint': report['dataset_fingerprint'],
               'run_dir': '/opt/ml/output/data/run', 'model_dir': '/opt/ml/model', 'device': '0'}
    write_json(stage / 'source/run-config.json', runtime)
    write_json(job_dir / 'dataset-validation.json', report)
    write_json(job_dir / 'config-snapshot.json', cfg)
    write_json(job_dir / 'input-manifest.json', [
        {'path': p.relative_to(stage).as_posix(), 'sha256': sha256(p), 'bytes': p.stat().st_size}
        for p in sorted(stage.rglob('*')) if p.is_file()])
    return job_dir


def submit_or_plan(cfg: dict, args) -> None:
    check_aws_config(cfg)
    tr, max_seconds = settings_for(cfg, args)
    dataset = local_path(cfg['paths']['dataset'])
    # Validate ALL splits before any remote write, including leakage into held-out test.
    all_report = validate_dataset(dataset, include_test=True)
    report = all_report if tr['evaluate_test'] else validate_dataset(dataset, include_test=False)
    weights = local_path(cfg['paths']['weights'])
    if not weights.is_file() or not weights.stat().st_size:
        raise ValueError('Missing weights. Run download-weights or set paths.weights to your trusted yolo26n.pt.')
    name = validate_job_name(args.job_name) if args.job_name else new_job_name(cfg)
    role = cfg['aws']['execution_role_arn'] or f"<resolve IAM role {cfg['aws']['role_name']}>"
    request = create_request(cfg, name, tr, max_seconds, role)
    mode = 'SUBMIT REQUESTED - AWS preflight follows' if args.command == 'submit' and args.execute else 'DRY RUN - no AWS calls'
    summary = {'mode': mode, 'training': tr,
               'split_counts': all_report['splits'], 'uploaded_splits': list(report['splits']),
               'weights_sha256': sha256(weights), 'dataset_fingerprint': report['dataset_fingerprint'],
               'request': request}
    print(json.dumps(summary, indent=2))
    if args.command == 'plan' or not args.execute:
        print('\nNo files uploaded and no training job created. Use submit --execute to authorize AWS writes.')
        return
    session, client_config, s3, role, identity = aws_preflight(cfg)
    request['RoleArn'] = role
    sm = session.client('sagemaker', config=client_config)
    # Avoid collisions BEFORE upload. A retry must never submit a second job with a new name automatically.
    from botocore.exceptions import ClientError
    try:
        sm.describe_training_job(TrainingJobName=name)
    except ClientError as exc:
        code = exc.response['Error']['Code']
        message = str(exc).lower()
        absent = code in {'ResourceNotFound', 'ResourceNotFoundException'} or (
            code == 'ValidationException' and any(x in message for x in ('not find', 'not found', 'does not exist')))
        if not absent:
            raise
    else:
        raise ValueError(f'Training job already exists: {name}')
    job_dir = stage_job(cfg, name, tr, report)
    write_json(job_dir / 'create-training-job.json', request)
    write_json(job_dir / 'aws-identity.json', identity)
    bucket = cfg['aws']['bucket']
    prefix = f"{cfg['aws']['prefix'].strip('/')}/{name}"
    extra = {'ServerSideEncryption': 'aws:kms', 'SSEKMSKeyId': cfg['aws']['kms_key_arn']} if cfg['aws']['kms_key_arn'] else {'ServerSideEncryption': 'AES256'}
    stage = job_dir / 'input'
    uploads = json.loads((job_dir / 'input-manifest.json').read_text())
    print(f'Uploading {len(uploads)} allowlisted files to s3://{bucket}/{prefix}/', flush=True)
    for entry in uploads:
        p = stage / entry['path']
        s3.upload_file(str(p), bucket, prefix + '/' + entry['path'], ExtraArgs=extra)
    # Exactly one explicit CreateTrainingJob request; network ambiguity is not auto-retried by us.
    # Use a client without automatic retries for this paid operation.
    from botocore.config import Config
    creator = session.client('sagemaker', config=Config(retries={'total_max_attempts': 1}, connect_timeout=10, read_timeout=60))
    try:
        response = creator.create_training_job(**request)
    except Exception:
        print(f'CreateTrainingJob did not return success. Check status for {name} before retrying.\n'
              f'Prepared request retained in {job_dir}. S3 inputs are retained, not deleted.', file=sys.stderr)
        raise
    write_json(job_dir / 'submission.json', {'job_name': name, 'job_arn': response['TrainingJobArn'],
                'region': cfg['aws']['region'], 'bucket': bucket, 'prefix': prefix})
    print(f"\nSubmitted: {name}\nARN: {response['TrainingJobArn']}\nLocal evidence: {job_dir}")
    print(f'python training/run.py status --job-name {name}\npython training/run.py download --job-name {name}')
    print('Closing this terminal does NOT stop the SageMaker job. Use stop --execute to request stopping it.')


def status(cfg: dict, name: str, wait: bool = False) -> dict:
    session, client_config = session_for(cfg)
    sm = session.client('sagemaker', config=client_config)
    while True:
        result = sm.describe_training_job(TrainingJobName=validate_job_name(name))
        fields = ('TrainingJobName', 'TrainingJobArn', 'TrainingJobStatus', 'SecondaryStatus', 'FailureReason',
                  'TrainingTimeInSeconds', 'BillableTimeInSeconds', 'ModelArtifacts', 'FinalMetricDataList')
        short = {k: result[k] for k in fields if k in result}
        print(json.dumps(short, indent=2, default=str), flush=True)
        if not wait or result['TrainingJobStatus'] in TERMINAL_STATES:
            return result
        time.sleep(15)


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract regular files/dirs only, with path, duplicate and expansion limits."""
    if destination.exists():
        raise ValueError('Artifact destination already exists; refusing overwrite.')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) > 20000 or sum(m.size for m in members) > 5 * 1024**3:
            raise ValueError('Archive exceeds extraction limits.')
        seen = set()
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in member.name or ':' in member.name:
                raise ValueError(f'Unsafe archive path: {member.name}')
            if not (member.isfile() or member.isdir()):
                raise ValueError(f'Links/special files are forbidden: {member.name}')
            key = str(relative)
            if member.isfile() and key in seen:
                raise ValueError('Duplicate archive entry.')
            seen.add(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.extract-', dir=destination.parent))
        try:
            for member in members:
                target = temporary / member.name
                if not target.resolve().is_relative_to(temporary.resolve()):
                    raise ValueError('Archive path escapes destination.')
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open('xb') as output:
                        shutil.copyfileobj(source, output)
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary)
            raise


def download_artifacts(cfg: dict, name: str) -> None:
    result = status(cfg, name)
    if result['TrainingJobStatus'] != 'Completed':
        raise ValueError('Download is allowed only for a Completed job.')
    uri = urlparse(result['ModelArtifacts']['S3ModelArtifacts'])
    if uri.scheme != 's3' or not uri.netloc or not uri.path:
        raise ValueError('Unexpected model artifact URI.')
    job_dir = local_path(cfg['paths']['work']) / 'jobs' / validate_job_name(name)
    dest = job_dir / 'artifacts'
    if dest.exists():
        raise ValueError(f'Artifacts already exist: {dest}')
    job_dir.mkdir(parents=True, exist_ok=True)
    archive = job_dir / 'model.tar.gz'
    partial = job_dir / ('model.tar.gz.partial-' + uuid.uuid4().hex[:8])
    session, client_config = session_for(cfg)
    try:
        session.client('s3', config=client_config).download_file(uri.netloc, uri.path.lstrip('/'), str(partial))
        safe_extract(partial, dest)
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    print(f'Model and metadata: {dest}\nArchive SHA-256: {sha256(archive)}')


def stop_job(cfg: dict, name: str, execute: bool) -> None:
    validate_job_name(name)
    if not execute:
        print(f'DRY RUN: would request StopTrainingJob for {name}. Add --execute. No AWS calls made.')
        return
    session, client_config = session_for(cfg)
    session.client('sagemaker', config=client_config).stop_training_job(TrainingJobName=name)
    print(f'Stop requested for {name}; use status to confirm Stopped. Inputs/artifacts were not deleted.')


def local_training(cfg: dict, args) -> None:
    tr, _ = settings_for(cfg, args)
    validate_dataset(local_path(cfg['paths']['dataset']), include_test=True)
    weights = local_path(cfg['paths']['weights'])
    if not weights.is_file():
        raise ValueError('Missing trusted yolo26n.pt weights.')
    name = new_job_name(cfg)
    job = local_path(cfg['paths']['work']) / 'local' / name
    job.mkdir(parents=True, exist_ok=False)
    runtime = {'training': tr, 'dataset': str(local_path(cfg['paths']['dataset'])), 'weights': str(weights),
               'weights_sha256': sha256(weights), 'run_dir': str(job / 'run'),
               'model_dir': str(job / 'artifacts'), 'device': args.device}
    write_json(job / 'run-config.json', runtime)
    subprocess.run([args.python or sys.executable, str(HERE / 'source/train.py'), '--config', str(job / 'run-config.json')], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Alternate full YAML config; otherwise config.yaml + config.local.yaml')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Convert masks, or validate/reuse an already converted dataset')
    prepare.add_argument('--source', help='Original VisA root/category path, relative to the repository root')
    sub.add_parser('validate', help='Validate all three dataset splits without AWS')
    sub.add_parser('download-weights', help='Download the official YOLO26n checkpoint; no training/AWS')
    for command in ('plan', 'submit', 'local'):
        cmd = sub.add_parser(command, help={'plan': 'Offline job plan, no AWS calls',
                  'submit': 'Dry run unless --execute is supplied', 'local': 'Optional direct GPU training using the same entry point'}[command])
        cmd.add_argument('--preset', choices=('smoke', 'full'), default='smoke')
        for key in ('epochs', 'batch', 'imgsz', 'workers'):
            cmd.add_argument('--' + key, type=int)
        cmd.add_argument('--evaluate-test', action='store_true', help='Explicit final test evaluation AFTER training')
        cmd.add_argument('--export-onnx', action='store_true')
        if command != 'local':
            cmd.add_argument('--job-name', help='Optional explicit job name; no automatic job retry')
        if command == 'submit':
            cmd.add_argument('--execute', action='store_true', help='Authorize S3 uploads and ONE paid SageMaker training job')
        if command == 'local':
            cmd.add_argument('--python', help='Python executable in a separate GPU training environment')
            cmd.add_argument('--device', default='0', choices=('0', 'cpu'), help='No automatic CPU fallback')
    for command in ('status', 'download', 'stop'):
        cmd = sub.add_parser(command)
        cmd.add_argument('--job-name', required=True)
        if command == 'status':
            cmd.add_argument('--wait', action='store_true', help='Poll until terminal; Ctrl+C does not stop the job')
        if command == 'stop':
            cmd.add_argument('--execute', action='store_true')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == 'prepare':
        prepare_dataset(cfg, args)
    elif args.command == 'validate':
        print(json.dumps({k: v for k, v in validate_dataset(local_path(cfg['paths']['dataset'])).items() if k != 'files'}, indent=2))
    elif args.command == 'download-weights':
        download_weights(cfg)
    elif args.command in ('plan', 'submit'):
        submit_or_plan(cfg, args)
    elif args.command == 'status':
        result = status(cfg, args.job_name, args.wait)
        return 1 if result['TrainingJobStatus'] in {'Failed', 'Stopped'} else 0
    elif args.command == 'download':
        download_artifacts(cfg, args.job_name)
    elif args.command == 'stop':
        stop_job(cfg, args.job_name, args.execute)
    elif args.command == 'local':
        local_training(cfg, args)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted locally. Any running SageMaker job continues; use stop --execute to stop it.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
