"""Offline regression tests. Fake model/backend tests are NOT real YOLO training."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'source'))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'visa_yolo_tools/tests'))
import run as runner
import common
import train as training
import visa_yolo_tools.convert_visa_to_yolo as converter
from test_converter import make_dataset


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = runner.load_config(HERE / 'config.yaml')

    def test_exact_model_version_and_instance(self):
        self.assertEqual(self.cfg['training']['model'], 'yolo26n.pt')
        self.assertEqual(self.cfg['training']['ultralytics_version'], '8.4.60')
        self.assertEqual(self.cfg['aws']['instance_type'], 'ml.g4dn.xlarge')
        self.assertEqual(self.cfg['aws']['instance_count'], 1)

    def test_smoke_settings(self):
        args = runner.build_parser().parse_args(['plan'])
        tr, limit = runner.settings_for(self.cfg, args)
        self.assertEqual((tr['epochs'], tr['batch'], tr['workers']), (1, 1, 0))
        self.assertFalse(tr['evaluate_test'])
        self.assertFalse(tr['export_onnx'])
        self.assertEqual(limit, 1800)

    def test_full_and_overrides(self):
        args = runner.build_parser().parse_args(['plan', '--preset', 'full', '--batch', '4', '--evaluate-test'])
        tr, limit = runner.settings_for(self.cfg, args)
        self.assertEqual((tr['epochs'], tr['batch']), (60, 4))
        self.assertTrue(tr['evaluate_test'])
        self.assertEqual(limit, 14400)

    def test_cli_negative_values_rejected(self):
        args = runner.build_parser().parse_args(['plan', '--batch', '-1'])
        with self.assertRaises(ValueError):
            runner.settings_for(self.cfg, args)

    def test_bad_imgsz_rejected(self):
        args = runner.build_parser().parse_args(['plan', '--imgsz', '641'])
        with self.assertRaises(ValueError):
            runner.settings_for(self.cfg, args)

    def test_deep_merge_retains_other_values(self):
        result = runner.deep_merge(self.cfg, {'aws': {'bucket': 'test-bucket'}})
        self.assertEqual(result['aws']['region'], 'eu-central-1')
        self.assertEqual(self.cfg['aws']['bucket'], '')

    def test_unknown_config_key_rejected(self):
        with self.assertRaises(ValueError):
            runner.deep_merge(self.cfg, {'aws': {'buckte': 'typo'}})

    def test_no_bucket_rejected(self):
        with self.assertRaises(ValueError):
            runner.check_aws_config(self.cfg)

    def test_role_arn_not_embedded(self):
        self.assertEqual(self.cfg['aws']['execution_role_arn'], '')
        self.assertEqual(self.cfg['aws']['role_name'], 'SageMakerExecutionRole')

    def test_valid_sagemaker_request_schema(self):
        from botocore.session import get_session
        from botocore.validate import validate_parameters
        self.cfg['aws']['bucket'] = 'example-test-bucket'
        tr, seconds = runner.settings_for(self.cfg, runner.build_parser().parse_args(['plan']))
        req = runner.create_request(self.cfg, 'visa-yolo-train-offline-test', tr, seconds,
                                    'arn:aws:iam::123456789012:role/SageMakerExecutionRole')
        shape = get_session().get_service_model('sagemaker').operation_model('CreateTrainingJob').input_shape
        validate_parameters(req, shape)
        self.assertEqual([x['ChannelName'] for x in req['InputDataConfig']], ['dataset', 'source'])
        self.assertFalse(req['EnableManagedSpotTraining'])
        self.assertEqual(req['AlgorithmSpecification']['ContainerEntrypoint'],
                         ['/bin/bash', '/opt/ml/input/data/source/bootstrap.sh'])

    def test_stop_without_execute_does_not_call_aws(self):
        with patch.object(runner, 'session_for', side_effect=AssertionError('AWS access forbidden')):
            with contextlib.redirect_stdout(io.StringIO()):
                runner.stop_job(self.cfg, 'valid-job', False)

    def test_job_name_no_path_traversal(self):
        for name in ('../../outside', '-bad', 'bad_', 'x' * 64):
            with self.assertRaises(ValueError):
                runner.validate_job_name(name)


class DatasetAndStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = make_dataset(self.root)
        self.dataset = self.root / 'yolo_chewinggum'
        with contextlib.redirect_stdout(io.StringIO()):
            code = converter.main(['--source', str(self.source), '--output', str(self.dataset), '--previews', '0'])
        self.assertEqual(code, 0)
        self.cfg = runner.load_config(HERE / 'config.yaml')
        weights = self.root / 'yolo26n.pt'
        weights.write_bytes(b'fake weights for offline tests only')
        self.cfg['paths'].update(dataset=str(self.dataset), weights=str(weights), work=str(self.root / 'work'))
        self.cfg['aws']['bucket'] = 'example-test-bucket'
        self.args = runner.build_parser().parse_args(['plan'])
        self.tr, _ = runner.settings_for(self.cfg, self.args)

    def tearDown(self):
        self.temp.cleanup()

    def test_validate_synthetic_dataset(self):
        report = common.validate_dataset(self.dataset)
        self.assertEqual(sum(x['images'] for x in report['splits'].values()), 30)
        self.assertEqual(set(report['splits']), {'train', 'val', 'test'})

    def test_explicit_empty_negative_labels_accepted(self):
        report = common.validate_dataset(self.dataset)
        self.assertGreater(report['splits']['train']['negative'], 0)

    def test_missing_label_rejected(self):
        next((self.dataset / 'labels/train').glob('*.txt')).unlink()
        with self.assertRaises(ValueError):
            common.validate_dataset(self.dataset)

    def test_nan_and_outside_boxes_rejected(self):
        label = next((self.dataset / 'labels/train').glob('*.txt'))
        for line in ('0 nan 0.4 0.1 0.1\n', '0 0.01 0.5 0.9 0.9\n', '1 0.5 0.5 0.2 0.2\n'):
            label.write_text(line)
            with self.assertRaises(ValueError):
                common.validate_dataset(self.dataset)

    def test_download_hook_rejected(self):
        path = self.dataset / 'data.yaml'
        data = common.read_yaml(path)
        data['download'] = 'echo unsafe'
        path.write_text(yaml.safe_dump(data))
        with self.assertRaises(ValueError):
            common.validate_dataset(self.dataset)

    def test_duplicate_image_between_splits_rejected(self):
        origin = next((self.dataset / 'images/train').iterdir())
        target = next((self.dataset / 'images/test').iterdir())
        shutil.copy2(origin, target)
        with self.assertRaisesRegex(ValueError, 'leaks'):
            common.validate_dataset(self.dataset)

    def test_moved_dataset_ignores_stale_yaml_root(self):
        path = self.dataset / 'data.yaml'
        data = common.read_yaml(path)
        data['path'] = '/nonexistent/old/machine/dataset'
        path.write_text(yaml.safe_dump(data))
        common.validate_dataset(self.dataset)

    def test_default_stage_excludes_test_and_arbitrary_repo_files(self):
        (self.dataset / '.env').write_text('SECRET=must-not-upload')
        (self.dataset / 'previews').mkdir(exist_ok=True)
        (self.dataset / 'previews/overlay.jpg').write_bytes(b'not a train input')
        report = common.validate_dataset(self.dataset, False)
        job = runner.stage_job(self.cfg, 'offline-stage', self.tr, report)
        paths = [x['path'] for x in json.loads((job / 'input-manifest.json').read_text())]
        self.assertFalse(any('/test/' in x or '.env' in x or 'previews/' in x for x in paths))
        self.assertIn('source/weights/yolo26n.pt', paths)
        self.assertIn('source/train.py', paths)
        self.assertIn('dataset/data.yaml', paths)
        remote = common.validate_dataset(job / 'input/dataset', False)
        self.assertEqual(remote['dataset_fingerprint'], report['dataset_fingerprint'])

    def test_explicit_test_stage_includes_test(self):
        self.tr['evaluate_test'] = True
        report = common.validate_dataset(self.dataset, True)
        job = runner.stage_job(self.cfg, 'offline-test-stage', self.tr, report)
        self.assertTrue((job / 'input/dataset/images/test').is_dir())

    def test_plan_and_submit_without_execute_never_call_aws(self):
        with patch.object(runner, 'session_for', side_effect=AssertionError('AWS forbidden')):
            with patch.object(runner, 'aws_preflight', side_effect=AssertionError('AWS forbidden')):
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.submit_or_plan(self.cfg, self.args)
                    runner.submit_or_plan(self.cfg, runner.build_parser().parse_args(['submit']))
        self.assertFalse((self.root / 'work').exists())

    def test_prepare_reuses_split(self):
        original = common.sha256(self.dataset / 'manifest.csv')
        with patch.object(runner.subprocess, 'run', side_effect=AssertionError('Must not reconvert')):
            with contextlib.redirect_stdout(io.StringIO()):
                runner.prepare_dataset(self.cfg, types.SimpleNamespace(source=None))
        self.assertEqual(common.sha256(self.dataset / 'manifest.csv'), original)

    def test_execute_submits_one_job_with_allowlisted_uploads_only(self):
        from botocore.exceptions import ClientError
        uploaded, created = [], []
        class FakeS3:
            def upload_file(self, path, bucket, key, ExtraArgs):
                uploaded.append((path, bucket, key, ExtraArgs))
        class FakeSM:
            def describe_training_job(self, **kw):
                raise ClientError({'Error': {'Code': 'ValidationException', 'Message': 'Could not find training job'}}, 'DescribeTrainingJob')
            def create_training_job(self, **kw):
                created.append(kw)
                return {'TrainingJobArn': 'arn:aws:sagemaker:eu-central-1:123456789012:training-job/offline-job'}
        class FakeSession:
            def client(self, name, config=None):
                return FakeSM()
        preflight = (FakeSession(), None, FakeS3(), 'arn:aws:iam::123456789012:role/SageMakerExecutionRole', {'Account': '123456789012'})
        args = runner.build_parser().parse_args(['submit', '--execute', '--job-name', 'offline-job'])
        with patch.object(runner, 'aws_preflight', return_value=preflight):
            with contextlib.redirect_stdout(io.StringIO()):
                runner.submit_or_plan(self.cfg, args)
        self.assertEqual(len(created), 1)
        self.assertGreater(len(uploaded), 1)
        self.assertFalse(any('/test/' in item[2] for item in uploaded))
        self.assertTrue(all(item[3] == {'ServerSideEncryption': 'AES256'} for item in uploaded))
        self.assertTrue((self.root / 'work/jobs/offline-job/submission.json').is_file())

    def test_aws_preflight_failure_does_not_stage_or_submit(self):
        args = runner.build_parser().parse_args(['submit', '--execute'])
        with patch.object(runner, 'aws_preflight', side_effect=ValueError('Wrong bucket region')):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'Wrong bucket region'):
                    runner.submit_or_plan(self.cfg, args)
        self.assertFalse((self.root / 'work').exists())

    def test_gpu_training_args_never_use_test_split(self):
        args = training.training_kwargs({'training': self.tr}, Path('/runtime/data.yaml'), Path('/run'))
        self.assertTrue(args['val'])
        self.assertEqual(args['task'], 'detect')
        self.assertNotIn('split', args)
        self.assertEqual(args['device'], '0')
        self.assertFalse(args['exist_ok'])

    def test_model_guard_rejects_wrong_family_and_scale(self):
        for name, scale in [('yolo11n.yaml', 'n'), ('yolo26s.yaml', 's'), ('yolo26.yaml', 's'), ('yolo26.yaml', None)]:
            fake = types.SimpleNamespace(task='detect', model=types.SimpleNamespace(yaml={'yaml_file': name, 'scale': scale}))
            with self.assertRaises(ValueError):
                training.check_model(fake)
        training.check_model(types.SimpleNamespace(task='detect', model=types.SimpleNamespace(yaml={'yaml_file': 'yolo26n.yaml', 'scale': 'n'})))

    def test_training_orchestration_with_fake_model_only(self):
        calls = []
        class FakeYOLO:
            def __init__(self, weights, task):
                self.task = task
                self.model = types.SimpleNamespace(yaml={'yaml_file': 'yolo26n.yaml', 'scale': 'n'})
            def add_callback(self, event, callback):
                calls.append(('callback', event))
            def train(self, **kw):
                calls.append(('train', kw))
                path = Path(kw['project']) / kw['name']
                (path / 'weights').mkdir(parents=True)
                (path / 'weights/best.pt').write_bytes(b'fake trained weights - not a real model')
                (path / 'weights/last.pt').write_bytes(b'fake last weights - not a real model')
                (path / 'results.csv').write_text('epoch,placeholder\n1,0\n')
                self.trainer = types.SimpleNamespace(save_dir=path)
                return types.SimpleNamespace(results_dict={'metrics/mAP50(B)': 0.0})
            def val(self, **kw):
                calls.append(('val', kw))
                return types.SimpleNamespace(results_dict={'metrics/mAP50(B)': 0.0})
        cfg = {'training': self.tr, 'dataset': str(self.dataset), 'weights': self.cfg['paths']['weights'],
               'run_dir': str(self.root / 'local-run'), 'model_dir': str(self.root / 'model'), 'device': 'cpu'}
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = training.execute_training(cfg, FakeYOLO, {'test': 'mocked - no torch/YOLO'})
        self.assertFalse(metrics['test_evaluated'])
        self.assertEqual([x[0] for x in calls], ['callback', 'train'])
        self.assertTrue((self.root / 'model/best.pt').exists())
        with self.assertRaises(ValueError):
            training.execute_training(cfg, FakeYOLO, {})


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, name='best.pt', kind=None):
        path = self.root / 'model.tar.gz'
        with tarfile.open(path, 'w:gz') as tar:
            member = tarfile.TarInfo(name)
            if kind:
                member.type = kind
                member.linkname = '/tmp/escape'
                tar.addfile(member)
            else:
                member.size = 4
                tar.addfile(member, io.BytesIO(b'test'))
        return path

    def test_safe_extraction(self):
        runner.safe_extract(self.archive('weights/best.pt'), self.root / 'output')
        self.assertEqual((self.root / 'output/weights/best.pt').read_bytes(), b'test')

    def test_rejects_traversal_absolute_and_links(self):
        for name, kind in [('../bad', None), ('/tmp/bad', None), ('bad', tarfile.SYMTYPE), ('bad', tarfile.LNKTYPE), ('C:/bad', None)]:
            with self.assertRaises(ValueError):
                runner.safe_extract(self.archive(name, kind), self.root / 'output')
            self.assertFalse((self.root / 'output').exists())

    def test_no_existing_destination_overwrite(self):
        (self.root / 'output').mkdir()
        with self.assertRaises(ValueError):
            runner.safe_extract(self.archive(), self.root / 'output')


if __name__ == '__main__':
    unittest.main()
