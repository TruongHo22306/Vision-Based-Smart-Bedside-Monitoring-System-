"""NCNN input/metadata/color/consistency contracts; no model downloads."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from patient_monitoring.pose_benchmark import model_identity, runtime_snapshot
from patient_monitoring.pose_comparison import match_people, compare_backends
from patient_monitoring.video_inspection import InspectionError


def directory(root):
    path = root/'trial_ncnn_model'
    path.mkdir()
    (path/'model.ncnn.param').write_text('synthetic param, never loaded')
    (path/'model.ncnn.bin').write_bytes(b'synthetic bin')
    (path/'metadata.yaml').write_text('task: pose\nbatch: 1\nkpt_shape: [17, 3]\nimgsz: [320, 320]\n')
    return path


def person(index, box, x=3.):
    return dict(detection_index=index, box_xyxy=box, box_confidence=.8,
        keypoints=[dict(index=i, name=str(i), x=x, y=4., confidence=.2) for i in range(17)])


class NCNNTests(unittest.TestCase):
    def test_missing_or_ambiguous_artifacts_and_shape_rejected(self):
        with TemporaryDirectory() as temp:
            path = directory(Path(temp))
            self.assertEqual(model_identity(path, 320)['backend'], 'ncnn')
            with self.assertRaises(InspectionError): model_identity(path, 640)
            (path/'extra.param').write_text('ambiguous')
            with self.assertRaises(InspectionError): model_identity(path, 320)
            (path/'extra.param').unlink()
            (path/'model.ncnn.bin').unlink()
            with self.assertRaises(InspectionError): model_identity(path, 320)

    def test_hash_covers_runtime_artifacts(self):
        with TemporaryDirectory() as temp:
            path = directory(Path(temp))
            old = model_identity(path, 320)
            (path/'model.ncnn.bin').write_bytes(b'changed')
            self.assertNotEqual(old['sha256'], model_identity(path,320)['sha256'])

    def test_ncnn_keeps_warmup_and_telemetry_exclusion(self):
        from test_pose_benchmark import BenchmarkTests
        fixture = BenchmarkTests()
        fixture.setUp()
        try:
            fixture.weights = directory(fixture.root)
            report, _, _ = fixture.fixture(imgsz=320)
            self.assertTrue(report['completed'])
            self.assertEqual(report['model_identity']['backend'],'ncnn')
            self.assertEqual(report['statistics']['predict_seconds_sum'],6)
            self.assertEqual(report['statistics']['measured_completed'],3)
        finally:
            fixture.doCleanups()

    def test_ncnn_threads_and_precision_separate_from_torch(self):
        opt = SimpleNamespace(num_threads=4, use_vulkan_compute=False, use_fp16_storage=True)
        backend = SimpleNamespace(format='ncnn', device='cpu', fp16=False, backend=SimpleNamespace(net=SimpleNamespace(opt=opt)))
        model = SimpleNamespace(predictor=SimpleNamespace(model=backend,args=SimpleNamespace(imgsz=320)))
        with patch('patient_monitoring.pose_benchmark.torch_threads',return_value={'intraop':8,'interop':8}):
            snapshot = runtime_snapshot(model, {'imgsz':320})
            self.assertEqual(snapshot['ncnn_options']['num_threads'],4)
            self.assertEqual(snapshot['torch_threads']['intraop'],8)
            self.assertIsNone(snapshot['actual_model_dtype'])
            opt.use_vulkan_compute=True
            with self.assertRaises(InspectionError): runtime_snapshot(model, {})

    def test_matching_reordered_detections_keeps_all_raw_joints(self):
        a = [person(0,[0,0,10,10]),person(1,[30,30,40,40])]
        b = [person(0,[30,30,40,40],6),person(1,[0,0,10,10],6)]
        result = match_people(a,b)
        self.assertEqual([(r['pt_detection_index'],r['ncnn_detection_index']) for r in result['matches']],[(0,1),(1,0)])
        self.assertEqual(result['matches'][0]['mean_joint_distance_pixels'],3)
        self.assertEqual(len(result['matches'][0]['keypoints']),17)
        self.assertEqual(a[0]['keypoints'][0]['confidence'],.2)

    def test_zero_unmatched_and_invalid_boxes(self):
        self.assertEqual(match_people([],[])['matches'],[])
        a=[person(0,[0,0,10,10]),person(1,[None,0,10,10])]
        self.assertEqual(match_people(a,[])['unmatched_pt'],[0,1])
        self.assertEqual(match_people(a,[person(0,[30,30,40,40])])['unmatched_ncnn'],[0])

    def test_comparison_invalid_and_existing_outputs(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            result=compare_backends(root/'v',root/'missing.pt',root/'ncnn',root/'out',frames=0)
            self.assertFalse(result['completed'])
            saved=(root/'out/comparison.json').read_bytes()
            with self.assertRaises(InspectionError): compare_backends(root/'v',root/'pt',root/'ncnn',root/'out')
            self.assertEqual(saved,(root/'out/comparison.json').read_bytes())

    def test_installed_pil_rgb_contract_round_trip(self):
        import numpy as np
        import torch
        from PIL import Image
        from ultralytics.data.loaders import LoadPilAndNumpy
        from ultralytics.engine.predictor import BasePredictor
        picture=Image.new('RGB',(2,2),(231,17,4))
        try:
            bgr=LoadPilAndNumpy._single_check(picture)
            np.testing.assert_array_equal(bgr[0,0],[4,17,231])
            predictor=SimpleNamespace(pre_transform=lambda xs:xs,device=torch.device('cpu'),model=SimpleNamespace(fp16=False))
            tensor=BasePredictor.preprocess(predictor,[bgr])
            torch.testing.assert_close(tensor[0,:,0,0],torch.tensor([231,17,4],dtype=torch.float32)/255)
        finally:
            picture.close()


if __name__ == '__main__':
    unittest.main()
