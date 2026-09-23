"""Synthetic pose-output fixtures only; these tests do not claim model accuracy."""

import copy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from patient_monitoring.pose_preview import (PREDICT_CONFIG, annotate, pose_preview,
    read_samples, serialize_people, visible_edges)
from patient_monitoring.video_inspection import InspectionError


class TensorStub:
    def __init__(self, values):
        self.values = values
    def cpu(self):
        return self
    def tolist(self):
        return copy.deepcopy(self.values)


def prediction(count):
    return SimpleNamespace(
        orig_shape=(16, 32),
        boxes=SimpleNamespace(xyxy=TensorStub([[0., 1., 30., 15.]] * count),
                              conf=TensorStub([.81] * count), cls=TensorStub([0.] * count)),
        keypoints=SimpleNamespace(data=TensorStub([
            [[12.125, 6.75, .05 if i == 0 else .9] for i in range(17)]
            for _ in range(count)])) if count else None,
    )


class PoseDataTests(unittest.TestCase):
    def test_multiple_people_all_keypoints_low_confidence_unchanged(self):
        result = prediction(2)
        people = serialize_people(result)
        self.assertEqual(len(people), 2)
        for person in people:
            self.assertEqual(len(person['keypoints']), 17)
            self.assertEqual(person['box_confidence'], .81)
            self.assertEqual(person['keypoints'][0]['x'], 12.125)
            self.assertEqual(person['keypoints'][0]['y'], 6.75)
            self.assertEqual(person['keypoints'][0]['confidence'], .05)
        saved = copy.deepcopy(people)
        with Image.new('RGB', (32, 16)) as picture:
            annotate(picture, people, .5)
        self.assertEqual(people, saved)

    def test_zero_detections_are_empty_not_missing_or_invented(self):
        self.assertEqual(serialize_people(prediction(0)), [])

    def test_edge_requires_both_endpoints_including_exact_threshold(self):
        points = [dict(x=0, y=0, confidence=.1) for _ in range(17)]
        points[5]['confidence'], points[6]['confidence'] = .5, .499
        self.assertNotIn((5, 6), visible_edges(points, .5))
        points[6]['confidence'] = .5
        self.assertEqual(visible_edges(points, .5), [(5, 6)])

    def test_missing_joints_raise_instead_of_padding(self):
        result = prediction(1)
        result.keypoints.data.values[0].pop()
        with self.assertRaises(InspectionError):
            serialize_people(result)


class PoseCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / 'input'
        self.input.mkdir()
        with Image.new('RGB', (32, 16), 'black') as picture:
            picture.save(self.input / 'referenced.png')
        self.write_csv('referenced.png')
        # A broken, unreferenced image must never be opened/inferred.
        (self.input / 'unreferenced.png').write_bytes(b'not an image')

    def write_csv(self, name):
        with (self.input / 'samples.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_index', 'actual_time_seconds', 'image_filename'])
            writer.writerow([59, '3.9840333333333335', name])

    def test_csv_mapping_and_only_referenced_pngs(self):
        records = read_samples(self.input)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['frame_index'], 59)
        self.assertEqual(records[0]['actual_time_seconds_text'], '3.9840333333333335')
        self.assertEqual(records[0]['width'], 32)

    def test_path_escape_is_rejected(self):
        self.write_csv('../outside.png')
        with self.assertRaises(InspectionError):
            read_samples(self.input)

    def test_zero_detection_run_still_writes_image_record_and_sheet(self):
        # Ultralytics 8.4.152 accepts half=False but does not expose that key
        # again on predictor.args. Reporting must not turn successful inference
        # into a failed run.
        effective_config = {key: value for key, value in PREDICT_CONFIG.items()
                            if key != "half"}
        model = SimpleNamespace(
            predict=lambda **kwargs: [prediction(0)],
            model=SimpleNamespace(parameters=lambda: iter([
                SimpleNamespace(device='cpu', dtype='torch.float32')
            ])),
            predictor=SimpleNamespace(args=SimpleNamespace(**effective_config)),
        )
        weights = self.root / 'synthetic-weights.txt'
        weights.write_text('unit fixture; no actual inference')
        output = self.root / 'output'
        with patch('patient_monitoring.pose_preview.ensure_model', return_value=weights), \
                patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda *a, **k: model)}), \
                patch.dict('os.environ', {}, clear=False):
            run = pose_preview(self.input, output)
        self.assertEqual(run['status'], 'complete', run['error'])
        self.assertEqual(run['processed_image_count'], 1)
        self.assertEqual(run['zero_detection_image_count'], 1)
        self.assertEqual(run['actual_model_dtype'], 'torch.float32')
        self.assertEqual(run['runtime_unreported_config'], ['half'])
        self.assertNotIn('half', run['effective_predict_config'])
        images = json.loads((output / 'poses.json').read_text())['images']
        self.assertEqual(images[0]['people'], [])
        self.assertEqual(images[0]['detection_status'], 'zero_detections')
        self.assertEqual(images[0]['frame_index'], 59)
        self.assertTrue((output / 'pose_contact_sheet.jpg').is_file())
        self.assertTrue((output / images[0]['annotated_image']).is_file())

    def test_existing_output_not_overwritten(self):
        with self.assertRaises(InspectionError):
            pose_preview(self.input, self.input)
        self.assertTrue((self.input / 'samples.csv').exists())


if __name__ == '__main__':
    unittest.main()
