"""Focused saved-prediction review tests; synthetic images are not research evidence."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from patient_monitoring.pose_extraction import quality_keypoint
from patient_monitoring.pose_preview import KEYPOINT_NAMES, sha256_file
from patient_monitoring.pose_review import (
    annotate_review, clipped_marker, gate_edges, review_poses, select_boundary_frames,
)
from patient_monitoring.video_inspection import InspectionError, TimelineAudit


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'synthetic.bin'
        self.source.write_bytes(b'synthetic fixture only')
        self.poses, self.summary_path = self.root / 'poses.jsonl', self.root / 'summary.json'
        self.output = self.root / 'review'
        self.frames, self.records = [], []
        audit = TimelineAudit()
        for index, (pts, count) in enumerate(zip((100, 107, 119, 167), (0, 2, 1, 2))):
            self.frames.append(SimpleNamespace(pts=pts, time_base=Fraction(1, 100),
                width=640, height=360, is_corrupt=False, to_image=lambda: Image.new('RGB', (640, 360))))
            people = []
            for detection in range(count):
                points = [quality_keypoint(dict(index=i, name=name, x=100.+i, y=150.+i,
                           confidence=.9), 640, 360) for i, name in enumerate(KEYPOINT_NAMES)]
                if index != 2:
                    points[15] = quality_keypoint(dict(index=15, name='left_ankle', x=641., y=360., confidence=.8), 640, 360)
                people.append(dict(detection_index=detection, box_xyxy=[90., 100., 300., 350.],
                                   box_confidence=.75, keypoints=points))
            self.records.append(dict(**audit.observe(pts, Fraction(1, 100)),
                source_path=str(self.source.resolve()), stream_index=0, width=640, height=360,
                inference_status='complete', person_count=count, people=people))
        self.summary = dict(source_sha256=sha256_file(self.source), completed=True, status='complete',
            decoded_frame_count=4, processed_frame_count=4, written_record_count=4,
            multiple_detection_frame_count=2, selected_stream_index=0)
        self.save_inputs()

    def save_inputs(self):
        self.poses.write_text(''.join(json.dumps(r) + '\n' for r in self.records), encoding='utf-8')
        self.summary_path.write_text(json.dumps(self.summary), encoding='utf-8')

    def run_review(self):
        container = MagicMock()
        container.__enter__.return_value = container
        stream = SimpleNamespace(index=0)
        container.streams.video = [stream]
        container.decode.return_value = iter(self.frames)
        # Importing Ultralytics is prohibited even indirectly in this utility.
        with patch('patient_monitoring.pose_review.av.open', return_value=container), \
                patch.dict('sys.modules', {'ultralytics': None}):
            result = review_poses(self.source, self.poses, self.summary_path, self.output)
        container.decode.assert_called_once_with(stream)
        return result

    def test_time_selection_is_distinct_and_uses_actual_timestamps(self):
        candidates = list(enumerate([1., 1.01, 1.02, 3., 5., 7., 9., 11.]))
        selected = select_boundary_frames(candidates)
        self.assertEqual([r['frame_index'] for r in selected], [0, 3, 4, 5, 6, 7])
        self.assertEqual(select_boundary_frames([]), [])
        tied = select_boundary_frames([(4, 1.), (5, 1.), (6, 1.)])
        self.assertEqual([r['frame_index'] for r in tied], [4, 5, 6])

    def test_all_multi_frames_and_exact_source_values_preserved(self):
        hashes = [sha256_file(p) for p in (self.source, self.poses, self.summary_path)]
        result = self.run_review()
        self.assertEqual(result['export_status'], 'complete')
        self.assertEqual(result['human_verdict'], 'pending')
        self.assertEqual(result['alignment_checked_frame_count'], 4)
        self.assertEqual(result['multiple_frame_indices'], [1, 3])
        self.assertEqual([r['frame_index'] for r in result['selected_frames']], [1, 3])
        for record in result['selected_frames']:
            original = self.records[record['frame_index']]
            self.assertEqual(record['pts'], original['pts'])
            for person, before in zip(record['people'], original['people']):
                self.assertEqual(person['keypoints'], before['keypoints'])
                self.assertEqual(person['box_xyxy'], before['box_xyxy'])
                self.assertEqual(person['box_confidence'], before['box_confidence'])
                self.assertEqual(person['human_verdict'], 'pending')
            with Image.open(self.output / record['annotated_image']) as image:
                self.assertEqual(image.size, (640, 360))
        for filename in ('multi_detection_sheet.jpg', 'boundary_review_sheet.jpg'):
            with Image.open(self.output / filename) as image:
                image.verify()
        self.assertEqual(hashes, [sha256_file(p) for p in (self.source, self.poses, self.summary_path)])

    def test_gate_edges_and_display_clipping_do_not_change_raw_predictions(self):
        record = self.records[1]
        before = deepcopy(record)
        self.assertNotIn((15, 13), gate_edges(record['people'][0]['keypoints']))
        self.assertIn((13, 11), gate_edges(record['people'][0]['keypoints']))
        self.assertEqual(clipped_marker(record['people'][0]['keypoints'][15], (640, 360)), (639, 359))
        with Image.new('RGB', (640, 360)) as image:
            self.assertEqual(annotate_review(image, record), [])
        self.assertEqual(record, before)

    def test_summary_mismatch_rejected_before_export(self):
        self.summary['multiple_detection_frame_count'] = 6
        self.save_inputs()
        with self.assertRaises(InspectionError):
            review_poses(self.source, self.poses, self.summary_path, self.output)
        self.assertFalse(self.output.exists())

    def test_timestamp_mismatch_marks_incomplete_even_on_unselected_frame(self):
        self.records[0]['pts'] += 1
        self.save_inputs()
        result = self.run_review()
        self.assertEqual(result['export_status'], 'incomplete')
        self.assertIn('alignment mismatch', result['errors'][0])
        self.assertEqual(result['human_verdict'], 'pending')

    def test_short_decode_is_incomplete(self):
        self.frames.pop()
        result = self.run_review()
        self.assertEqual(result['export_status'], 'incomplete')
        self.assertIn('length differ', result['errors'][0])

    def test_existing_output_is_not_replaced(self):
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('unchanged')
        with self.assertRaises(InspectionError):
            review_poses(self.source, self.poses, self.summary_path, self.output)
        self.assertEqual(sentinel.read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main()
