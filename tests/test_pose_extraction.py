"""Synthetic inference/decoder fixtures, never research or pose-accuracy evidence."""

import contextlib
from fractions import Fraction
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import av
from PIL import Image

from patient_monitoring.pose_extraction import (
    PREDICT_CONFIG, extract_poses, quality_keypoint, quality_people,
)
from patient_monitoring.video_inspection import InspectionError, TimelineAudit
from test_pose_preview import prediction


class QualityGateTests(unittest.TestCase):
    def test_independent_flags_and_exact_boundaries(self):
        # finite, confidence pass, at/outside boundary, gate pass
        cases = [
            ((1., 1., .5), (True, True, False, True)),
            ((1., 1., .499), (True, False, False, False)),
            ((0., 1., .9), (True, True, True, False)),
            ((32., 1., .9), (True, True, True, False)),
            ((1., 0., .9), (True, True, True, False)),
            ((1., 16., .9), (True, True, True, False)),
            ((-2., 1., .9), (True, True, True, False)),
            ((33., 17., .1), (True, False, True, False)),
            ((float('nan'), 1., .9), (False, True, False, False)),
            ((1., float('inf'), .9), (False, True, False, False)),
            ((1., 1., float('inf')), (False, False, False, False)),
            ((0., 1., float('nan')), (False, False, True, False)),
        ]
        flags = ('finite_values', 'confidence_pass', 'at_or_outside_boundary',
                 'passes_quality_gate')
        for values, expected in cases:
            with self.subTest(values=values):
                raw = dict(zip(('x', 'y', 'confidence'), values))
                result = quality_keypoint(raw, 32, 16)
                self.assertEqual(tuple(result[k] for k in flags), expected)
                for key in ('x', 'y', 'confidence'):
                    if result['finite_components'][key]:
                        self.assertEqual(result[key], raw[key])
                    else:
                        self.assertIsNone(result[key])
                json.dumps(result, allow_nan=False)

    def test_people_and_low_face_points_not_filtered(self):
        result = prediction(2)
        result.keypoints.data.values[0][0] = [float('nan'), 6.75, .05]
        result.boxes.xyxy.values[0][0] = float('inf')
        result.boxes.conf.values[1] = float('nan')
        people = quality_people(result, 32, 16)
        self.assertEqual([p['detection_index'] for p in people], [0, 1])
        self.assertEqual([len(p['keypoints']) for p in people], [17, 17])
        self.assertIsNone(people[0]['keypoints'][0]['x'])
        self.assertEqual(people[0]['keypoints'][0]['confidence'], .05)
        self.assertTrue(people[0]['keypoints'][5]['passes_quality_gate'])
        self.assertIsNone(people[0]['box_xyxy'][0])
        self.assertFalse(people[0]['box_finite_values'][0])
        self.assertIsNone(people[1]['box_confidence'])
        json.dumps(people, allow_nan=False)
        self.assertEqual(quality_people(prediction(0), 32, 16), [])


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'unit-input.bin'
        self.source.write_bytes(b'synthetic test placeholder')
        self.output = self.root / 'new-output'

    @staticmethod
    def frame(pts, base=Fraction(1, 100), corrupt=False):
        return SimpleNamespace(
            pts=pts, time_base=base, width=32, height=16, is_corrupt=corrupt,
            to_image=lambda: Image.new('RGB', (32, 16)),
        )

    def run_fixture(self, frames, results, reported=0):
        stream = SimpleNamespace(index=3, frames=reported, time_base=Fraction(1, 100),
                                 codec_context=SimpleNamespace(name='unit-codec'))
        container = MagicMock()
        container.__enter__.return_value = container
        container.streams.video = [stream]
        container.decode.return_value = iter(frames)
        model = SimpleNamespace(
            predict=Mock(side_effect=results),
            model=SimpleNamespace(parameters=lambda: iter([
                SimpleNamespace(device='cpu', dtype='torch.float32')
            ])),
            predictor=SimpleNamespace(args=SimpleNamespace(
                **{k: v for k, v in PREDICT_CONFIG.items() if k != 'half'}
            )),
        )
        with patch('patient_monitoring.pose_extraction.av.open', return_value=container), \
                patch('patient_monitoring.pose_extraction.load_verified_model', return_value=model):
            summary = extract_poses(self.source, self.output)
        container.decode.assert_called_once_with(stream)
        rows = [json.loads(line) for line in
                (self.output / 'poses.jsonl').read_text(encoding='utf-8').splitlines()]
        saved = json.loads((self.output / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, summary)
        return summary, rows, model

    def test_every_frame_matches_m1_timeline_and_zero_multiple_preserved(self):
        frames = [self.frame(p) for p in (500, None, 504, 504, 502, 511)]
        summary, rows, model = self.run_fixture(frames, [[prediction(n)] for n in (0, 2, 1, 0, 1, 1)])
        audit = TimelineAudit()
        for frame, row in zip(frames, rows):
            for key, value in audit.observe(frame.pts, frame.time_base).items():
                self.assertEqual(row[key], value)
        self.assertEqual([r['person_count'] for r in rows], [0, 2, 1, 0, 1, 1])
        self.assertEqual(rows[0]['people'], [])
        self.assertEqual(rows[0]['detection_status'], 'zero_detections')
        self.assertEqual([p['detection_index'] for p in rows[1]['people']], [0, 1])
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['decoded_frame_count'], 6)
        self.assertEqual(summary['processed_frame_count'], 6)
        self.assertEqual(summary['written_record_count'], 6)
        self.assertEqual(summary['zero_detection_frame_count'], 2)
        self.assertEqual(summary['multiple_detection_frame_count'], 1)
        self.assertEqual(summary['person_detection_count'], 5)
        self.assertEqual(summary['per_joint'][0]['low_confidence_count'], 5)
        self.assertEqual(summary['per_joint'][0]['observation_count'], 5)
        self.assertEqual(summary['per_joint'][5]['passes_quality_gate_count'], 5)
        for key in ('missing_timestamp_count', 'duplicate_timestamp_count',
                    'decreasing_timestamp_count'):
            self.assertEqual(summary['timestamp_audit'][key], 1)
        self.assertEqual(model.predict.call_count, 6)
        for call in model.predict.call_args_list:
            self.assertEqual({k: call.kwargs[k] for k in PREDICT_CONFIG}, PREDICT_CONFIG)

    def test_stream_writes_before_requesting_next_decoded_frame(self):
        def decode():
            for index in range(3):
                if index:
                    with (self.output / 'poses.jsonl').open(encoding='utf-8') as stream:
                        self.assertEqual(sum(1 for _ in stream), index)
                yield self.frame(500 + index * 7)
        summary, rows, _ = self.run_fixture(decode(), [[prediction(0)]] * 3)
        self.assertTrue(summary['completed'])
        self.assertEqual(len(rows), 3)

    def test_inference_failure_is_not_zero_detection_and_keeps_failed_frame(self):
        summary, rows, _ = self.run_fixture(
            [self.frame(0), self.frame(7), self.frame(14)],
            [[prediction(1)], RuntimeError('unit inference failure')],
        )
        self.assertFalse(summary['completed'])
        self.assertEqual(summary['status'], 'incomplete')
        self.assertEqual(summary['decoded_frame_count'], 2)
        self.assertEqual(summary['processed_frame_count'], 1)
        self.assertEqual(summary['written_record_count'], 2)
        self.assertEqual(summary['zero_detection_frame_count'], 0)
        self.assertEqual(rows[1]['inference_status'], 'failed')
        self.assertIsNone(rows[1]['people'])
        self.assertIsNone(rows[1]['person_count'])
        self.assertEqual(summary['errors'][0]['stage'], 'inference')

    def test_decode_error_keeps_partial_run(self):
        def decode():
            yield self.frame(0)
            raise av.InvalidDataError(1094995529, 'unit decode error')
        summary, rows, _ = self.run_fixture(decode(), [[prediction(0)]])
        self.assertEqual(summary['status'], 'incomplete')
        self.assertFalse(summary['decode_reached_eof'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary['errors'][0]['stage'], 'decode')

    def test_corrupt_and_reported_shortfall_and_empty_are_incomplete(self):
        for i, (frames, reported) in enumerate((
            ([self.frame(0, corrupt=True)], 1), ([self.frame(0)], 2), ([], 0),
        )):
            with self.subTest(case=i):
                self.output = self.root / f'output{i}'
                summary, _, _ = self.run_fixture(frames, [[prediction(0)]], reported)
                self.assertEqual(summary['status'], 'incomplete')
                self.assertFalse(summary['completed'])

    def test_model_block_keeps_incomplete_reports_and_cli_fails(self):
        from patient_monitoring.__main__ import main
        with patch('patient_monitoring.pose_extraction.load_verified_model',
                   side_effect=InspectionError('unit model unavailable')), \
                patch('sys.argv', ['patient_monitoring', 'extract-poses', str(self.source),
                                   '--output-dir', str(self.output)]), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_context:
                main()
        self.assertEqual(exit_context.exception.code, 1)
        run = json.loads((self.output / 'run_info.json').read_text())
        self.assertEqual(run['status'], 'incomplete')
        self.assertEqual(run['processed_frame_count'], 0)
        self.assertEqual(run['errors'][0]['stage'], 'model_load')

    def test_input_existing_output_are_untouched(self):
        self.output.mkdir()
        prior = self.output / 'poses.jsonl'
        prior.write_text('keep this report')
        with self.assertRaises(InspectionError):
            extract_poses(self.source, self.output)
        self.assertEqual(prior.read_text(), 'keep this report')
        self.assertEqual(self.source.read_bytes(), b'synthetic test placeholder')


if __name__ == '__main__':
    unittest.main()
