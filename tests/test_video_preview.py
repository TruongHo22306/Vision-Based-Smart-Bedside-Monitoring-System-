"""Focused synthetic checks; generated fixtures are temporary and not research data."""

from contextlib import closing
import csv
from fractions import Fraction as F
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from patient_monitoring.video_inspection import COLUMNS, InspectionError, TimelineAudit
from patient_monitoring.video_preview import interval_summary, preview_video, select_samples, timeline_records


class SelectionTests(unittest.TestCase):
    def test_irregular_nonzero_timeline_uses_nearest_not_frame_spacing(self):
        records = [(0, F(5)), (1, F('5.08')), (2, F('5.91')), (3, F('7.4')), (4, F(9))]
        rows = select_samples(iter(records), 0, 4, F(5), F(9), 4)
        self.assertEqual([r['frame_index'] for r in rows], [0, 2, 3, 4])
        self.assertEqual([r['actual_time_seconds'] for r in rows], [0, .91, 2.4, 4])
        self.assertAlmostEqual(rows[1]['target_time_seconds'], 4 / 3)

    def test_short_stream_keeps_both_endpoints(self):
        rows = select_samples(iter([(3, F(7)), (9, F(12))]), 3, 9, F(7), F(12), 2)
        self.assertEqual([r['frame_index'] for r in rows], [3, 9])
        self.assertEqual([r['actual_time_seconds'] for r in rows], [0, 5])

    def test_single_valid_frame(self):
        rows = select_samples(iter([(7, F(-2))]), 7, 7, F(-2), F(-2), 1)
        self.assertEqual(rows[0]['frame_index'], 7)
        self.assertEqual(rows[0]['actual_time_seconds'], 0)

    def test_ties_preserve_first_and_last_distinct_frames(self):
        rows = select_samples(iter([(2, F(10)), (4, F(10)), (8, F(10))]), 2, 8, F(10), F(10), 3)
        self.assertEqual([r['frame_index'] for r in rows], [2, 2, 8])
        self.assertEqual(len({r['image_filename'] for r in rows}), 2)

    def test_nearest_tie_uses_lower_frame_index(self):
        rows = select_samples(iter(enumerate(map(F, [0, 2, 4, 6]))), 0, 3, F(0), F(6), 3)
        self.assertEqual([r['frame_index'] for r in rows], [0, 1, 3])


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'fixture.bin'
        self.source.write_bytes(b'synthetic test only')
        self.output = self.root / 'preview'

    def make_audit(self, times):
        def fake_inspect(source, directory):
            directory.mkdir()
            audit = TimelineAudit()
            with (directory / 'timeline.csv').open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=COLUMNS)
                writer.writeheader()
                for time in times:
                    writer.writerow(audit.observe(time, F(1, 100)))
            return dict(**audit.summary(), selected_stream_index=0,
                        decode_completed=True, decode_status='complete', error=None, warnings=[])
        return fake_inspect

    def test_short_preview_reports_fewer_frames_and_disk_intervals(self):
        with patch('patient_monitoring.video_preview.inspect_video', side_effect=self.make_audit([700, None, 750])), \
                patch('patient_monitoring.video_preview.export_images', return_value=2):
            result = preview_video(self.source, self.output, 8)
        self.assertEqual(result['preview_status'], 'complete')
        self.assertEqual(result['distinct_sample_count'], 2)
        self.assertEqual(result['target_count'], 2)
        self.assertEqual(result['missing_timestamp_count'], 1)
        self.assertEqual(result['interval_count'], 1)
        self.assertEqual(result['median_interval_seconds'], .5)
        self.assertTrue(any('Only 2 distinct' in w for w in result['warnings']))
        saved = json.loads((self.output / 'timing_summary.json').read_text())
        self.assertEqual(saved, result)

    def test_missing_input_and_existing_output_are_protected(self):
        with self.assertRaises(InspectionError):
            preview_video(self.root / 'missing.mp4', self.output)
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        report = self.output / 'metadata.json'
        report.write_bytes(b'keep M1.1')
        with self.assertRaises(InspectionError):
            preview_video(self.source, self.output)
        self.assertEqual(report.read_bytes(), b'keep M1.1')

    def test_no_valid_times_fails_clearly(self):
        with patch('patient_monitoring.video_preview.inspect_video', side_effect=self.make_audit([None, None])):
            result = preview_video(self.source, self.output)
        self.assertEqual(result['preview_status'], 'failed')
        self.assertIn('No valid presentation timestamps', result['error'])
        self.assertEqual(result['interval_count'], 0)
        self.assertIsNone(result['median_interval_seconds'])

    def test_decreasing_timing_is_reported_without_misleading_preview(self):
        with patch('patient_monitoring.video_preview.inspect_video', side_effect=self.make_audit([100, 100, 90])):
            result = preview_video(self.source, self.output)
        self.assertEqual(result['preview_status'], 'failed')
        self.assertEqual(result['duplicate_timestamp_count'], 1)
        self.assertEqual(result['decreasing_timestamp_count'], 1)
        self.assertEqual(result['interval_count'], 2)
        self.assertEqual(result['min_interval_seconds'], -.1)
        self.assertEqual(result['median_interval_seconds'], -.05)
        self.assertEqual(result['max_interval_seconds'], 0)

    def test_interval_median_even_odd_and_empty(self):
        with closing(sqlite3.connect(self.root / 'unit.sqlite')) as database:
            database.execute('CREATE TABLE intervals(seconds REAL NOT NULL)')
            self.assertIsNone(interval_summary(database)['median_interval_seconds'])
            for value in (.04, .11, .08):
                database.execute('INSERT INTO intervals VALUES (?)', (value,))
            self.assertEqual(interval_summary(database)['median_interval_seconds'], .08)
            database.execute('INSERT INTO intervals VALUES (0.06)')
            self.assertAlmostEqual(interval_summary(database)['median_interval_seconds'], .07)


if __name__ == '__main__':
    unittest.main()
