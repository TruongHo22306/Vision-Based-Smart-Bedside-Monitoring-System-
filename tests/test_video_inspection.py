"""Synthetic unit fixtures only; never research data or a substitute video dataset."""

import csv
from fractions import Fraction
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import wave

import av

from patient_monitoring.video_inspection import InspectionError, TimelineAudit, inspect_video


def frame(pts, time_base=Fraction(1, 100), corrupt=False):
    return SimpleNamespace(pts=pts, time_base=time_base, is_corrupt=corrupt)


class TimelineTests(unittest.TestCase):
    def test_nonzero_origin_and_unequal_intervals(self):
        audit = TimelineAudit()
        rows = [audit.observe(pts, Fraction(1, 100)) for pts in (500, 504, 511)]
        self.assertEqual([r["presentation_time_seconds"] for r in rows], [5, 5.04, 5.11])
        self.assertEqual([r["relative_time_seconds"] for r in rows], [0, .04, .11])
        self.assertEqual([r["delta_seconds"] for r in rows], [None, .04, .07])
        self.assertEqual(audit.summary()["timestamp_span_seconds"], .11)

    def test_missing_values_preserved_and_previous_valid_retained(self):
        audit = TimelineAudit()
        leading = audit.observe(None, Fraction(1, 100))
        first = audit.observe(200, Fraction(1, 100))
        missing_pts = audit.observe(None, Fraction(1, 100))
        missing_base = audit.observe(205, None)
        invalid_base = audit.observe(206, Fraction(0))
        last = audit.observe(211, Fraction(1, 100))
        for row in (leading, missing_pts, missing_base, invalid_base):
            for key in ("presentation_time_seconds", "relative_time_seconds", "delta_seconds"):
                self.assertIsNone(row[key])
        self.assertEqual(missing_base["pts"], 205)
        self.assertEqual(first["relative_time_seconds"], 0)
        self.assertIsNone(first["delta_seconds"])
        self.assertEqual(last["delta_seconds"], .11)
        self.assertEqual(audit.missing_count, 4)
        self.assertEqual(audit.first_timestamp["frame_index"], 1)

    def test_duplicate_and_decreasing_compare_exact_times_not_raw_pts(self):
        audit = TimelineAudit()
        audit.observe(10, Fraction(1, 3))
        repeated = audit.observe(20, Fraction(1, 6))
        audit.observe(None, Fraction(1, 3))
        decreasing = audit.observe(9, Fraction(1, 3))
        self.assertEqual(repeated["delta_seconds"], 0)
        self.assertAlmostEqual(decreasing["delta_seconds"], -1 / 3)
        self.assertEqual((audit.duplicate_count, audit.decreasing_count), (1, 1))
        self.assertAlmostEqual(audit.summary()["timestamp_span_seconds"], -1 / 3)

    def test_zero_and_negative_origins_are_valid(self):
        for origin in (0, -50):
            audit = TimelineAudit()
            audit.observe(origin, Fraction(1, 100))
            row = audit.observe(origin + 4, Fraction(1, 100))
            self.assertEqual(row["relative_time_seconds"], .04)
            self.assertEqual(audit.first_timestamp["pts"], origin)

    def test_all_missing_has_null_span(self):
        audit = TimelineAudit()
        audit.observe(None, None)
        summary = audit.summary()
        self.assertIsNone(summary["first_valid_timestamp"])
        self.assertIsNone(summary["last_valid_timestamp"])
        self.assertIsNone(summary["timestamp_span_seconds"])


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "synthetic-unit-input.bin"
        self.source.write_bytes(b"unit-test-placeholder-not-video")
        self.output = self.root / "audit"

    def fake_container(self, frames, reported=0, duration=None):
        stream = SimpleNamespace(
            index=3, codec_context=SimpleNamespace(name="unit-codec", width=16, height=8),
            average_rate=None, frames=reported, time_base=Fraction(1, 100), duration=duration,
        )
        container = MagicMock()
        container.__enter__.return_value = container
        container.streams.video = [stream, SimpleNamespace(index=7)]
        container.duration = None
        container.decode.return_value = iter(frames)
        return container, stream

    def run_fake(self, frames, reported=0, duration=None):
        container, stream = self.fake_container(frames, reported, duration)
        with patch("patient_monitoring.video_inspection.av.open", return_value=container):
            result = inspect_video(self.source, self.output)
        container.decode.assert_called_once_with(stream)
        container.__exit__.assert_called_once()
        saved = json.loads((self.output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        with (self.output / "timeline.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return result, rows

    def test_first_stream_selected_csv_missing_and_duration_distinction(self):
        result, rows = self.run_fake([frame(500), frame(None), frame(511)], duration=15)
        self.assertTrue(result["decode_completed"])
        self.assertEqual(result["selected_stream_index"], 3)
        self.assertEqual(result["actual_decoded_count"], 3)
        self.assertEqual(result["timestamp_span_seconds"], .11)
        self.assertEqual(result["stream_duration_seconds"], .15)
        self.assertIsNone(result["container_duration_seconds"])
        self.assertIsNone(result["reported_frame_rate"])
        self.assertIsNone(result["reported_frame_count"])
        self.assertEqual(rows[1]["pts"], "")
        self.assertEqual(rows[1]["presentation_time_seconds"], "")
        self.assertEqual(rows[2]["delta_seconds"], "0.11")
        self.assertEqual(self.source.read_bytes(), b"unit-test-placeholder-not-video")

    def test_decode_failure_keeps_partial_csv_and_metadata(self):
        def broken_decode():
            yield frame(500)
            yield frame(504)
            raise av.InvalidDataError(1094995529, "synthetic decode failure")
        result, rows = self.run_fake(broken_decode())
        self.assertEqual(result["decode_status"], "incomplete")
        self.assertFalse(result["decode_completed"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(result["actual_decoded_count"], 2)
        self.assertIn("synthetic decode failure", result["error"])

    def test_reported_count_shortfall_is_incomplete(self):
        result, _ = self.run_fake([frame(0)], reported=2)
        self.assertEqual(result["decode_status"], "incomplete")
        self.assertFalse(result["decode_completed"])

    def test_empty_decode_is_incomplete(self):
        result, rows = self.run_fake([])
        self.assertEqual(result["decode_status"], "incomplete")
        self.assertEqual(rows, [])

    def test_corrupt_frame_is_incomplete(self):
        result, _ = self.run_fake([frame(0, corrupt=True)])
        self.assertEqual(result["corrupt_frame_count"], 1)
        self.assertEqual(result["decode_status"], "incomplete")

    def test_real_pyav_audio_only_input_has_no_video(self):
        # Tiny temporary WAV exercises real PyAV, not a substitute video dataset.
        audio = self.root / "unit-audio.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b"\x00\x00" * 80)
        result = inspect_video(audio, self.output)
        self.assertEqual(result["decode_status"], "no_video_stream")
        self.assertFalse(result["decode_completed"])
        self.assertIsNone(result["selected_stream_index"])
        self.assertIsNone(result["codec"])

    def test_cli_failure_exits_nonzero_and_keeps_error_report(self):
        result = subprocess.run(
            [sys.executable, "-m", "patient_monitoring", "inspect-video",
             str(self.source), "--output-dir", str(self.output)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Inspection incomplete", result.stderr)
        saved = json.loads((self.output / "metadata.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["decode_completed"])

    def test_real_pyav_invalid_input_reports_failed(self):
        result = inspect_video(self.source, self.output)
        self.assertEqual(result["decode_status"], "failed")
        self.assertEqual(result["actual_decoded_count"], 0)
        self.assertFalse(result["decode_completed"])
        self.assertIsNotNone(result["error"])

    def test_missing_directory_and_empty_inputs_rejected(self):
        empty = self.root / "empty.mp4"
        empty.touch()
        for source in (self.root / "missing.mp4", self.root, empty):
            with self.subTest(source=source):
                with self.assertRaises(InspectionError):
                    inspect_video(source, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_reports_and_input_alias_never_overwritten(self):
        self.output.mkdir()
        existing = self.output / "timeline.csv"
        existing.write_bytes(b"preserve-report")
        with self.assertRaises(InspectionError):
            inspect_video(self.source, self.output)
        self.assertEqual(existing.read_bytes(), b"preserve-report")
        self.assertFalse((self.output / "metadata.json").exists())
        input_alias = self.root / "metadata.json"
        input_alias.write_bytes(b"preserve-input")
        with self.assertRaises(InspectionError):
            inspect_video(input_alias, self.root)
        self.assertEqual(input_alias.read_bytes(), b"preserve-input")


if __name__ == "__main__":
    unittest.main()
