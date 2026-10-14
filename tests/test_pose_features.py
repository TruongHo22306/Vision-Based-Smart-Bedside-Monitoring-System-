"""Geometry/timing checks on synthetic poses, never labels or model inference."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from patient_monitoring.pose_features import FeatureExtractor, extract_features, geometry
from patient_monitoring.pose_preview import KEYPOINT_NAMES
from patient_monitoring.video_inspection import InspectionError


def row(index=0, pts=0, shift=0, status=None):
    coordinates = {5: (40, 30), 6: (60, 30), 11: (40, 50), 12: (60, 50),
                   13: (40, 70), 14: (60, 70), 15: (40, 90), 16: (80, 70)}
    points = []
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = coordinates.get(i, (50, 10))
        points.append(dict(index=i, name=name, x=x + shift, y=y, confidence=.9,
                           finite_components=dict(x=True, y=True, confidence=True),
                           finite_values=True, confidence_pass=True, at_or_outside_boundary=False,
                           passes_quality_gate=True))
    person = dict(detection_index=0, box_xyxy=[20 + shift, 10, 90 + shift, 95],
                  box_confidence=.8, keypoints=points)
    return dict(schema_version=1, source_path="synthetic.mp4", stream_index=0, width=200, height=100,
                frame_index=index, pts=pts, time_base="1/1000", presentation_time_seconds=pts / 1000,
                relative_time_seconds=pts / 1000, delta_seconds=None if index == 0 else .001,
                timestamp_valid=True, status=status or ("seeded" if index == 0 else "matched"),
                selected_person=person, selected_source_detection_index=0,
                reference_frame_index=None if index == 0 else index - 1,
                association_ended=False, reason="synthetic fixture")


def point(r, index):
    return r["selected_person"]["keypoints"][index]


class GeometryTests(unittest.TestCase):
    def test_known_geometry_units_and_preservation(self):
        r = row(); before = deepcopy(r)
        result = FeatureExtractor("synthetic").step(r)
        f = result["features"]
        self.assertEqual(f["hip_midpoint_px"]["value"], [50, 50])
        self.assertEqual(f["shoulder_midpoint_px"]["value"], [50, 30])
        self.assertEqual(f["torso_scale_px"]["value"], 20)
        self.assertEqual(f["hip_image_xy"]["value"], [.25, .5])
        self.assertEqual(f["box_size_px"]["value"], [70, 85])
        self.assertEqual(f["box_size_image"]["value"], [.35, .85])
        self.assertEqual(f["left_knee_angle_deg"]["value"], 180)
        self.assertEqual(f["right_knee_angle_deg"]["value"], 90)
        self.assertEqual(f["torso_inclination_deg"]["value"], 0)
        self.assertEqual(result["joints"][11]["body_xy"]["value"], [-.5, 0])
        self.assertEqual(r, before)
        for key in ("frame_index", "pts", "time_base", "presentation_time_seconds", "relative_time_seconds", "delta_seconds"):
            self.assertEqual(result[key], r[key])

    def test_translation_and_isotropic_scale_invariance(self):
        r = row(); changed = deepcopy(r)
        for p in changed["selected_person"]["keypoints"]:
            p["x"], p["y"] = p["x"] * 2 + 10, p["y"] * 2 + 15
        changed.update(width=500, height=250)
        a, b = geometry(r), geometry(changed)
        self.assertEqual(b[0]["torso_scale_px"]["value"], 40)
        for x, y in zip(a[1], b[1]):
            self.assertEqual(x["body_xy"], y["body_xy"])
        self.assertEqual(a[0]["left_knee_angle_deg"], b[0]["left_knee_angle_deg"])

    def test_torso_angle_from_upward_image_vertical(self):
        for coordinates, expected in [((70, 50), 90), ((30, 50), 90), ((50, 70), 180)]:
            r = row()
            for i in (5, 6):
                point(r, i)["x"], point(r, i)["y"] = coordinates
            self.assertEqual(geometry(r)[0]["torso_inclination_deg"]["value"], expected)

    def test_missing_face_does_not_reject_body(self):
        r = row(); point(r, 0)["passes_quality_gate"] = False
        f, joints, accepted = geometry(r)
        self.assertTrue(accepted)
        self.assertTrue(all(f[k]["valid"] for k in f))
        self.assertIsNone(joints[0]["body_xy"]["value"])

    def test_missing_ankle_affects_only_its_knee(self):
        r = row(); r["selected_person"]["keypoints"].pop(15)
        f = geometry(r)[0]
        self.assertFalse(f["left_knee_angle_deg"]["valid"])
        self.assertTrue(f["right_knee_angle_deg"]["valid"])
        self.assertTrue(f["torso_scale_px"]["valid"])

    def test_saved_gate_not_confidence_controls_eligibility(self):
        r = row(); p = point(r, 11)
        p.update(confidence=.999, passes_quality_gate=False, at_or_outside_boundary=True)
        f, joints, _ = geometry(r)
        self.assertFalse(f["hip_midpoint_px"]["valid"])
        self.assertEqual(joints[11]["source_quality"]["at_or_outside_boundary"], True)
        self.assertTrue(f["box_size_image"]["valid"])

    def test_null_values_even_if_mask_is_inconsistent(self):
        r = row(); point(r, 5)["x"] = None
        f, joints, _ = geometry(r)
        self.assertFalse(f["torso_scale_px"]["valid"])
        self.assertEqual(joints[5]["pixel_xy"]["reason"], "invalid_values_despite_saved_gate")
        self.assertTrue(f["hip_image_xy"]["valid"])

    def test_degenerate_torso_and_knee_are_null(self):
        r = row()
        point(r, 5).update(x=40, y=50); point(r, 6).update(x=60, y=50)
        point(r, 13).update(x=40, y=50)
        f, joints, _ = geometry(r)
        self.assertIsNone(f["torso_scale_px"]["value"])
        self.assertEqual(f["torso_scale_px"]["reason"], "degenerate_torso_scale")
        self.assertIsNone(f["left_knee_angle_deg"]["value"])
        self.assertTrue(f["hip_image_xy"]["valid"])
        self.assertTrue(all(j["body_xy"]["value"] is None for j in joints))

    def test_bad_box_and_dimensions_do_not_zero_fill(self):
        r = row(); r["selected_person"]["box_xyxy"] = [20, 10, 20, 80]
        self.assertIsNone(geometry(r)[0]["box_size_px"]["value"])
        r["width"] = 0
        self.assertEqual(geometry(r)[0]["hip_image_xy"]["reason"], "invalid_image_dimensions")

    def test_reordered_keypoints_and_duplicate_identity(self):
        r = row(); r["selected_person"]["keypoints"].reverse()
        self.assertEqual(geometry(r)[0]["torso_scale_px"]["value"], 20)
        r["selected_person"]["keypoints"].append(deepcopy(r["selected_person"]["keypoints"][5]))  # index11
        self.assertFalse(geometry(r)[0]["hip_midpoint_px"]["valid"])


class DifferenceTests(unittest.TestCase):
    def test_irregular_pts_ignores_nominal_delta_and_detection_index(self):
        engine = FeatureExtractor("synthetic")
        engine.step(row())
        r = row(1, 100, 10); r["selected_source_detection_index"] = 9
        r["selected_person"]["detection_index"] = 9
        r["selected_person"]["box_xyxy"][2] += 10
        out = engine.step(r)
        self.assertEqual(out["delta_seconds"], .001)  # Original field retained; NOT denominator.
        self.assertEqual(out["temporal_pair"]["delta_seconds"], .1)
        self.assertAlmostEqual(out["features"]["hip_image_velocity_per_second"]["value"][0], .5)
        self.assertAlmostEqual(out["features"]["box_size_image_velocity_per_second"]["value"][0], .5)
        out = engine.step(row(2, 300, 30))
        self.assertAlmostEqual(out["features"]["hip_image_velocity_per_second"]["value"][0], .5)

    def test_unresolved_rows_block_resume_difference(self):
        engine = FeatureExtractor("synthetic"); engine.step(row())
        r = row(1, 100, status="ambiguous"); r["selected_person"] = None
        out = engine.step(r)
        self.assertTrue(all(f["value"] is None for f in out["features"].values()))
        out = engine.step(row(2, 200))
        self.assertEqual(out["temporal_pair"]["reason"], "unresolved_target_barrier")
        self.assertTrue(engine.step(row(3, 300))["temporal_pair"]["valid"])

    def test_one_bad_hip_frame_blocks_hip_difference_both_sides(self):
        engine = FeatureExtractor("synthetic"); engine.step(row())
        r = row(1, 100); point(r, 11)["passes_quality_gate"] = False
        a = engine.step(r); b = engine.step(row(2, 200))
        for out in (a, b):
            self.assertFalse(out["features"]["hip_image_velocity_per_second"]["valid"])
            self.assertTrue(out["features"]["box_size_image_velocity_per_second"]["valid"])
        self.assertTrue(engine.step(row(3, 300))["features"]["hip_image_velocity_per_second"]["valid"])

    def test_gap_equality_allowed_and_exceeded_gap_breaks(self):
        engine = FeatureExtractor("synthetic", .5); engine.step(row())
        self.assertTrue(engine.step(row(1, 500))["temporal_pair"]["valid"])
        out = engine.step(row(2, 1001))
        self.assertEqual(out["temporal_pair"]["reason"], "maximum_time_gap_exceeded")
        self.assertIsNone(out["features"]["box_size_image_velocity_per_second"]["value"])
        self.assertTrue(engine.step(row(3, 1100))["temporal_pair"]["valid"])

    def test_skipped_frame_no_bridge(self):
        engine = FeatureExtractor("synthetic"); engine.step(row())
        self.assertEqual(engine.step(row(2, 100))["temporal_pair"]["reason"], "nonconsecutive_frame_index")

    def test_track_break_seed_reference_and_context(self):
        for update in (dict(status="seeded"), dict(track_break=True), dict(track_id="new"),
                       dict(reference_frame_index=99), dict(source_path="other.mp4"), dict(width=201)):
            engine = FeatureExtractor("synthetic"); engine.step(row())
            r = row(1, 100); r.update(update)
            self.assertFalse(engine.step(r)["temporal_pair"]["valid"], update)

    def test_missing_duplicate_decreasing_and_inconsistent_timestamps(self):
        for change in (dict(pts=None), dict(time_base=None), dict(pts=100, presentation_time_seconds=.1),
                       dict(pts=50, presentation_time_seconds=.05), dict(presentation_time_seconds=88),
                       dict(timestamp_valid=False)):
            engine = FeatureExtractor("synthetic"); engine.step(row(0, 100))
            r = row(1, 200); r.update(change)
            self.assertFalse(engine.step(r)["timestamp_valid"], change)
            out = engine.step(row(2, 300))
            self.assertFalse(out["temporal_pair"]["valid"], change)

    def test_timebase_changes_use_rational_time(self):
        engine = FeatureExtractor("synthetic"); engine.step(row())
        r = row(1, 100, 10); r.update(pts=3, time_base="1/30")
        self.assertEqual(engine.step(r)["temporal_pair"]["delta_seconds"], .1)

    def test_invalid_configuration_and_conflicting_id(self):
        for gap in (0, -1, float("nan"), float("inf"), None):
            with self.assertRaises(InspectionError):
                FeatureExtractor("synthetic", gap)
        r = row(); r["video_id"] = "other"
        with self.assertRaises(InspectionError):
            FeatureExtractor("synthetic").step(r)


class FileTests(unittest.TestCase):
    def test_stream_preserves_all_rows_masks_and_summaries_without_model(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "target_poses.jsonl"
            rows = [row(), row(1, 100), row(2, 300)]
            rows[1].update(status="no_detections", selected_person=None, selected_source_detection_index=None)
            raw = "".join(json.dumps(r) + "\n" for r in rows); path.write_text(raw)
            with patch.dict("sys.modules", {"ultralytics": None, "torch": None}):
                summary = extract_features(path, Path(tmp) / "out", "synthetic")
            self.assertTrue(summary["completed"])
            output = [json.loads(x) for x in (Path(tmp) / "out/features.jsonl").read_text().splitlines()]
            self.assertEqual(len(output), 3)
            self.assertEqual(summary["feature_valid_counts"]["hip_midpoint_px"], 2)
            self.assertEqual(summary["feature_valid_counts"]["hip_image_velocity_per_second"], 0)
            for src, result in zip(rows, output):
                for k in ("frame_index", "pts", "time_base", "presentation_time_seconds", "relative_time_seconds", "delta_seconds"):
                    self.assertEqual(src[k], result[k])
            self.assertEqual(output[0]["joints"][11]["source_quality"]["passes_quality_gate"], True)
            self.assertEqual(path.read_text(), raw)
            with self.assertRaises(InspectionError):
                extract_features(path, Path(tmp) / "out", "synthetic")

    def test_partial_failure_and_empty_input_are_incomplete(self):
        for text in (json.dumps(row()) + "\nnot-json\n", ""):
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / "target_poses.jsonl"; path.write_text(text)
                summary = extract_features(path, Path(tmp) / "out", "synthetic")
                self.assertFalse(summary["completed"])
                self.assertTrue(summary["errors"])
                provenance = json.loads((Path(tmp) / "out/provenance.json").read_text())
                self.assertEqual(provenance["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
