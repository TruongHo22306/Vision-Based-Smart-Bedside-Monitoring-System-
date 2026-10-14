import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from patient_monitoring.action_baseline import Baseline, Config, FEATURES, decide, run_baseline


def frame(i, t, *, torso=5, knee=170, aspect=.5, hip=0, box=0):
    def f(v):
        return dict(value=v, valid=True, reason=None)
    return dict(schema_version=1, feature_version="per_frame_v001", video_id="arbitrary",
                frame_index=i, pts=round(t * 10000), time_base="1/10000", presentation_time_seconds=t,
                relative_time_seconds=t, delta_seconds=None, source_path="irrelevant", stream_index=0,
                width=1280, height=720, timestamp_valid=True, target_status="seeded" if i == 0 else "matched",
                selected_source_detection_index=0,
                temporal_pair=dict(valid=i > 0, previous_frame_index=i - 1, reason=None),
                features=dict(zip(FEATURES, [f(torso), f(knee), f(knee), f([aspect * 100, 100]),
                                            f([hip, 0]), f([box, 0])])) )


def sequence(**kwargs):
    return [frame(i, i / 10, **kwargs) for i in range(21)]


class WindowTests(unittest.TestCase):
    def test_time_coverage_not_frame_count(self):
        e = Baseline()
        results = [e.step(frame(i, t)) for i, t in enumerate([0, .01, .02, .03, .2, .4, .6, .8])]
        self.assertTrue(all(x["predicted_label"] == "unknown" for x in results[:-1]))
        self.assertEqual(results[-1]["predicted_label"], "standing")
        self.assertAlmostEqual(results[-1]["support"][FEATURES[0]]["coverage_fraction"], .8)

    def test_exact_window_clipping_and_weighting(self):
        e = Baseline(replace(Config(), max_gap_seconds=1))
        rows = [frame(i, t, torso=a) for i, (t, a) in enumerate([(0, 0), (.3, 10), (.7, 20), (1.2, 30)])]
        r = [e.step(x) for x in rows][-1]
        self.assertAlmostEqual(r["support"][FEATURES[0]]["value"], .1 * 10 + .4 * 20 + .5 * 30)
        self.assertAlmostEqual(r["support"][FEATURES[0]]["covered_seconds"], 1)

    def test_prefix_causality_and_no_id_label_leakage(self):
        a, b = Baseline(), Baseline()
        rows = sequence()
        for original in rows:
            changed = copy.deepcopy(original)
            changed.update(video_id="lying_moving_sitting", source_path="moving.mp4", annotation="moving")
            ra, rb = a.step(original), b.step(changed)
            self.assertEqual((ra["predicted_label"], ra["support"]), (rb["predicted_label"], rb["support"]))
        first = [Baseline().step(rows[0])]
        self.assertEqual(first[0]["predicted_label"], "unknown")

    def test_unresolved_barrier_and_recovery(self):
        e = Baseline()
        rows = sequence()
        rows[10].update(target_status="unresolved", selected_source_detection_index=None)
        r = [e.step(x) for x in rows]
        self.assertEqual(r[10]["reasons"], ["unresolved_target"])
        self.assertTrue(all(x["predicted_label"] == "unknown" for x in r[11:19]))
        self.assertEqual(r[19]["predicted_label"], "standing")

    def test_gap_and_saved_track_breaks(self):
        mutations = [dict(target_status="seeded"), dict(track_break=True), dict(track_id=2),
                     dict(video_id="next"), dict(source_path="next"), dict(frame_index=40),
                     dict(temporal_pair=dict(valid=False, reason="explicit_track_break"))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                e = Baseline()
                for r in sequence()[:10]:
                    e.step(r)
                row = frame(10, 1)
                row.update(mutation)
                out = e.step(row)
                self.assertEqual(out["predicted_label"], "unknown")
                self.assertIsNotNone(out["window"]["reset_reason"])
        e = Baseline()
        e.step(frame(0, 0))
        self.assertEqual(e.step(frame(1, .5001))["window"]["reset_reason"], "maximum_time_gap_exceeded")
        e = Baseline()
        e.step(frame(0, 0))
        self.assertIsNone(e.step(frame(1, .5))["window"]["reset_reason"])

    def test_duplicate_decreasing_missing_inconsistent_timestamps(self):
        for mutation in [dict(pts=0, presentation_time_seconds=0), dict(pts=-1), dict(pts=None),
                         dict(time_base="0"), dict(presentation_time_seconds=999), dict(timestamp_valid=False)]:
            e = Baseline()
            e.step(frame(0, 0))
            row = frame(1, .1)
            row.update(mutation)
            self.assertEqual(e.step(row)["predicted_label"], "unknown")

    def test_missing_values_not_zero_filled(self):
        e = Baseline()
        rows = sequence()
        for r in rows:
            r["features"][FEATURES[1]] = dict(value=None, valid=False, reason="occluded")
        out = [e.step(r) for r in rows][-1]
        self.assertEqual(out["predicted_label"], "unknown")
        self.assertIsNone(out["support"][FEATURES[1]]["value"])
        self.assertEqual(out["input_features"][FEATURES[1]], rows[-1]["features"][FEATURES[1]])
        self.assertEqual(out["support"][FEATURES[1]]["covered_seconds"], 0)

    def test_missing_static_endpoint_excludes_both_intervals(self):
        e = Baseline()
        rows = sequence()[:11]
        rows[5]["features"][FEATURES[0]] = dict(value=None, valid=False, reason="missing")
        r = [e.step(x) for x in rows][-1]
        self.assertAlmostEqual(r["support"][FEATURES[0]]["covered_seconds"], .8)
        self.assertEqual(r["support"][FEATURES[0]]["contributing_intervals"], 8)

    def test_current_missing_invalidates_despite_history(self):
        e = Baseline()
        rows = sequence()
        rows[-1]["features"][FEATURES[4]] = dict(value=None, valid=False, reason="missing")
        out = [e.step(r) for r in rows][-1]
        self.assertIn("insufficient:" + FEATURES[4], out["reasons"])

    def test_motion_is_norm_of_time_weighted_vector(self):
        e = Baseline()
        rows = sequence()[:11]
        for i, r in enumerate(rows):
            r["features"][FEATURES[4]]["value"] = [.1 if i % 2 else -.1, 0]
        out = [e.step(r) for r in rows][-1]
        self.assertAlmostEqual(out["support"][FEATURES[4]]["value"], 0)


class RuleTests(unittest.TestCase):
    def predict(self, **kwargs):
        e = Baseline()
        return [e.step(r) for r in sequence(**kwargs)][-1]

    def test_four_labels_and_conflicts(self):
        for options, expected in [({}, "standing"), ({"knee": 100}, "sitting"),
                                  ({"torso": 90, "aspect": 2}, "lying"),
                                  ({"hip": .06}, "moving"), ({"box": .06}, "moving"),
                                  ({"knee": 100, "hip": .06}, "unknown"),
                                  ({"torso": 90, "aspect": 2, "hip": .06}, "unknown"),
                                  ({"hip": .02}, "unknown"), ({"knee": 145}, "unknown"),
                                  ({"torso": 45, "hip": .06}, "unknown")]:
            self.assertEqual(self.predict(**options)["predicted_label"], expected, options)
        self.assertEqual(self.predict(knee=100, hip=.06)["reasons"], ["motion_posture_conflict"])

    def test_no_all_17_joint_requirement(self):
        row = self.predict()
        self.assertEqual(row["predicted_label"], "standing")  # Fixtures contain no joint array.

    def test_config_validation(self):
        for kw in [dict(window_seconds=0), dict(window_seconds=float("nan")), dict(minimum_coverage=1.1),
                   dict(minimum_intervals=1.5), dict(moving_hip_speed=.01), dict(standing_min_knee_deg=130)]:
            with self.assertRaises(ValueError):
                Config(**kw)


class OutputTests(unittest.TestCase):
    def test_preserves_rows_metadata_masks_provenance_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, out = Path(tmp) / "input.jsonl", Path(tmp) / "output"
            rows = sequence()
            p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            result = run_baseline(p, out)
            self.assertTrue(result["completed"])
            saved = [json.loads(x) for x in (out / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(saved), len(rows))
            for before, after in zip(rows, saved):
                for k in ("video_id", "frame_index", "pts", "time_base", "presentation_time_seconds"):
                    self.assertEqual(before[k], after[k])
                self.assertEqual(before["features"], after["input_features"])
            self.assertEqual(result["transition_count"], 1)
            self.assertEqual(result["transitions"][0]["frame_index"], 8)
            with self.assertRaises(ValueError):
                run_baseline(p, out)

    def test_partial_and_empty_runs_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.jsonl"
            p.write_text(json.dumps(frame(0, 0)) + "\nBAD\n")
            result = run_baseline(p, Path(tmp) / "partial")
            self.assertFalse(result["completed"])
            self.assertEqual(result["frame_count"], 1)
            p.write_text("")
            self.assertFalse(run_baseline(p, Path(tmp) / "empty")["completed"])


if __name__ == "__main__":
    unittest.main()
