"""Experimental causal rules on saved features; no model, labels or training."""

from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform


VERSION = "action_rules_v001"
FEATURES = ("torso_inclination_deg", "left_knee_angle_deg", "right_knee_angle_deg",
            "box_size_px", "hip_image_velocity_per_second", "box_size_image_velocity_per_second")
META = ("video_id", "frame_index", "pts", "time_base", "presentation_time_seconds",
        "relative_time_seconds", "delta_seconds", "source_path", "stream_index", "width", "height")


@dataclass(frozen=True)
class Config:
    window_seconds: float = 1.0
    minimum_coverage: float = 0.8
    minimum_intervals: int = 3
    max_gap_seconds: float = 0.5
    upright_max_deg: float = 30.0
    lying_min_deg: float = 60.0
    lying_max_deg: float = 120.0
    lying_min_box_aspect: float = 1.2
    sitting_max_knee_deg: float = 135.0
    standing_min_knee_deg: float = 155.0
    quiet_hip_speed: float = 0.015
    moving_hip_speed: float = 0.03
    quiet_box_speed: float = 0.015
    moving_box_speed: float = 0.03

    def __post_init__(self):
        if any(not finite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("All configuration values must be finite and positive")
        if not 0 < self.minimum_coverage <= 1 or type(self.minimum_intervals) is not int:
            raise ValueError("Coverage must be in (0,1]; minimum_intervals must be an integer")
        if not (self.upright_max_deg < self.lying_min_deg < self.lying_max_deg <= 180
                and self.sitting_max_knee_deg < self.standing_min_knee_deg <= 180
                and self.quiet_hip_speed < self.moving_hip_speed
                and self.quiet_box_speed < self.moving_box_speed):
            raise ValueError("Invalid or overlapping rule thresholds")


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def value(row, name):
    f = row.get("features", {}).get(name)
    if not isinstance(f, dict) or f.get("valid") is not True:
        return None
    v = f.get("value")
    if name in FEATURES[:3]:
        return v if finite(v) and 0 <= v <= 180 else None
    if not isinstance(v, list) or len(v) != 2 or not all(finite(x) for x in v):
        return None
    if name == "box_size_px" and min(v) <= 0:
        return None
    return v


def decide(support, config):
    """No source identifiers, filenames or annotations enter this decision."""
    reasons = [f"insufficient:{name}" for name in FEATURES if not support[name]["valid"]]
    if reasons:
        return "unknown", reasons, None, None
    torso, left, right, aspect, hip, box = (support[n]["value"] for n in FEATURES)
    upright = torso <= config.upright_max_deg
    if config.lying_min_deg <= torso <= config.lying_max_deg and aspect >= config.lying_min_box_aspect:
        posture = "lying"
    elif upright and max(left, right) <= config.sitting_max_knee_deg:
        posture = "sitting"
    elif upright and min(left, right) >= config.standing_min_knee_deg:
        posture = "standing"
    else:
        posture = None
    movement = ("high" if hip >= config.moving_hip_speed or box >= config.moving_box_speed else
                "quiet" if hip <= config.quiet_hip_speed and box <= config.quiet_box_speed else "indeterminate")
    if movement == "indeterminate":
        return "unknown", ["motion_threshold_band"], posture, movement
    if movement == "high":
        if posture in ("sitting", "lying"):
            return "unknown", ["motion_posture_conflict"], posture, movement
        if upright:
            return "moving", ["upright_with_temporal_motion"], posture, movement
        return "unknown", ["motion_without_upright_support"], posture, movement
    if posture:
        return posture, ["quiet_with_geometry_support"], posture, movement
    return "unknown", ["ambiguous_posture_geometry"], posture, movement


class Baseline:
    def __init__(self, config=Config()):
        self.config = config
        self.history = deque()
        self.previous = None
        self.context = None
        self.high_water = None

    def step(self, row):
        if not isinstance(row, dict) or row.get("schema_version") != 1 or row.get("feature_version") != "per_frame_v001":
            raise ValueError("Expected saved per_frame_v001 feature schema 1")
        if not isinstance(row.get("features"), dict):
            raise ValueError("Missing features object")
        if not isinstance(row.get("video_id"), str) or not row["video_id"].strip():
            raise ValueError("A saved video_id is required for boundary isolation, never label prediction")
        c = self.config
        context = tuple(row.get(k) for k in ("video_id", "source_path", "stream_index", "width", "height"))
        if context != self.context:
            self.high_water = None
        now, error = None, None
        try:
            base = Fraction(row.get("time_base"))
            if base <= 0 or type(row.get("pts")) is not int:
                raise ValueError()
            now = base * row["pts"]
            if (row.get("timestamp_valid") is not True or not finite(row.get("presentation_time_seconds"))
                    or not math.isclose(float(now), row["presentation_time_seconds"], rel_tol=0, abs_tol=1e-9)):
                error = "invalid_timestamp"
            elif self.high_water is not None and now <= self.high_water:
                error = "nonincreasing_timestamp"
            else:
                self.high_water = now
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            error = "invalid_timestamp"
        time_valid = error is None
        if type(row.get("frame_index")) is not int or row["frame_index"] < 0:
            error = "invalid_frame_index"
        if row.get("target_status") not in ("seeded", "matched") or row.get("selected_source_detection_index") is None:
            error = error or "unresolved_target"
        prev = self.previous
        pair = row.get("temporal_pair") or {}
        reset = error
        if reset is None:
            if prev is None:
                reset = "segment_start"
            elif context != self.context:
                reset = "video_source_context_changed"
            elif row.get("target_status") == "seeded" or row.get("track_break") is True:
                reset = "reseed_or_track_break"
            elif any(row.get(k) != prev[1].get(k) for k in ("track_id", "association_segment_id")):
                reset = "track_identifier_changed"
            elif row["frame_index"] != prev[1]["frame_index"] + 1:
                reset = "nonconsecutive_frame_index"
            elif pair.get("valid") is not True or pair.get("previous_frame_index") != prev[1]["frame_index"]:
                reset = "saved_temporal_barrier:" + str(pair.get("reason"))
            elif now - prev[0] > Fraction(str(c.max_gap_seconds)):
                reset = "maximum_time_gap_exceeded"
            elif now <= prev[0]:
                reset = "nonincreasing_timestamp"
        if reset:
            self.history.clear()
        if error is None:
            # Store only needed feature values/masks and continuity metadata, not the 17 joints.
            saved = {k: row.get(k) for k in (*META, "target_status", "track_id", "association_segment_id")}
            saved["features"] = {k: row["features"].get(k) for k in FEATURES}
            self.history.append((now, saved))
            cutoff = now - Fraction(str(c.window_seconds))
            while len(self.history) > 1 and self.history[1][0] <= cutoff:
                self.history.popleft()
            self.previous = (now, saved)
        else:
            self.previous = None
        self.context = context
        support = {}
        for name in FEATURES:
            covered, count, total = 0.0, 0, [0.0, 0.0]
            entries = list(self.history)
            for (ta, a), (tb, b) in zip(entries, entries[1:]):
                duration = float(tb - max(ta, now - Fraction(str(c.window_seconds))))
                va, vb = value(a, name), value(b, name)
                # A derivative already validates both endpoints; a static value needs both masks.
                velocity = name.endswith("velocity_per_second")
                if duration <= 0 or vb is None or (not velocity and va is None):
                    continue
                measured = vb if velocity else [vb[0] / vb[1] if name == "box_size_px" else vb, 0.0]
                covered += duration
                count += 1
                for axis in (0, 1):
                    total[axis] += measured[axis] * duration
            valid = (error is None and value(row, name) is not None and
                     covered + 1e-12 >= c.window_seconds * c.minimum_coverage and count >= c.minimum_intervals)
            aggregate = ((math.hypot(*total) if name.endswith("velocity_per_second") else total[0]) / covered
                         if covered > 0 else None)
            support[name] = dict(value=aggregate if valid else None, valid=valid,
                                 reason=None if valid else "current_value_missing_or_insufficient_time_coverage",
                                 covered_seconds=covered, coverage_fraction=covered / c.window_seconds,
                                 contributing_intervals=count,
                                 mean_vector=[x / covered for x in total] if valid and name.endswith("velocity_per_second") else None,
                                 statistic="norm_of_time_weighted_vector_mean" if name.endswith("velocity_per_second") else "time_weighted_mean",
                                 units="image_fraction_per_second" if name.endswith("velocity_per_second") else
                                       "pixel_width_over_height" if name == "box_size_px" else "degrees")
        label, reasons, posture, motion = decide(support, c)
        if error:
            label, reasons = "unknown", [error]
        result = {k: row.get(k) for k in META}
        result.update(schema_version=1, baseline_version=VERSION, predicted_label=label, reasons=reasons,
                      posture_evidence=posture, motion_evidence=motion,
                      target_status=row.get("target_status"), timestamp_valid=time_valid,
                      selected_source_detection_index=row.get("selected_source_detection_index"),
                      input_features={k: row["features"].get(k) for k in FEATURES},
                      support=support, window=dict(duration_seconds=c.window_seconds,
                      start_exclusive_seconds=float(now) - c.window_seconds if now is not None else None,
                      end_inclusive_seconds=float(now) if now is not None else None,
                      oldest_retained_frame=self.history[0][1]["frame_index"] if self.history else None,
                      reset_reason=reset))
        return result


def run_baseline(path, output, config=Config()):
    path, output = Path(path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output exists; use a NEW versioned directory")
    digest = sha256(path)
    output.mkdir(parents=True, exist_ok=False)
    config_dict = asdict(config)
    config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
    provenance = dict(baseline_version=VERSION, status="incomplete", input_path=str(path), input_sha256=digest,
                      config=config_dict, config_sha256=config_hash, code_sha256=sha256(__file__),
                      python_version=platform.python_version(), created_at_utc=datetime.now(timezone.utc).isoformat(),
                      inference_performed=False, annotations_used=False, thresholds_validated=False)
    summary = dict(status="incomplete", completed=False, frame_count=0, label_counts={}, unknown_reason_counts={},
                   transitions=[], errors=[], denominator="All written frame rows; counts and transitions are diagnostics, not accuracy.")
    write_json(output / "provenance.json", provenance)
    write_json(output / "summary.json", summary)
    labels, unknowns, previous = Counter(), Counter(), None
    engine = Baseline(config)
    try:
        with path.open(encoding="utf-8") as source, (output / "predictions.jsonl").open("x", encoding="utf-8") as dest:
            for line in source:
                row = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {v}")))
                result = engine.step(row)
                result.update(input_sha256=digest, config_sha256=config_hash)
                dest.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                label = result["predicted_label"]
                labels[label] += 1
                if label == "unknown":
                    unknowns.update(result["reasons"])
                if previous and previous["video_id"] == result["video_id"] and previous["predicted_label"] != label:
                    summary["transitions"].append(dict(frame_index=result["frame_index"], pts=result["pts"],
                        time_base=result["time_base"], presentation_time_seconds=result["presentation_time_seconds"],
                        from_label=previous["predicted_label"], to_label=label, reasons=result["reasons"],
                        window_reset_reason=result["window"]["reset_reason"]))
                previous = result
                summary["frame_count"] += 1
        if not summary["frame_count"]:
            raise ValueError("Empty feature input")
        if sha256(path) != digest:
            raise ValueError("Input changed during run")
        summary.update(status="complete", completed=True)
    except (Exception, KeyboardInterrupt) as exc:
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    summary.update(label_counts={k: labels[k] for k in ("standing", "sitting", "lying", "moving", "unknown")},
                   unknown_reason_counts=dict(unknowns), transition_count=len(summary["transitions"]))
    provenance.update(status=summary["status"], errors=summary["errors"],
                      output_sha256=sha256(output / "predictions.jsonl") if (output / "predictions.jsonl").exists() else None)
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", provenance)
    return summary
