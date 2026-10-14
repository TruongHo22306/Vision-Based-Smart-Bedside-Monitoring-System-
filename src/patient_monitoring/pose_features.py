"""Versioned causal per-frame geometry from saved selected poses; no inference."""

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from importlib import metadata
import json
import math
from pathlib import Path
import platform

from .pose_preview import KEYPOINT_NAMES, sha256_file
from .target_selection import valid_box, write_json
from .video_inspection import COLUMNS, InspectionError

FEATURE_VERSION = "per_frame_v001"
EPSILON_PX = 1e-9  # Numerical zero-length guard, not a pose-quality threshold.
QUALITY_FIELDS = ("finite_components", "finite_values", "confidence_pass",
                  "at_or_outside_boundary", "passes_quality_gate")
FEATURE_NAMES = ("hip_midpoint_px", "shoulder_midpoint_px", "torso_scale_px",
                 "left_knee_angle_deg", "right_knee_angle_deg", "torso_inclination_deg",
                 "hip_image_xy", "box_size_px", "box_size_image",
                 "hip_image_velocity_per_second", "box_size_image_velocity_per_second")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def feature(value=None, reason=None):
    """All missing/invalid results use JSON null, never a replacement zero."""
    if reason is None:
        values = value if isinstance(value, (list, tuple)) else [value]
        if not all(finite(v) for v in values):
            value, reason = None, "nonfinite_result"
    return dict(value=None if reason else value, valid=reason is None, reason=reason)


def midpoint(a, b):
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


def knee_angle(hip, knee, ankle):
    """Interior angle hip-knee-ankle in degrees, 180 = straight."""
    u = [hip[i] - knee[i] for i in (0, 1)]
    v = [ankle[i] - knee[i] for i in (0, 1)]
    nu, nv = math.hypot(*u), math.hypot(*v)
    if not finite(nu) or not finite(nv):
        return feature(reason="nonfinite_knee_geometry")
    if min(nu, nv) <= EPSILON_PX:
        return feature(reason="degenerate_knee_limb")
    cosine = sum((u[i] / nu) * (v[i] / nv) for i in (0, 1))
    return feature(math.degrees(math.acos(max(-1., min(1., cosine)))))


def geometry(row):
    """Only individual saved gate masks decide joint eligibility; no rethresholding."""
    person = row.get("selected_person")
    accepted = (row.get("status") in ("seeded", "matched")
                and isinstance(person, dict) and not row.get("association_ended", False))
    points = person.get("keypoints", []) if isinstance(person, dict) else []
    by_index = {}
    duplicates = set()
    if isinstance(points, list):
        for p in points:
            if isinstance(p, dict) and type(p.get("index")) is int:
                index = p["index"]
                if index in by_index:
                    duplicates.add(index)
                by_index[index] = p
    joints = []
    for index, name in enumerate(KEYPOINT_NAMES):
        p = by_index.get(index)
        reason = None
        if not accepted:
            reason = "target_unresolved"
        elif p is None:
            reason = "joint_missing"
        elif index in duplicates or p.get("name") != name:
            reason = "invalid_joint_identity"
        elif p.get("passes_quality_gate") is not True:
            reason = "saved_quality_gate_not_passed"
        elif not all(finite(p.get(k)) for k in ("x", "y", "confidence")):
            reason = "invalid_values_despite_saved_gate"
        xy = feature([p["x"], p["y"]] if reason is None else None, reason)
        joints.append(dict(index=index, name=name, pixel_xy=xy,
                           source_quality={k: p.get(k) for k in QUALITY_FIELDS} if p else None))

    def required(indices):
        bad = [joints[i]["name"] for i in indices if not joints[i]["pixel_xy"]["valid"]]
        return "invalid_required_joints:" + ",".join(bad) if bad else None

    def mid(indices):
        error = required(indices)
        return feature(reason=error) if error else feature(midpoint(*(joints[i]["pixel_xy"]["value"] for i in indices)))

    hips, shoulders = mid((11, 12)), mid((5, 6))
    if hips["valid"] and shoulders["valid"]:
        vector = [shoulders["value"][i] - hips["value"][i] for i in (0, 1)]
        distance = math.hypot(*vector)
        scale = feature(distance) if distance > EPSILON_PX else feature(reason="degenerate_torso_scale")
        inclination = (feature(math.degrees(math.atan2(abs(vector[0]), -vector[1])))
                       if scale["valid"] else feature(reason=scale["reason"]))
    else:
        scale = feature(reason="hip_or_shoulder_midpoint_invalid")
        inclination = feature(reason=scale["reason"])
    for joint in joints:
        p = joint["pixel_xy"]
        error = p["reason"] if not p["valid"] else scale["reason"]
        joint["body_xy"] = (feature(reason=error) if error else
                            feature([(p["value"][i] - hips["value"][i]) / scale["value"] for i in (0, 1)]))

    angles = []
    for indices in ((11, 13, 15), (12, 14, 16)):
        error = required(indices)
        angles.append(feature(reason=error) if error else
                      knee_angle(*(joints[i]["pixel_xy"]["value"] for i in indices)))
    w, h = row.get("width"), row.get("height")
    dimensions_valid = all(type(v) is int and v > 0 for v in (w, h))
    hip_image = (feature([hips["value"][0] / w, hips["value"][1] / h])
                 if hips["valid"] and dimensions_valid else
                 feature(reason="invalid_image_dimensions" if not dimensions_valid else hips["reason"]))
    box_error = "target_unresolved" if not accepted else (
        "invalid_image_dimensions" if not dimensions_valid else valid_box(person.get("box_xyxy"), w, h))
    if box_error:
        box_px = box_image = feature(reason=box_error)
    else:
        x1, y1, x2, y2 = person["box_xyxy"]
        box_px, box_image = feature([x2 - x1, y2 - y1]), feature([(x2 - x1) / w, (y2 - y1) / h])
    features = dict(hip_midpoint_px=hips, shoulder_midpoint_px=shoulders, torso_scale_px=scale,
                    left_knee_angle_deg=angles[0], right_knee_angle_deg=angles[1],
                    torso_inclination_deg=inclination, hip_image_xy=hip_image,
                    box_size_px=box_px, box_size_image=box_image)
    return features, joints, accepted


class FeatureExtractor:
    """Retain only the immediate prior row; rejected rows are barriers, not skipped."""

    def __init__(self, video_id, max_time_gap=.5):
        if not isinstance(video_id, str) or not video_id.strip():
            raise InspectionError("An explicit nonempty video ID is required")
        if not finite(max_time_gap) or max_time_gap <= 0:
            raise InspectionError("max_time_gap must be finite and > 0 seconds")
        self.video_id = video_id
        self.max_gap = Fraction(str(max_time_gap))
        self.previous = None
        self.context = self.high_water = None

    def step(self, row):
        if not isinstance(row, dict) or row.get("schema_version") != 1:
            raise InspectionError("Expected target-pose schema_version=1 object")
        if row.get("video_id", self.video_id) != self.video_id:
            raise InspectionError("Input video_id conflicts with explicit video ID")
        features, joints, accepted = geometry(row)
        context = tuple(row.get(k) for k in ("source_path", "stream_index", "width", "height"))
        if context != self.context:
            self.high_water = None
        self.context = context
        timestamp_error, current = None, None
        try:
            base = Fraction(row.get("time_base"))
            if base <= 0 or type(row.get("pts")) is not int:
                raise ValueError()
            current = row["pts"] * base
            reported = row.get("presentation_time_seconds")
            if not finite(reported) or not math.isclose(float(current), reported, rel_tol=0, abs_tol=1e-9):
                timestamp_error = "presentation_time_disagrees_with_pts"
            elif row.get("timestamp_valid") is not True:
                timestamp_error = "source_timestamp_invalid"
            elif self.high_water is not None and current <= self.high_water:
                timestamp_error = "nonincreasing_timestamp"
            if timestamp_error is None:
                self.high_water = current
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            timestamp_error = "missing_or_invalid_pts_time_base"
        index = row.get("frame_index")
        index_valid = type(index) is int and index >= 0
        prev = self.previous
        reason, dt = None, None
        if prev is None:
            reason = "no_previous_frame"
        elif not index_valid or not prev["index_valid"] or index != prev["index"] + 1:
            reason = "nonconsecutive_frame_index"
        elif context != prev["context"]:
            reason = "source_stream_or_dimensions_changed"
        elif not accepted or not prev["accepted"]:
            reason = "unresolved_target_barrier"
        elif row.get("status") == "seeded" or row.get("track_break") is True:
            reason = "explicit_track_break_or_reseed"
        elif any(row.get(k) != prev["row"].get(k) for k in ("track_id", "association_segment_id")):
            reason = "explicit_track_identifier_changed"
        elif row.get("reference_frame_index") != prev["index"]:
            reason = "association_reference_not_previous_frame"
        elif timestamp_error or prev["timestamp_error"]:
            reason = "invalid_timestamp_barrier"
        else:
            dt = current - prev["time"]
            if dt <= 0:
                reason = "nonpositive_delta"
            elif dt > self.max_gap:
                reason = "maximum_time_gap_exceeded"
        for source, target in (("hip_image_xy", "hip_image_velocity_per_second"),
                               ("box_size_image", "box_size_image_velocity_per_second")):
            error = reason
            if error is None and not features[source]["valid"]:
                error = "current_feature_invalid:" + features[source]["reason"]
            if error is None and not prev["features"][source]["valid"]:
                error = "previous_feature_invalid:" + prev["features"][source]["reason"]
            features[target] = (feature(reason=error) if error else
                                feature([(features[source]["value"][i] - prev["features"][source]["value"][i]) / float(dt)
                                         for i in (0, 1)]))
        result = {k: row.get(k) for k in (*COLUMNS, "source_path", "stream_index", "width", "height")}
        result.update(schema_version=1, feature_version=FEATURE_VERSION, video_id=self.video_id,
                      target_status=row.get("status"), target_reason=row.get("reason"),
                      selected_source_detection_index=row.get("selected_source_detection_index"),
                      timestamp_valid=timestamp_error is None, timestamp_error=timestamp_error,
                      source_timestamp_valid=row.get("timestamp_valid"),
                      temporal_pair=dict(valid=reason is None, reason=reason,
                                         previous_frame_index=prev["index"] if prev else None,
                                         delta_seconds=float(dt) if dt is not None else None),
                      features=features, joints=joints)
        # Never retain the last valid row across an invalid/missing row.
        self.previous = dict(index=index, index_valid=index_valid, context=context, accepted=accepted,
                             timestamp_error=timestamp_error, time=current, features=features,
                             row={k: row.get(k) for k in ("track_id", "association_segment_id")})
        return result


def extract_features(path: Path, output: Path, video_id: str, max_time_gap=.5):
    engine = FeatureExtractor(video_id, max_time_gap)
    path, output = path.resolve(), output.resolve()
    if output.exists():
        raise InspectionError("Output exists; choose a new versioned output directory")
    digest = sha256_file(path)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(max_time_gap_seconds=max_time_gap, degenerate_length_epsilon_px=EPSILON_PX,
                  quality_policy="Reuse saved passes_quality_gate; never rethreshold confidence",
                  units="pixels, degrees, image fractions, image fractions per second",
                  timing="Exact rational PTS deltas; presentation timestamp consistency tolerance1e-9s",
                  temporal_policy="Immediate adjacent input frames only; source/track/seed/reference/timestamp/gap barriers")
    summary = dict(status="incomplete", completed=False, feature_version=FEATURE_VERSION, video_id=video_id,
                   input_frame_count=0, written_frame_count=0, feature_valid_counts={k: 0 for k in FEATURE_NAMES},
                   feature_invalid_reasons={}, normalized_joint_valid_counts={n: 0 for n in KEYPOINT_NAMES},
                   all_17_normalized_valid_frames=0, core_geometry_valid_frames=0,
                   timestamp_invalid_frames=0, temporal_pair_valid_frames=0, target_status_counts={},
                   denominator="Per-feature valid counts use all written frame rows, including unresolved; per-joint counts use one slot per written frame. No accuracy ratios.",
                   config=config, errors=[])
    provenance = dict(status="incomplete", feature_version=FEATURE_VERSION, video_id=video_id,
                      created_at_utc=datetime.now(timezone.utc).isoformat(), input_path=str(path), input_sha256=digest,
                      code_sha256=sha256_file(Path(__file__)), config=config, python=platform.python_version(),
                      project_version=metadata.version("patient-monitoring"), inference_performed=False,
                      package_versions={name: metadata.version(name) for name in ("av", "Pillow")},
                      helper_code_sha256={name: sha256_file(Path(__file__).with_name(name))
                                          for name in ("pose_preview.py", "target_selection.py", "video_inspection.py")},
                      labels_used=False, source_video_not_decoded=True,
                      video_id_origin="Explicit caller ID, checked against any embedded video_id; not guessed from filename")
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", provenance)
    statuses, failures = Counter(), {k: Counter() for k in FEATURE_NAMES}
    try:
        with path.open(encoding="utf-8") as source, (output / "features.jsonl").open("x", encoding="utf-8") as dest:
            for line in source:
                row = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Nonstandard JSON {v}")))
                summary["input_frame_count"] += 1
                result = engine.step(row)
                dest.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                summary["written_frame_count"] += 1
                statuses[result["target_status"]] += 1
                summary["timestamp_invalid_frames"] += int(not result["timestamp_valid"])
                summary["temporal_pair_valid_frames"] += int(result["temporal_pair"]["valid"])
                for name, f in result["features"].items():
                    summary["feature_valid_counts"][name] += int(f["valid"])
                    if not f["valid"]:
                        failures[name][f["reason"]] += 1
                for joint in result["joints"]:
                    summary["normalized_joint_valid_counts"][joint["name"]] += int(joint["body_xy"]["valid"])
                summary["all_17_normalized_valid_frames"] += int(all(j["body_xy"]["valid"] for j in result["joints"]))
                summary["core_geometry_valid_frames"] += int(all(result["features"][k]["valid"] for k in FEATURE_NAMES[:9]))
        if not summary["written_frame_count"]:
            raise InspectionError("Empty target-pose input")
        if sha256_file(path) != digest:
            raise InspectionError("Input changed during extraction")
        summary.update(status="complete", completed=True)
    except (Exception, KeyboardInterrupt) as exc:
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    summary["target_status_counts"] = dict(statuses)
    summary["feature_invalid_reasons"] = {k: dict(v) for k, v in failures.items()}
    provenance.update(status=summary["status"], errors=summary["errors"],
                      output_sha256=sha256_file(output / "features.jsonl") if (output / "features.jsonl").exists() else None)
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", provenance)
    return summary
