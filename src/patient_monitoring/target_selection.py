"""Experimental causal single-target association; saved boxes only, never inference."""

from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path

from .pose_preview import sha256_file
from .video_inspection import COLUMNS, InspectionError, TimelineAudit


def valid_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return "box must contain four coordinates"
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
        return "nonfinite or nonnumeric coordinate"
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return "box is degenerate, inverted, or outside image bounds"
    return None


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union


class TargetAssociator:
    """Reference state advances only on explicit seed or an accepted unique IoU match."""

    def __init__(self, seed_frame, seed_detection, min_iou=.1, ambiguity_margin=.05, max_gap=.5):
        if any(type(v) is not int or v < 0 for v in (seed_frame, seed_detection)):
            raise InspectionError("Seed frame/detection must be nonnegative integers")
        if (any(not math.isfinite(v) for v in (min_iou, ambiguity_margin, max_gap))
                or not 0 <= min_iou <= 1 or not 0 <= ambiguity_margin <= 1 or max_gap <= 0):
            raise InspectionError("Require IoU/margin in [0,1] and finite max gap > 0")
        self.config = dict(seed_frame=seed_frame, seed_detection=seed_detection, min_iou=min_iou,
                           ambiguity_margin=ambiguity_margin, max_gap_seconds=max_gap)
        self.max_gap = Fraction(str(max_gap))
        self.audit = TimelineAudit()
        self.last_time = self.last_box = self.last_frame = self.high_water_time = None
        self.seed_seen = self.ended = False
        self.context = None

    def timestamp(self, row):
        try:
            base = Fraction(row["time_base"])
            if base <= 0 or type(row["pts"]) is not int:
                raise ValueError("Missing/invalid PTS or time_base")
            expected = self.audit.observe(row["pts"], base)
            current = row["pts"] * base
        except (TypeError, ValueError, ZeroDivisionError):
            self.audit.observe(None, None)
            return None, "missing/invalid PTS or time_base"
        if any(row[k] != expected[k] for k in COLUMNS):
            return None, "timestamp fields disagree with M1.1 PTS timeline"
        if self.high_water_time is not None and current <= self.high_water_time:
            return None, "duplicate or decreasing presentation timestamp"
        self.high_water_time = current
        return current, None

    def step(self, row):
        if type(row["frame_index"]) is not int or row["frame_index"] != self.audit.decoded_count:
            raise InspectionError("Expected one consecutively indexed input row per frame, starting at zero")
        width, height = row["width"], row["height"]
        if any(type(v) is not int or v <= 0 for v in (width, height)):
            raise InspectionError("Invalid source image dimensions")
        context = (row["source_path"], row["stream_index"], width, height)
        if self.context is not None and context != self.context:
            raise InspectionError("Source/stream/dimensions changed within the sequence")
        self.context = context
        current, time_error = self.timestamp(row)
        people = row["people"]
        detection_error = None
        if row["inference_status"] != "complete" or not isinstance(people, list):
            people = []
            detection_error = "inference not complete; detections not evaluated"
        elif row["person_count"] != len(people):
            detection_error = "person_count disagrees with people"
        indices = [p["detection_index"] for p in people]
        if any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
            detection_error = "invalid or duplicate frame-local detection indices"
        candidates = []
        for person in people:
            error = valid_box(person["box_xyxy"], width, height)
            candidates.append(dict(source_detection_index=person["detection_index"], box_valid=error is None,
                box_error=error, iou=(box_iou(self.last_box, person["box_xyxy"])
                                    if error is None and self.last_box is not None else None)))
        result = {k: row[k] for k in (*COLUMNS, "source_path", "stream_index", "width", "height")}
        result.update(schema_version=1, selected_source_detection_index=None, selected_person=None,
            status=None, reason=None, reference_frame_index=self.last_frame,
            reference_time_seconds=float(self.last_time) if self.last_time is not None else None,
            elapsed_since_reference_seconds=float(current-self.last_time)
                if current is not None and self.last_time is not None else None,
            timestamp_valid=time_error is None, timestamp_error=time_error,
            candidates=candidates, best_iou=None, runner_up_iou=None, iou_separation=None)

        def reject(status, reason):
            result.update(status=status, reason=reason, association_ended=self.ended)
            return result

        def accept(person, status):
            self.last_box = tuple(person["box_xyxy"])
            self.last_time, self.last_frame = current, row["frame_index"]
            result.update(selected_source_detection_index=person["detection_index"], selected_person=person,
                          status=status, reason="explicit seed" if status == "seeded" else "unique highest IoU meets configured gates",
                          association_ended=False)
            return result

        if row["frame_index"] < self.config["seed_frame"]:
            return reject("awaiting_seed", "before explicit seed frame")
        if row["frame_index"] == self.config["seed_frame"]:
            self.seed_seen = True
            seed = next((p for p in people if p["detection_index"] == self.config["seed_detection"]), None)
            if time_error or detection_error or seed is None or valid_box(seed["box_xyxy"], width, height):
                self.ended = True
                return reject("seed_rejected", time_error or detection_error or "seed missing or invalid box; explicit reseed required")
            return accept(seed, "seeded")
        if self.ended:
            return reject("association_ended", "explicit reseed required; no automatic reacquisition")
        if time_error:
            return reject("invalid_timestamp", time_error)
        if current - self.last_time > self.max_gap:
            self.ended = True
            return reject("gap_timeout", "elapsed time from last accepted detection exceeds max gap; explicit reseed required")
        if detection_error:
            return reject("invalid_detections", detection_error)
        if not people:
            return reject("no_detections", "no saved person detections")
        ranked = sorted((c for c in candidates if c["box_valid"]), key=lambda c: -c["iou"])
        if not ranked:
            return reject("invalid_boxes", "no geometrically valid boxes")
        best = ranked[0]
        runner = ranked[1]["iou"] if len(ranked) > 1 else None
        separation = best["iou"] - runner if runner is not None else None
        result.update(best_iou=best["iou"], runner_up_iou=runner, iou_separation=separation)
        if best["iou"] < self.config["min_iou"]:
            return reject("low_iou", "highest IoU below minimum")
        if runner is not None and (separation <= 0 or separation < self.config["ambiguity_margin"]):
            return reject("ambiguous", "highest IoU not sufficiently separated from runner-up")
        chosen = next(p for p in people if p["detection_index"] == best["source_detection_index"])
        return accept(chosen, "matched")


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_target(poses: Path, output: Path, seed_frame, seed_detection,
                  min_iou=.1, ambiguity_margin=.05, max_gap=.5):
    engine = TargetAssociator(seed_frame, seed_detection, min_iou, ambiguity_margin, max_gap)
    poses, output = poses.resolve(), output.resolve()
    if output.exists():
        raise InspectionError("Output exists; choose a fresh directory")
    digest = sha256_file(poses)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(task="M2.4", status="incomplete", completed=False, input_path=str(poses),
        input_sha256=digest, config=engine.config, parameters_validated=False,
        method="Causal IoU to last accepted box; explicit seed; no confidence/keypoint quality ranking",
        single_candidate_policy="No runner-up: ambiguity margin not applicable. Exact ties always rejected.",
        gap_policy="PTS-based; elapsed > max_gap ends association permanently for this run. Equality is allowed.",
        experimental_note="Single-person baseline; not validated tracking or an identity guarantee",
        input_frame_count=0, written_frame_count=0, selected_frame_count=0, unresolved_frame_count=0,
        status_counts={}, gap_ranges=[], errors=[])
    summary_path = output / "selection_summary.json"
    write_json(summary_path, summary)
    statuses = Counter()
    gap = None
    try:
        with poses.open(encoding="utf-8") as source, (output / "target_poses.jsonl").open("x", encoding="utf-8") as target:
            for line in source:
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonstandard JSON value: {value}")))
                summary["input_frame_count"] += 1
                selected = engine.step(row)
                target.write(json.dumps(selected, ensure_ascii=False, allow_nan=False) + "\n")
                target.flush()
                summary["written_frame_count"] += 1
                statuses[selected["status"]] += 1
                if selected["selected_person"] is None:
                    summary["unresolved_frame_count"] += 1
                    t = selected["presentation_time_seconds"] if selected["timestamp_valid"] else None
                    if gap is None:
                        gap = dict(start_frame_index=row["frame_index"], start_time_seconds=t,
                            end_frame_index=row["frame_index"], end_time_seconds=t, frame_count=0, status_counts={},
                            last_accepted_time_seconds=selected["reference_time_seconds"],
                            resumed_at_frame_index=None, resumed_at_time_seconds=None,
                            elapsed_reference_to_resume_seconds=None, unresolved_timestamp_span_seconds=None)
                        summary["gap_ranges"].append(gap)
                    gap.update(end_frame_index=row["frame_index"], end_time_seconds=t, frame_count=gap["frame_count"] + 1)
                    gap["status_counts"][selected["status"]] = gap["status_counts"].get(selected["status"], 0) + 1
                    gap["unresolved_timestamp_span_seconds"] = (
                        t - gap["start_time_seconds"] if t is not None and gap["start_time_seconds"] is not None else None)
                else:
                    summary["selected_frame_count"] += 1
                    if gap is not None:
                        gap.update(resumed_at_frame_index=row["frame_index"], resumed_at_time_seconds=row["presentation_time_seconds"],
                                   elapsed_reference_to_resume_seconds=selected["elapsed_since_reference_seconds"])
                        gap = None
                summary["status_counts"] = dict(statuses)
                if summary["written_frame_count"] % 50 == 0:
                    write_json(summary_path, summary)
        if not engine.seed_seen:
            raise InspectionError("Explicit seed frame not present in input")
        if statuses["seed_rejected"]:
            raise InspectionError("Explicit seed rejected; rerun with a valid explicit seed in a new output directory")
        if sha256_file(poses) != digest:
            raise InspectionError("Input predictions changed during association")
        summary.update(status="complete", completed=True)
    except (Exception, KeyboardInterrupt) as exc:
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    summary["association_ended"] = engine.ended
    summary["timestamp_audit"] = engine.audit.summary()
    summary["gap_range_definition"] = "Contiguous null-selection rows, including pre-seed/ended rows. Times are original PTS; invalid times are null. Span excludes resumed frame, not inferred missing duration."
    write_json(summary_path, summary)
    return summary
