"""Sequential, timestamp-preserving pose extraction using the verified M2.1 model."""

from datetime import datetime, timezone
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform

import av

from .pose_preview import (
    KEYPOINT_NAMES, MODEL_NAME, MODEL_SHA256, MODEL_URL, PREDICT_CONFIG,
    serialize_people, sha256_file,
)
from .video_inspection import InspectionError, TimelineAudit, rational_text

QUALITY_CONFIDENCE = 0.5


def quality_keypoint(point, width, height):
    """Keep raw finite values; null only nonfinite values, with independent flags."""
    finite = {key: math.isfinite(point[key]) for key in ("x", "y", "confidence")}
    coordinates_finite = finite["x"] and finite["y"]
    confidence_pass = finite["confidence"] and point["confidence"] >= QUALITY_CONFIDENCE
    boundary = coordinates_finite and (
        point["x"] <= 0 or point["x"] >= width
        or point["y"] <= 0 or point["y"] >= height
    )
    return {
        **point,
        **{key: point[key] if valid else None for key, valid in finite.items()},
        "finite_components": finite,
        "finite_values": all(finite.values()),
        "confidence_pass": confidence_pass,
        "at_or_outside_boundary": boundary,
        "passes_quality_gate": coordinates_finite and confidence_pass and not boundary,
    }


def quality_people(prediction, width, height):
    people = serialize_people(prediction)
    for person in people:
        box = person["box_xyxy"]
        person["box_finite_values"] = [math.isfinite(value) for value in box]
        person["box_xyxy"] = [value if math.isfinite(value) else None for value in box]
        confidence = person["box_confidence"]
        person["box_confidence_finite"] = math.isfinite(confidence)
        person["box_confidence"] = confidence if math.isfinite(confidence) else None
        person["keypoints"] = [
            quality_keypoint(point, width, height) for point in person["keypoints"]
        ]
    return people


def write_report(path, report):
    # Only called in a newly created output directory owned by this run.
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def load_verified_model(run):
    """Load only cached M2.1 weights. Never install packages or download a model."""
    models_dir = Path(__file__).resolve().parents[2] / "models"
    weights = models_dir / MODEL_NAME
    if not weights.is_file():
        raise InspectionError(f"Verified M2.1 weights missing: {weights}")
    actual_hash = sha256_file(weights)
    run["model"].update(path=str(weights), sha256=actual_hash)
    if actual_hash != MODEL_SHA256:
        raise InspectionError("Model SHA-256 differs from verified M2.1 weights")
    os.environ["YOLO_CONFIG_DIR"] = str(models_dir / "ultralytics_settings")
    os.environ["MPLCONFIGDIR"] = str(models_dir / "matplotlib_settings")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import YOLO
    return YOLO(str(weights), task="pose")


def update_counts(summary, people):
    summary["processed_frame_count"] += 1
    summary["zero_detection_frame_count"] += int(len(people) == 0)
    summary["multiple_detection_frame_count"] += int(len(people) > 1)
    summary["person_detection_count"] += len(people)
    for person in people:
        for point, counts in zip(person["keypoints"], summary["per_joint"]):
            counts["observation_count"] += 1
            finite = point["finite_components"]
            counts["finite_confidence_count"] += int(finite["confidence"])
            counts["finite_coordinates_count"] += int(finite["x"] and finite["y"])
            counts["nonfinite_count"] += int(not point["finite_values"])
            counts["low_confidence_count"] += int(
                finite["confidence"] and not point["confidence_pass"]
            )
            counts["boundary_count"] += int(point["at_or_outside_boundary"])
            counts["passes_quality_gate_count"] += int(point["passes_quality_gate"])


def extract_poses(source: Path, output_dir: Path):
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise InspectionError(f"Input must be a nonempty regular video file: {source}")
    if output_dir.exists() or output_dir.is_symlink():
        raise InspectionError("Output exists; choose a new directory. No previous output is replaced.")
    source_hash = sha256_file(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    audit = TimelineAudit()
    summary = {
        "schema_version": 1, "status": "incomplete", "completed": False,
        "source_path": str(source), "source_sha256": source_hash,
        "decoded_frame_count": 0, "processed_frame_count": 0, "written_record_count": 0,
        "decode_reached_eof": False, "reported_frame_count": None,
        "selected_stream_index": None, "corrupt_frame_count": 0,
        "zero_detection_frame_count": 0, "multiple_detection_frame_count": 0,
        "person_detection_count": 0, "errors": [], "warnings": [],
        "quality_confidence": QUALITY_CONFIDENCE,
        "count_definitions": {
            "frame_detection_counts": "Successfully inferred and written frames only; errors are not zero detections.",
            "per_joint_observation_count": "One joint per detected person in each processed frame; zero-detection frames add no joints.",
            "low_confidence_count": "Finite confidence < 0.5; nonfinite values counted separately.",
            "boundary_count": "Both coordinates finite and at least one at/outside an image boundary.",
            "overlap": "Low-confidence, boundary and nonfinite counts may overlap; counts are not ratios.",
        },
        "per_joint": [dict(
            index=i, name=name, observation_count=0, finite_confidence_count=0,
            finite_coordinates_count=0, nonfinite_count=0, low_confidence_count=0,
            boundary_count=0, passes_quality_gate_count=0,
        ) for i, name in enumerate(KEYPOINT_NAMES)],
    }
    run = {
        "task": "M2.2", "schema_version": 1, "status": "incomplete", "errors": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source), "source_sha256": source_hash,
        "python_version": platform.python_version(),
        "package_versions": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "model": {"filename": MODEL_NAME, "official_url": MODEL_URL, "sha256": None},
        "config": dict(PREDICT_CONFIG), "quality_confidence": QUALITY_CONFIDENCE,
        "quality_gate_note": "Provisional conservative filter, not accuracy or proof of joint visibility.",
        "detection_index_scope": "Frame-local ordering, not a persistent tracking ID.",
        "timestamp_convention": "M1.1: pts*time_base; relative to first valid PTS; no FPS substitution.",
        "resampling": False,
    }

    def checkpoint():
        summary["decoded_frame_count"] = audit.decoded_count
        summary["timestamp_audit"] = audit.summary()
        run["decoded_frame_count"] = audit.decoded_count
        run["processed_frame_count"] = summary["processed_frame_count"]
        run["written_record_count"] = summary["written_record_count"]
        write_report(output_dir / "summary.json", summary)
        write_report(output_dir / "run_info.json", run)

    checkpoint()
    stage = "model_load"
    active_index = None
    try:
        model = load_verified_model(run)
        with (output_dir / "poses.jsonl").open("x", encoding="utf-8") as output:
            stage = "decode"
            with av.open(str(source), mode="r") as container:
                if not container.streams.video:
                    raise InspectionError("Input contains no video stream")
                stream = container.streams.video[0]
                summary["selected_stream_index"] = stream.index
                summary["reported_frame_count"] = stream.frames or None
                run["stream"] = {
                    "index": stream.index, "codec": stream.codec_context.name,
                    "time_base": rational_text(stream.time_base),
                }
                for frame in container.decode(stream):
                    row = audit.observe(frame.pts, frame.time_base)
                    active_index = row["frame_index"]
                    corrupt = bool(frame.is_corrupt)
                    summary["corrupt_frame_count"] += int(corrupt)
                    record = {
                        "schema_version": 1, "source_path": str(source),
                        "stream_index": stream.index, **row,
                        "width": frame.width, "height": frame.height, "is_corrupt": corrupt,
                    }
                    stage = "inference"
                    try:
                        with frame.to_image() as picture:
                            prediction = model.predict(source=picture, **PREDICT_CONFIG)[0]
                        if tuple(prediction.orig_shape) != (frame.height, frame.width):
                            raise InspectionError("Pose result dimensions do not match decoded frame")
                        people = quality_people(prediction, frame.width, frame.height)
                    except (Exception, KeyboardInterrupt) as exc:
                        record.update(
                            inference_status="failed", detection_status="not_evaluated",
                            person_count=None, people=None, error=f"{type(exc).__name__}: {exc}",
                        )
                        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        output.flush()
                        summary["written_record_count"] += 1
                        raise
                    record.update(
                        inference_status="complete", detection_status=(
                            "zero_detections" if not people else "multiple_detections"
                            if len(people) > 1 else "detected"
                        ), person_count=len(people), people=people,
                    )
                    stage = "write_record"
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    summary["written_record_count"] += 1
                    update_counts(summary, people)
                    if summary["processed_frame_count"] % 25 == 0:
                        checkpoint()
                        print(f"Processed {summary['processed_frame_count']} frames", flush=True)
                    stage = "decode"
                summary["decode_reached_eof"] = True
        stage = "completion_checks"
        if audit.decoded_count == 0:
            raise InspectionError("No decoded frames; completion cannot be verified")
        reported = summary["reported_frame_count"]
        if reported and reported != audit.decoded_count:
            summary["warnings"].append("Decoded count differs from reported stream count")
            if audit.decoded_count < reported:
                raise InspectionError("Decoded frame count is below reported frame count")
        if summary["corrupt_frame_count"]:
            raise InspectionError("Decoder returned corrupt frames; clean completion unverified")
        if sha256_file(source) != source_hash:
            raise InspectionError("Source video changed during extraction")
        parameter = next(model.model.parameters())
        run["actual_model_device"] = str(parameter.device)
        run["actual_model_dtype"] = str(parameter.dtype)
        run["effective_predict_config"] = {
            key: getattr(model.predictor.args, key) for key in PREDICT_CONFIG
            if hasattr(model.predictor.args, key)
        }
        run["runtime_unreported_config"] = [
            key for key in PREDICT_CONFIG if not hasattr(model.predictor.args, key)
        ]
        summary.update(status="complete", completed=True)
        run["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        error = {"stage": stage, "last_decoded_frame_index": active_index,
                 "type": type(exc).__name__, "message": str(exc)}
        summary["errors"].append(error)
        run["errors"].append(error)
    summary["warnings"].extend([
        "Confidence/gate pass is not accuracy or evidence of visibility; partial-body poses may be unreliable.",
        "Timestamp anomalies are preserved, not repaired. Completion does not imply timestamps are usable.",
        "EOF/count checks cannot rule out undetectable silent truncation.",
    ])
    run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    return summary
