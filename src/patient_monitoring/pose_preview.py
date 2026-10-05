"""CPU-only YOLO11n pose smoke test on CSV-referenced preview PNGs."""

import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import urllib.request

from PIL import Image, ImageDraw, ImageFont

from .video_inspection import InspectionError

MODEL_NAME = "yolo11n-pose.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/" + MODEL_NAME
MODEL_SHA256 = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
EDGES = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
         (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
         (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
PREDICT_CONFIG = dict(device="cpu", imgsz=640, batch=1, rect=False, half=False,
                      conf=0.25, iou=0.7, max_det=300, augment=False,
                      save=False, verbose=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_samples(input_dir):
    """Validate up to twelve CSV rows; never discover images by directory scanning."""
    records = []
    seen = {}
    with (input_dir / "samples.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"frame_index", "actual_time_seconds", "image_filename"}
        if not required.issubset(reader.fieldnames or []):
            raise InspectionError("samples.csv lacks frame_index/actual_time_seconds/image_filename.")
        for csv_index, row in enumerate(reader):
            if csv_index >= 12:
                raise InspectionError("This smoke test accepts up to 12 preview CSV rows.")
            frame_index = int(row["frame_index"])
            timestamp = float(row["actual_time_seconds"])
            path = (input_dir / row["image_filename"]).resolve()
            if (frame_index < 0 or not math.isfinite(timestamp) or not path.is_relative_to(input_dir)
                    or path.suffix.lower() != ".png" or not path.is_file()):
                raise InspectionError(f"Invalid preview row {csv_index + 2}.")
            identity = (frame_index, row["actual_time_seconds"])
            if path in seen:
                if seen[path][0] != identity:
                    raise InspectionError("One PNG has conflicting frame/time mappings.")
                records[seen[path][1]]["csv_row_indices"].append(csv_index)
                continue
            if any(record["frame_index"] == frame_index for record in records):
                raise InspectionError("One frame index maps to different PNGs.")
            with Image.open(path) as picture:
                if picture.format != "PNG":
                    raise InspectionError(f"Referenced input is not PNG: {path}")
                width, height = picture.size
                picture.verify()
            seen[path] = (identity, len(records))
            records.append(dict(
                source_image=str(path), image_filename=row["image_filename"],
                frame_index=frame_index, actual_time_seconds=timestamp,
                actual_time_seconds_text=row["actual_time_seconds"],
                csv_row_indices=[csv_index], width=width, height=height,
                source_sha256=sha256_file(path),
            ))
    if not records:
        raise InspectionError("samples.csv contains no images.")
    return records


def ensure_model(models_dir):
    """Use only official pinned weights; verify hash before deserializing."""
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / MODEL_NAME
    if not destination.exists():
        partial = models_dir / (MODEL_NAME + ".part")
        with partial.open("xb") as file:
            with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
                shutil.copyfileobj(response, file)
        if sha256_file(partial) != MODEL_SHA256:
            raise InspectionError("Downloaded weights SHA-256 mismatch; model not loaded.")
        partial.rename(destination)
    if sha256_file(destination) != MODEL_SHA256:
        raise InspectionError("Local model SHA-256 mismatch; refusing to load or overwrite it.")
    return destination


def serialize_people(result):
    """Preserve unrounded original-image x/y/confidence from keypoints.data."""
    if result.boxes is None:
        raise InspectionError("Pose result has no boxes field.")
    boxes = result.boxes.xyxy.cpu().tolist()
    if not boxes:
        return []
    confidences = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    if result.keypoints is None:
        raise InspectionError("Detected boxes have no keypoints; refusing to invent joints.")
    keypoints = result.keypoints.data.cpu().tolist()
    if not (len(boxes) == len(confidences) == len(classes) == len(keypoints)):
        raise InspectionError("Box/keypoint counts do not match.")
    people = []
    for index, (box, confidence, class_id, joints) in enumerate(zip(boxes, confidences, classes, keypoints)):
        if len(joints) != 17 or any(len(point) != 3 for point in joints):
            raise InspectionError("Expected exactly 17 x/y/confidence keypoints per person.")
        people.append(dict(
            detection_index=index, class_id=int(class_id),
            box_xyxy=box, box_confidence=confidence,
            keypoints=[dict(index=i, name=KEYPOINT_NAMES[i], x=point[0], y=point[1],
                            confidence=point[2]) for i, point in enumerate(joints)],
        ))
    return people


def visible_edges(keypoints, threshold):
    return [(a, b) for a, b in EDGES
            if keypoints[a]["confidence"] >= threshold
            and keypoints[b]["confidence"] >= threshold]


def annotate(picture, people, threshold, label=None):
    """Visualization threshold affects drawing only, never serialized predictions."""
    draw = ImageDraw.Draw(picture)
    for person in people:
        draw.rectangle(person["box_xyxy"], outline="#ffbb33", width=3)
        points = person["keypoints"]
        for a, b in visible_edges(points, threshold):
            draw.line((points[a]["x"], points[a]["y"], points[b]["x"], points[b]["y"]),
                      fill="#23e6a8", width=3)
        for point in points:
            if point["confidence"] >= threshold:
                x, y = point["x"], point["y"]
                draw.ellipse((x-4, y-4, x+4, y+4), fill="#ff4477")
    if label:
        font = ImageFont.load_default(size=18)
        bounds = draw.textbbox((8, 8), label, font=font)
        draw.rectangle((0, 0, bounds[2]+8, bounds[3]+8), fill="#20252b")
        draw.text((8, 8), label, font=font, fill="white")


def pose_contact_sheet(output_dir, records):
    columns = (len(records) + 1) // 2
    cell_width, image_height, cell_height = 384, 240, 290
    font = ImageFont.load_default(size=16)
    with Image.new("RGB", (columns * cell_width, 2 * cell_height), "#20252b") as sheet:
        draw = ImageDraw.Draw(sheet)
        for slot, record in enumerate(records):
            x, y = (slot % columns) * cell_width, (slot // columns) * cell_height
            with Image.open(output_dir / record["annotated_image"]) as picture:
                picture.thumbnail((cell_width-16, image_height-16), Image.Resampling.LANCZOS)
                sheet.paste(picture, (x+(cell_width-picture.width)//2, y+(image_height-picture.height)//2))
            draw.text((x+8, y+image_height),
                      f"Frame {record['frame_index']} | t={record['actual_time_seconds']:.6f} s\n"
                      f"Detected persons: {record['person_count']}", font=font, fill="white")
        with (output_dir / "pose_contact_sheet.jpg").open("xb") as file:
            sheet.save(file, format="JPEG", quality=92)


def pose_preview(input_dir: Path, output_dir: Path, keypoint_confidence: float = 0.5):
    if not math.isfinite(keypoint_confidence) or not 0 <= keypoint_confidence <= 1:
        raise InspectionError("--keypoint-confidence must be between 0 and 1.")
    input_dir, output_dir = input_dir.expanduser().resolve(), output_dir.expanduser().resolve()
    records = read_samples(input_dir)
    csv_hash = sha256_file(input_dir / "samples.csv")
    if output_dir.exists():
        raise InspectionError("Output directory already exists; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    run = dict(
        task="M2.1", status="failed", error=None,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        package_versions={d.metadata['Name']: d.version for d in metadata.distributions()},
        model=dict(filename=MODEL_NAME, official_url=MODEL_URL, sha256=None),
        config={**PREDICT_CONFIG, "keypoint_display_confidence": keypoint_confidence},
        input_csv=str(input_dir / "samples.csv"), input_csv_sha256=csv_hash,
        expected_image_count=len(records), processed_image_count=0,
        human_pose_review="pending", warnings=[
            "Confidence is a model score, not measured accuracy.",
            "Smoke test only; YOLO11n Pose is not the final model choice.",
            "Pilot includes partial-body frames; no missing joints are synthesized.",
        ],
    )
    results = []
    try:
        weights = ensure_model(models_dir)
        run["model"].update(path=str(weights), sha256=sha256_file(weights))
        # Isolate library settings/caches in this fresh workspace before import.
        os.environ["YOLO_CONFIG_DIR"] = str(models_dir / "ultralytics_settings")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["MPLCONFIGDIR"] = str(models_dir / "matplotlib_settings")
        from ultralytics import YOLO
        model = YOLO(str(weights), task="pose")
        for source in records:
            with Image.open(source["source_image"]) as original, original.convert("RGB") as picture:
                prediction = model.predict(source=picture, **PREDICT_CONFIG)[0]
                if tuple(prediction.orig_shape) != (source["height"], source["width"]):
                    raise InspectionError("Prediction dimensions do not match the source PNG.")
                people = serialize_people(prediction)
                record = {**source, "person_count": len(people),
                          "detection_status": "detected" if people else "zero_detections",
                          "people": people,
                          "annotated_image": f"pose_frame_{source['frame_index']:06d}.png"}
                annotate(picture, people, keypoint_confidence,
                         f"Frame {source['frame_index']} | t={source['actual_time_seconds']:.6f} s | Detected persons: {len(people)}")
                with (output_dir / record["annotated_image"]).open("xb") as file:
                    picture.save(file, format="PNG")
            if sha256_file(Path(source["source_image"])) != source["source_sha256"]:
                raise InspectionError("Source image changed during inference.")
            results.append(record)
            run["processed_image_count"] = len(results)
        if sha256_file(input_dir / "samples.csv") != csv_hash:
            raise InspectionError("samples.csv changed during inference.")
        pose_contact_sheet(output_dir, results)
        first_parameter = next(model.model.parameters())
        run["actual_model_device"] = str(first_parameter.device)
        run["actual_model_dtype"] = str(first_parameter.dtype)
        run["effective_predict_config"] = {
            key: getattr(model.predictor.args, key)
            for key in PREDICT_CONFIG
            if hasattr(model.predictor.args, key)
        }
        run["runtime_unreported_config"] = [
            key for key in PREDICT_CONFIG if not hasattr(model.predictor.args, key)
        ]
        run["status"] = "complete"
    except Exception as exc:
        # Keep partial records and a precise failure; never present partial inference as success.
        run["error"] = f"{type(exc).__name__}: {exc}"
    run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run["zero_detection_image_count"] = sum(r["person_count"] == 0 for r in results)
    for filename, data in (("poses.json", {"images": results}), ("run_info.json", run)):
        with (output_dir / filename).open("x", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
    return run
