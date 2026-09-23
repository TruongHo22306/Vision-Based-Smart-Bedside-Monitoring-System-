"""Review saved M2.2 predictions without importing or running an inference model."""

import colorsys
from itertools import zip_longest
import json
import math
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from .pose_preview import EDGES, sha256_file
from .video_inspection import COLUMNS, InspectionError, TimelineAudit

BOUNDARY_JOINTS = (13, 14, 15, 16)


def read_records(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def boundary_points(person):
    return [p for p in person["keypoints"]
            if p["index"] in BOUNDARY_JOINTS and p["at_or_outside_boundary"]]


def select_boundary_frames(candidates, limit=6):
    """Nearest unused candidates to evenly spaced actual PTS targets; stable ties."""
    if not candidates:
        return []
    if any(t is None or not math.isfinite(t) for _, t in candidates):
        raise InspectionError("Boundary sampling requires finite presentation timestamps")
    ordered = sorted(candidates, key=lambda item: (item[1], item[0]))
    count = min(limit, len(ordered))
    remaining = dict(ordered)
    selected = []
    for i in range(count):
        target = ordered[0][1] if count == 1 else (
            ordered[0][1] + (ordered[-1][1] - ordered[0][1]) * i / (count - 1)
        )
        index = min(remaining, key=lambda k: (abs(remaining[k] - target), k))
        selected.append(dict(frame_index=index, target_presentation_time_seconds=target))
        del remaining[index]
    return sorted(selected, key=lambda item: item["frame_index"])


def scan_predictions(poses, summary, source):
    multi, boundary = [], []
    count = 0
    for row in read_records(poses):
        if (row["frame_index"] != count or row["inference_status"] != "complete"
                or row["person_count"] != len(row["people"])
                or Path(row["source_path"]).resolve() != source):
            raise InspectionError(f"Invalid/incomplete prediction record at frame {count}")
        if row["person_count"] > 1:
            multi.append(count)
        if any(boundary_points(p) for p in row["people"]):
            boundary.append((count, row["presentation_time_seconds"]))
        for index, person in enumerate(row["people"]):
            if (person["detection_index"] != index or len(person["keypoints"]) != 17
                    or [p["index"] for p in person["keypoints"]] != list(range(17))):
                raise InspectionError(f"Invalid detection/keypoint indices at frame {count}")
        count += 1
    if (not summary["completed"] or summary["status"] != "complete"
            or any(summary[k] != count for k in
                   ("decoded_frame_count", "processed_frame_count", "written_record_count"))
            or summary["multiple_detection_frame_count"] != len(multi)):
        raise InspectionError("JSONL counts/completion do not match summary.json")
    return count, multi, boundary


def detection_color(index):
    # Distinct hues within a frame; colors/indices do not establish identity over time.
    rgb = colorsys.hsv_to_rgb((0.36 + index * 0.61803398875) % 1, .85, 1)
    return tuple(round(channel * 255) for channel in rgb)


def gate_edges(points):
    return [(a, b) for a, b in EDGES
            if points[a]["passes_quality_gate"] and points[b]["passes_quality_gate"]]


def clipped_marker(point, size):
    """Return a display-only position, never modify raw coordinates or flags."""
    x, y = point["x"], point["y"]
    if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
        return None
    return (min(max(x, 0), size[0] - 1), min(max(y, 0), size[1] - 1))


def score(value):
    return "null" if value is None else f"{value:.3f}"


def legend_lines(record):
    lines = []
    for person in record["people"]:
        index = person["detection_index"]
        passed = sum(p["passes_quality_gate"] for p in person["keypoints"])
        color = detection_color(index)
        lines.append((f"D{index} | box conf={score(person['box_confidence'])} | gate={passed}/17", color))
        flagged = boundary_points(person)
        for start in range(0, len(flagged), 2):
            names = "; ".join(f"{p['name']} c={score(p['confidence'])}"
                              for p in flagged[start:start + 2])
            lines.append((f"  X D{index}: {names}", "#ff80e5"))
    return lines


def annotate_review(picture, record):
    draw = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=18)
    issues = []
    for person in record["people"]:
        index, box = person["detection_index"], person["box_xyxy"]
        color = detection_color(index)
        if all(v is not None and math.isfinite(v) for v in box):
            draw.rectangle(box, outline=color, width=3)
            position = clipped_marker(dict(x=box[0], y=box[1]), picture.size)
            text = f"D{index} conf={score(person['box_confidence'])} gate={sum(p['passes_quality_gate'] for p in person['keypoints'])}/17"
            x = min(position[0], max(0, picture.width - draw.textlength(text, font=font) - 8))
            # Keep box labels below the 56-pixel frame header, including top-clipped boxes.
            y = min(max(56, position[1]), max(0, picture.height - 24))
            draw.rectangle((x, y, x + draw.textlength(text, font=font) + 6, y + 23), fill="#10151b")
            draw.text((x + 3, y), text, fill=color, font=font)
        else:
            issues.append(f"D{index}: nonfinite box cannot be drawn; raw box retained in manifest")
        points = person["keypoints"]
        for a, b in gate_edges(points):
            draw.line((points[a]["x"], points[a]["y"], points[b]["x"], points[b]["y"]),
                      fill=color, width=3)
        for point in points:
            if point["passes_quality_gate"]:
                x, y = point["x"], point["y"]
                draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
    # Boundary markers have a different shape/color and never become skeleton endpoints.
    marker_labels = []
    for person in record["people"]:
        for point in boundary_points(person):
            position = clipped_marker(point, picture.size)
            if position is None:
                issues.append(f"D{person['detection_index']} {point['name']}: flagged but nonfinite coordinates")
                continue
            x, y = position
            draw.line((x-8, y-8, x+8, y+8), fill="#ff80e5", width=4)
            draw.line((x-8, y+8, x+8, y-8), fill="#ff80e5", width=4)
            text = f"X D{person['detection_index']} {point['name']} c={score(point['confidence'])}"
            label_width = draw.textlength(text, font=font) + 8
            left = min(max(0, x + 12), max(0, picture.width - label_width))
            top = min(max(0, y - 26), max(0, picture.height - 24))
            while top > 48 and any(left < r[2] and left + label_width > r[0]
                                  and top < r[3] and top + 24 > r[1] for r in marker_labels):
                top -= 24
            bounds = (left, top, left + label_width, top + 23)
            marker_labels.append(bounds)
            draw.rectangle(bounds, fill="#10151b")
            draw.text((left + 4, top), text, font=font, fill="#ff80e5")
    time = record["presentation_time_seconds"]
    title = f"Frame {record['frame_index']} | PTS time={score(time)} s | persons={record['person_count']}"
    lines = [(title, "white"), ("D = frame-local index, NOT tracking ID; X = boundary (display clipped)", "white")]
    draw.rectangle((0, 0, min(picture.width, 760), 24 * len(lines) + 8), fill="#10151b")
    for i, (text, color) in enumerate(lines):
        draw.text((8, 4 + i*24), text, fill=color, font=font)
    return issues


def review_sheet(output, filename, records, title):
    columns = min(3, max(1, len(records)))
    rows = max(1, math.ceil(len(records) / columns))
    width, photo_height = 640, 360
    caption_height = 58 + 24 * max((len(legend_lines(r)) for r in records), default=1)
    cell_height = photo_height + caption_height
    with Image.new("RGB", (columns * width, 72 + rows * cell_height), "#10151b") as sheet:
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=18)
        draw.text((12, 8), title, font=ImageFont.load_default(size=23), fill="white")
        draw.text((12, 40), "D: frame-local, NOT tracking ID | X: boundary, display clipped | human verdict: PENDING", font=font, fill="white")
        if not records:
            draw.text((12, 100), "No matching frames in saved predictions", font=font, fill="white")
        for slot, record in enumerate(records):
            x, y = (slot % columns)*width, 72 + (slot // columns)*cell_height
            with Image.open(output / record["annotated_image"]) as picture:
                picture.thumbnail((width-8, photo_height-8), Image.Resampling.LANCZOS)
                sheet.paste(picture, (x+(width-picture.width)//2, y+(photo_height-picture.height)//2))
            title_line = f"Frame {record['frame_index']} | PTS={record['presentation_time_seconds']:.6f} s | n={record['person_count']}"
            draw.text((x+8, y+photo_height), title_line, font=font, fill="white")
            for i, (text, color) in enumerate(legend_lines(record)):
                draw.text((x+8, y+photo_height+28+i*24), text, font=font, fill=color)
        with (output / filename).open("xb") as handle:
            sheet.save(handle, format="JPEG", quality=95)


def review_poses(source: Path, poses: Path, summary_path: Path, output: Path):
    source, poses, summary_path, output = (p.expanduser().resolve() for p in (source, poses, summary_path, output))
    if output.exists():
        raise InspectionError("Output directory exists; choose a new directory")
    inputs = {str(p): sha256_file(p) for p in (source, poses, summary_path)}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if inputs[str(source)] != summary["source_sha256"]:
        raise InspectionError("Source video hash does not match extraction summary")
    count, multi, boundary = scan_predictions(poses, summary, source)
    chosen = select_boundary_frames(boundary)
    boundary_indices = [s["frame_index"] for s in chosen]
    wanted = set(multi) | set(boundary_indices)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=1, task="M2.3", export_status="incomplete", human_verdict="pending",
        input_sha256=inputs, prediction_count=count, reported_multiple_count=summary["multiple_detection_frame_count"],
        actual_multiple_count=len(multi), multiple_frame_indices=multi,
        boundary_candidate_count=len(boundary), boundary_selection=chosen,
        boundary_selection_method="Nearest unused frame to 6 evenly spaced presentation-time targets across eligible span; ties choose lower index",
        gate_source="Saved passes_quality_gate flags, unchanged; no threshold adjustment or inference",
        detection_index_scope="Frame-local, not persistent tracking IDs",
        display_note="Colored lines use passing endpoints only. Pink X marks finite flagged knee/ankle coordinates; outside coordinates clip only for display. Raw values remain unchanged.",
        limitation="Confidence/gate is not accuracy or proof of visibility; no false-positive verdict inferred",
        selected_frames=[], alignment_checked_frame_count=0, errors=[],
    )
    manifest_path = output / "review_manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    save_manifest()
    audit = TimelineAudit()
    try:
        with av.open(str(source)) as container:
            stream = next(s for s in container.streams.video if s.index == summary["selected_stream_index"])
            for frame, record in zip_longest(container.decode(stream), read_records(poses)):
                if frame is None or record is None:
                    raise InspectionError("Decoded frame count and JSONL length differ")
                timeline = audit.observe(frame.pts, frame.time_base)
                if (any(timeline[k] != record[k] for k in COLUMNS)
                        or (frame.width, frame.height) != (record["width"], record["height"])
                        or stream.index != record["stream_index"] or frame.is_corrupt):
                    raise InspectionError(f"Source alignment mismatch/corrupt frame {timeline['frame_index']}")
                manifest["alignment_checked_frame_count"] += 1
                if record["frame_index"] not in wanted:
                    continue
                reasons = []
                if record["frame_index"] in multi:
                    reasons.append("person_count > 1: include every matching frame")
                if record["frame_index"] in boundary_indices:
                    reasons.append("knee/ankle at_or_outside_boundary: time-distributed sample")
                record.update(selection_reasons=reasons, human_verdict="pending",
                              annotated_image=f"frame_{record['frame_index']:06d}.png")
                for person in record["people"]:
                    person.update(human_verdict="pending", display_color=detection_color(person["detection_index"]),
                                  passing_keypoint_count=sum(p["passes_quality_gate"] for p in person["keypoints"]))
                with frame.to_image() as picture:
                    record["display_issues"] = annotate_review(picture, record)
                    with (output / record["annotated_image"]).open("xb") as handle:
                        picture.save(handle, format="PNG")
                manifest["selected_frames"].append(record)
        if {r["frame_index"] for r in manifest["selected_frames"]} != wanted:
            raise InspectionError("Not every selected frame was exported")
        for name, indices, title in (
            ("multi_detection_sheet.jpg", multi, "All multiple-detection frames | saved predictions"),
            ("boundary_review_sheet.jpg", boundary_indices, "Knee / ankle boundary review | saved predictions"),
        ):
            review_sheet(output, name, [r for r in manifest["selected_frames"] if r["frame_index"] in indices], title)
        if any(sha256_file(Path(path)) != digest for path, digest in inputs.items()):
            raise InspectionError("An input changed during review export")
        manifest["export_status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
    save_manifest()
    return manifest
