"""Bounded-memory timestamp sampling and a two-row video contact sheet."""

import csv
from contextlib import closing
from fractions import Fraction
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import av
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from .video_inspection import InspectionError, inspect_video


SAMPLE_FIELDS = (
    "target_time_seconds", "actual_time_seconds", "frame_index", "image_filename",
)


def select_samples(records, first_index, last_index, first_time, last_time, count):
    """Nearest exact time per target; ties use lower index except pinned endpoints.

    records is a streaming iterable of (frame_index, Fraction presentation_time).
    The result has one mapping per target; several targets may select one frame.
    """
    if count < 1:
        raise ValueError("At least one sampling target is required.")
    targets = [first_time] if count == 1 else [
        first_time + (last_time - first_time) * Fraction(i, count - 1)
        for i in range(count)
    ]
    best = [None] * count
    for index, time in records:
        for position, target in enumerate(targets):
            # Always include the actual first/last valid frames, also on PTS ties.
            pinned = first_index if position == 0 else (
                last_index if position == count - 1 else None
            )
            if pinned is not None and index != pinned:
                continue
            candidate = (abs(time - target), index, time)
            if best[position] is None or candidate[:2] < best[position][:2]:
                best[position] = candidate
    if any(item is None for item in best):
        raise InspectionError("Insufficient valid frames for the requested targets.")
    return [
        {
            "target_time_seconds": float(target - first_time),
            "actual_time_seconds": float(item[2] - first_time),
            "frame_index": item[1],
            "image_filename": f"frame_{item[1]:06d}.png",
        }
        for target, item in zip(targets, best)
    ]


def timeline_records(timeline_path, database):
    """Read exact PTS/time_base from CSV, storing only intervals in disk SQLite."""
    previous = None
    with timeline_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["pts"] or not row["time_base"]:
                continue
            base = Fraction(row["time_base"])
            if base <= 0:
                continue
            time = int(row["pts"]) * base
            if previous is not None:
                database.execute("INSERT INTO intervals VALUES (?)", (float(time - previous),))
            previous = time
            yield int(row["frame_index"]), time


def interval_summary(database):
    """Exact order-statistic median of stored second values, with disk sorting."""
    count, minimum, maximum = database.execute(
        "SELECT COUNT(*), MIN(seconds), MAX(seconds) FROM intervals"
    ).fetchone()
    median = None
    if count:
        middle = [row[0] for row in database.execute(
            "SELECT seconds FROM intervals ORDER BY seconds LIMIT ? OFFSET ?",
            (2 if count % 2 == 0 else 1, (count - 1) // 2),
        )]
        median = sum(middle) / len(middle)
    return dict(interval_count=count, min_interval_seconds=minimum,
                median_interval_seconds=median, max_interval_seconds=maximum)


def export_images(source, output_dir, mappings, expected):
    """Second sequential decode: write one full-resolution PNG at a time."""
    selected = {row["frame_index"]: row for row in mappings}
    ordered = sorted(selected)
    slots = {index: position for position, index in enumerate(ordered)}
    columns = (len(ordered) + 1) // 2
    cell_width, image_height, label_height = 384, 240, 36
    cell_height = image_height + label_height
    font = ImageFont.load_default(size=17)
    saved = 0
    decoded = 0
    with Image.new("RGB", (columns * cell_width, 2 * cell_height), "#20252b") as sheet:
        draw = ImageDraw.Draw(sheet)
        with av.open(str(source), mode="r") as container:
            stream = container.streams.video[0]
            if stream.index != expected["selected_stream_index"]:
                raise InspectionError("Video stream changed between decoding passes.")
            for index, frame in enumerate(container.decode(stream)):
                decoded += 1
                if index not in selected:
                    continue
                row = selected[index]
                first = expected["first_valid_timestamp"]
                origin = first["pts"] * Fraction(first["time_base"])
                if (frame.pts is None or frame.time_base is None
                        or float(frame.pts * frame.time_base - origin) != row["actual_time_seconds"]):
                    raise InspectionError("Selected timestamps changed between decoding passes.")
                if frame.is_corrupt:
                    raise InspectionError(f"Selected frame {index} is corrupt.")
                # PyAV's Pillow conversion does not require NumPy.
                with frame.to_image() as picture:
                    with (output_dir / row["image_filename"]).open("xb") as file:
                        picture.save(file, format="PNG")
                    picture.thumbnail((cell_width - 16, image_height - 16), Image.Resampling.LANCZOS)
                    position = slots[index]
                    x = (position % columns) * cell_width
                    y = (position // columns) * cell_height
                    sheet.paste(picture, (x + (cell_width - picture.width) // 2,
                                          y + (image_height - picture.height) // 2))
                label = f"Frame {index} | t={row['actual_time_seconds']:.6f} s"
                draw.text((x + 8, y + image_height + 6), label, font=font, fill="white")
                saved += 1
        if decoded != expected["actual_decoded_count"] or saved != len(selected):
            raise InspectionError("Decoded frame count changed or selected frames were not exported.")
        with (output_dir / "contact_sheet.jpg").open("xb") as file:
            sheet.save(file, format="JPEG", quality=92)
    return saved


def preview_video(source: Path, output_dir: Path, samples: int = 8) -> dict:
    """Audit to temporary disk, sample timestamps, then export to a new directory."""
    if not 2 <= samples <= 12:
        raise InspectionError("--samples must be between 2 and 12 to include both endpoints.")
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise InspectionError(f"Input is missing, not a regular file, or empty: {source}")
    if output_dir.exists():
        raise InspectionError(f"Output directory already exists; choose a new directory: {output_dir}")
    # Exclusive directory creation protects M1.1 and all previous preview outputs.
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(
        source_path=str(source), pyav_version=av.__version__, pillow_version=pillow_version,
        requested_samples=samples, target_count=0, distinct_sample_count=0,
        exported_image_count=0, preview_status="failed", error=None, warnings=[],
        interval_count=0, min_interval_seconds=None, median_interval_seconds=None,
        max_interval_seconds=None,
    )
    try:
        with TemporaryDirectory(prefix="patient-preview-") as temporary:
            scratch = Path(temporary)
            audit = inspect_video(source, scratch / "audit")
            for key in ("actual_decoded_count", "missing_timestamp_count",
                        "duplicate_timestamp_count", "decreasing_timestamp_count",
                        "first_valid_timestamp", "last_valid_timestamp", "timestamp_span_seconds",
                        "decode_completed", "decode_status"):
                summary[key] = audit[key]
            summary["warnings"].extend(audit["warnings"])
            valid_count = audit["actual_decoded_count"] - audit["missing_timestamp_count"]
            with closing(sqlite3.connect(scratch / "intervals.sqlite")) as database:
                database.execute("PRAGMA temp_store=FILE")
                database.execute("PRAGMA cache_size=-2048")
                database.execute("CREATE TABLE intervals(seconds REAL NOT NULL)")
                records = timeline_records(scratch / "audit" / "timeline.csv", database)
                if valid_count and audit["decode_completed"] and not audit["decreasing_timestamp_count"]:
                    first, last = audit["first_valid_timestamp"], audit["last_valid_timestamp"]
                    mappings = select_samples(
                        records, first["frame_index"], last["frame_index"],
                        first["pts"] * Fraction(first["time_base"]),
                        last["pts"] * Fraction(last["time_base"]), min(samples, valid_count),
                    )
                else:
                    for _ in records:
                        pass
                    mappings = []
                summary.update(interval_summary(database))
            if not audit["decode_completed"]:
                raise InspectionError(f"Initial decode is not complete: {audit['error'] or audit['decode_status']}")
            if not valid_count:
                raise InspectionError("No valid presentation timestamps; no frames can be selected.")
            if audit["decreasing_timestamp_count"]:
                raise InspectionError("Decreasing timestamps prevent an ordered preview; see timing summary.")
            summary["target_count"] = len(mappings)
            summary["distinct_sample_count"] = len({row["frame_index"] for row in mappings})
            if summary["distinct_sample_count"] < samples:
                summary["warnings"].append(
                    f"Only {summary['distinct_sample_count']} distinct frames selected for {samples} requested; short stream or nearest-target collisions."
                )
            summary["exported_image_count"] = export_images(source, output_dir, mappings, audit)
            with (output_dir / "samples.csv").open("x", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=SAMPLE_FIELDS)
                writer.writeheader()
                writer.writerows(mappings)
            summary["preview_status"] = "complete"
    except (InspectionError, av.FFmpegError, OSError, ValueError, sqlite3.Error) as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["warnings"].append("Preview failed; any images present are partial output, not a complete preview.")
    summary["warnings"].append("Reported average FPS does not imply constant spacing; human visual review is pending.")
    with (output_dir / "timing_summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    return summary
