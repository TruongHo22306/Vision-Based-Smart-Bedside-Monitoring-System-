"""Sequential video/timestamp audit; no frame storage or FPS-derived timing."""

import csv
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

import av


class InspectionError(Exception):
    """Invalid input or output that prevents an inspection report."""


COLUMNS = (
    "frame_index", "pts", "time_base", "presentation_time_seconds",
    "relative_time_seconds", "delta_seconds",
)


def rational_text(value: Fraction | None) -> str | None:
    return None if value is None else f"{value.numerator}/{value.denominator}"


@dataclass
class TimelineAudit:
    """Compare exact rational times, retaining only first and previous valid time."""

    decoded_count: int = 0
    missing_count: int = 0
    duplicate_count: int = 0
    decreasing_count: int = 0
    first_time: Fraction | None = None
    previous_time: Fraction | None = None
    first_timestamp: dict | None = None
    last_timestamp: dict | None = None

    def observe(self, pts: int | None, time_base: Fraction | None) -> dict:
        row = dict.fromkeys(COLUMNS)
        row.update(frame_index=self.decoded_count, pts=pts,
                   time_base=rational_text(time_base))
        self.decoded_count += 1
        if pts is None or time_base is None or time_base <= 0:
            self.missing_count += 1
            return row

        current = pts * time_base
        delta = None if self.previous_time is None else current - self.previous_time
        if delta is not None:
            self.duplicate_count += int(delta == 0)
            self.decreasing_count += int(delta < 0)
        if self.first_time is None:
            self.first_time = current
        row.update(
            presentation_time_seconds=float(current),
            relative_time_seconds=float(current - self.first_time),
            delta_seconds=None if delta is None else float(delta),
        )
        timestamp = {key: row[key] for key in COLUMNS[:4]}
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        self.previous_time = current
        return row

    def summary(self) -> dict:
        span = None
        if self.first_time is not None and self.previous_time is not None:
            span = float(self.previous_time - self.first_time)
        return {
            "actual_decoded_count": self.decoded_count,
            "missing_timestamp_count": self.missing_count,
            "duplicate_timestamp_count": self.duplicate_count,
            "decreasing_timestamp_count": self.decreasing_count,
            "first_valid_timestamp": self.first_timestamp,
            "last_valid_timestamp": self.last_timestamp,
            "timestamp_span_seconds": span,
        }


def inspect_video(source: Path, output_dir: Path) -> dict:
    """Write fresh reports and return metadata; incomplete decoding is explicit."""
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source.is_file():
        raise InspectionError(f"Input is not an existing regular file: {source}")
    if source.stat().st_size == 0:
        raise InspectionError(f"Input file is empty: {source}")

    metadata_path = output_dir / "metadata.json"
    timeline_path = output_dir / "timeline.csv"
    for target in (metadata_path, timeline_path):
        # Refuse existing files, symlinks and hardlinks, including the input itself.
        if target.resolve() == source or target.exists() or target.is_symlink():
            raise InspectionError(f"Refusing to overwrite output/input: {target}")
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = TimelineAudit()
    metadata = {
        "source_path": str(source),
        "pyav_version": av.__version__,
        "selected_stream_index": None,
        "codec": None,
        "width": None,
        "height": None,
        "reported_frame_rate": None,
        "reported_frame_rate_fraction": None,
        "reported_frame_count": None,
        "container_duration_seconds": None,
        "stream_duration_seconds": None,
        "stream_time_base": None,
        "decode_status": "failed",
        "decode_completed": False,
        "error": None,
        "corrupt_frame_count": 0,
        "warnings": [],
    }
    warnings = metadata["warnings"]
    # Exclusive creation also protects against targets appearing after validation.
    with metadata_path.open("x", encoding="utf-8") as metadata_file, \
            timeline_path.open("x", encoding="utf-8", newline="") as timeline_file:
        writer = csv.DictWriter(timeline_file, fieldnames=COLUMNS)
        writer.writeheader()
        try:
            with av.open(str(source), mode="r") as container:
                if container.duration is not None and container.duration >= 0:
                    metadata["container_duration_seconds"] = (
                        container.duration / av.time_base
                    )
                if not container.streams.video:
                    metadata["decode_status"] = "no_video_stream"
                    metadata["error"] = "Input contains no video stream."
                else:
                    stream = container.streams.video[0]
                    context = stream.codec_context
                    rate = stream.average_rate
                    metadata.update(
                        selected_stream_index=stream.index,
                        codec=context.name or None,
                        width=context.width or None,
                        height=context.height or None,
                        reported_frame_rate=(
                            float(rate) if rate is not None and rate > 0 else None
                        ),
                        reported_frame_rate_fraction=(
                            rational_text(rate) if rate is not None and rate > 0
                            else None
                        ),
                        reported_frame_count=stream.frames or None,
                        stream_time_base=rational_text(stream.time_base),
                    )
                    if (stream.duration is not None and stream.duration >= 0
                            and stream.time_base is not None):
                        metadata["stream_duration_seconds"] = float(
                            stream.duration * stream.time_base
                        )
                    for frame in container.decode(stream):
                        writer.writerow(audit.observe(frame.pts, frame.time_base))
                        metadata["corrupt_frame_count"] += int(frame.is_corrupt)
                    metadata["decode_status"] = "complete"
                    metadata["decode_completed"] = True
        except (av.FFmpegError, OSError, ValueError) as exc:
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            metadata["decode_status"] = (
                "incomplete" if audit.decoded_count else "failed"
            )
            metadata["decode_completed"] = False
            warnings.append("Opening, decoding or writing the timeline failed; output may be partial.")

        if metadata["decode_completed"]:
            if audit.decoded_count == 0:
                warnings.append("No frames decoded; completion cannot be verified.")
                metadata["decode_completed"] = False
            reported = metadata["reported_frame_count"]
            if reported is not None and reported != audit.decoded_count:
                warnings.append("Decoded count differs from the reported count; reported metadata may be inaccurate.")
                if audit.decoded_count < reported:
                    metadata["decode_completed"] = False
            if metadata["corrupt_frame_count"]:
                warnings.append("Decoder returned corrupt frames; a clean complete decode is not verified.")
                metadata["decode_completed"] = False
            if not metadata["decode_completed"]:
                metadata["decode_status"] = "incomplete"

        for count, message in (
            (audit.missing_count, "Frames with missing PTS or missing/nonpositive time_base were preserved without inferred timing."),
            (audit.duplicate_count, "Duplicate presentation times relative to the preceding valid timestamp were found."),
            (audit.decreasing_count, "Decreasing presentation times relative to the preceding valid timestamp were found."),
        ):
            if count:
                warnings.append(message)
        if audit.first_time is None:
            warnings.append("No valid presentation timestamp is available.")
        warnings.extend([
            "Timestamp span is last valid time minus first valid time, not full video duration; the final frame display duration is not added.",
            "File presentation timestamps and reported durations cannot prove the original recording's real-world duration.",
            "Completion means the decoder reached EOF without detected errors or a reported count shortfall; silent truncation may be undetectable.",
        ])
        metadata.update(audit.summary())
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2, allow_nan=False)
        metadata_file.write("\n")
    return metadata
