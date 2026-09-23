"""Review selections after association; review indices cannot influence selection."""

from itertools import zip_longest
from pathlib import Path

import av
from PIL import ImageDraw, ImageFont

from .pose_preview import sha256_file
from .pose_review import annotate_review, read_records, review_sheet, score, select_boundary_frames
from .target_selection import valid_box, write_json
from .video_inspection import COLUMNS, InspectionError, TimelineAudit


def target_review(video, poses, output, review_frames):
    video, poses, output = (Path(p).resolve() for p in (video, poses, output))
    target = output / "target_poses.jsonl"
    review_dir = output / "review"
    if review_dir.exists():
        raise InspectionError("Selection review directory exists; refusing to replace it")
    if any(type(i) is not int or i < 0 for i in review_frames):
        raise InspectionError("Review indices must be nonnegative integers")
    unresolved = [(r["frame_index"], r["presentation_time_seconds"]) for r in read_records(target)
                  if r["selected_source_detection_index"] is None]
    valid_times = [(i, t) for i, t in unresolved if t is not None]
    chosen = select_boundary_frames(valid_times) if valid_times else [
        dict(frame_index=i, target_presentation_time_seconds=None) for i, _ in unresolved[:6]]
    unresolved_indices = [r["frame_index"] for r in chosen]
    wanted = set(review_frames) | set(unresolved_indices)
    review_dir.mkdir()
    hashes = {str(p): sha256_file(p) for p in (video, poses, target)}
    manifest = dict(status="incomplete", human_verdict="pending", input_sha256=hashes,
        requested_review_frames=sorted(set(review_frames)), unresolved_frame_count=len(unresolved),
        unresolved_review_frames=unresolved_indices, selected_frames=[], errors=[], aligned_frame_count=0,
        note="Yellow box = associated target; original frame-local detection colors retained. No review indices affect selection.",
        unresolved_sampling="Up to six distinct frames distributed in valid PTS; if none valid, first six by source index")
    manifest_path = review_dir / "review_manifest.json"
    write_json(manifest_path, manifest)
    audit = TimelineAudit()
    try:
        with av.open(str(video)) as container:
            first = next(read_records(poses))
            if Path(first["source_path"]).resolve() != video:
                raise InspectionError("Review video path differs from prediction source")
            stream = next(s for s in container.streams.video if s.index == first["stream_index"])
            for frame, record, selection in zip_longest(container.decode(stream), read_records(poses), read_records(target)):
                if any(item is None for item in (frame, record, selection)):
                    raise InspectionError("Video/prediction/selection frame counts differ")
                timing = audit.observe(frame.pts, frame.time_base)
                if (any(timing[k] != record[k] or record[k] != selection[k] for k in COLUMNS)
                        or (frame.width, frame.height) != (record["width"], record["height"])
                        or record["stream_index"] != stream.index or frame.is_corrupt):
                    raise InspectionError(f"Review source alignment failed at frame {timing['frame_index']}")
                manifest["aligned_frame_count"] += 1
                if record["frame_index"] not in wanted:
                    continue
                record.update(annotated_image=f"frame_{record['frame_index']:06d}.png", selection=selection,
                    review_reasons=(["requested visual comparison"] if record["frame_index"] in review_frames else [])
                        + (["unresolved association"] if record["frame_index"] in unresolved_indices else []))
                with frame.to_image() as picture:
                    # Invalid boxes remain in JSON but cannot safely be drawn as rectangles.
                    display = {**record, "people": [p for p in record["people"]
                        if valid_box(p["box_xyxy"], record["width"], record["height"]) is None]}
                    record["undrawable_detection_indices"] = [p["detection_index"] for p in record["people"]
                        if valid_box(p["box_xyxy"], record["width"], record["height"]) is not None]
                    record["display_issues"] = annotate_review(picture, display)
                    draw = ImageDraw.Draw(picture)
                    chosen_person = selection["selected_person"]
                    if chosen_person is not None:
                        draw.rectangle(chosen_person["box_xyxy"], outline="#ffe44d", width=5)
                    label = f"TARGET D{selection['selected_source_detection_index']}" if chosen_person is not None else "TARGET NULL"
                    text = f"{label} | {selection['status']} | IoU={score(selection['best_iou'])} | separation={score(selection['iou_separation'])}"
                    draw.rectangle((0, 0, picture.width, 36), fill="#10151b")
                    draw.text((8, 2), text, fill="#ffe44d", font=ImageFont.load_default(size=27))
                    with (review_dir / record["annotated_image"]).open("xb") as handle:
                        picture.save(handle, format="PNG")
                manifest["selected_frames"].append(record)
        if {r["frame_index"] for r in manifest["selected_frames"]} != wanted:
            raise InspectionError("A requested review frame was not found")
        for filename, indices, title in (
            ("reviewed_selection_sheet.jpg", review_frames, "Requested frames | yellow box = associated target"),
            ("unresolved_selection_sheet.jpg", unresolved_indices, "Unresolved association | no missing poses filled"),
        ):
            review_sheet(review_dir, filename, [r for r in manifest["selected_frames"] if r["frame_index"] in indices], title)
        if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
            raise InspectionError("An input changed during selection review")
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
    write_json(manifest_path, manifest)
    return manifest
