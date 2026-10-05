# P001_SE01_V001 — full-video extraction and target selection

## Input and configuration

Imported input: `data/raw/P001/P001_SE01/P001_SE01_V001.mp4`.
SHA-256 verified against import provenance:
`2ccad3c23be44d145c892d52eeccd0d3b4fea9d4b4dc98cf64299ec3cbd9a767`.
Inspection timeline: `outputs/fresh_20260924_130142_inspect/timeline.csv`.
The earlier inspection recorded the external original path; verified byte identity
links it to this imported copy. This task reads only the imported source.

Existing local YOLO11n-pose PT640, CPU float32, batch1, rect=False,
conf=.25, IoU=.7, max_det300; quality confidence=.5. Actual configuration and
package versions match the existing PT640 environment. No downloads or changes.

## Actual results and verification

- Extraction complete: **284 decoded/processed/written frames, 284 detections**;
  **0 zero-detection, 0 multiple-detection frames**. No corrupt frames or errors.
- Every extracted and selected frame index/PTS/time base/presentation/relative/delta
  value matches the inspection timeline. Missing/duplicate/decreasing timestamps=0.
  Last PTS=572176 at 19.072533333333332 s, time base 1/30000.
- Before association, Codex viewed the saved-prediction overlay at frame0. D0 box
  corresponds to the visible standing subject in a light shirt/dark shorts. Header
  partly obscures the upper head in that overlay; no joint-accuracy claim is made.
  Seed evidence is saved in the new association notes directory.
- Existing causal IoU selection: seed0/D0, min IoU=.1, ambiguity margin=.05,
  max elapsed gap=.5 s unchanged. **1 seeded + 283 matched, 0 unresolved**, no gap ranges.
  Seed choice was visual, not a rule to always choose D0 or rank by confidence.
- All selected raw people and 17 keypoints/independent quality flags are unchanged
  from source predictions, including points rejected by the gate.
- Saved-output review covers **0,40,81,121,162,202,242,283** from the preview CSV.
  There are no zero/multiple/unresolved cases to add. Review decode aligns all
  284 frames; eight full-size PNGs and sheet decode successfully; input hashes match.
- Both CLI commands exit0. Existing `half` deprecation warnings remain;
  actual dtype float32. No code changes or new regression-test claims.

## Assistant sample review supplied in the conversation

The assistant reviewed the uploaded selection sheet. Frames **0,40,81,121,162,202,242,283**
all show the yellow box selecting the visible standing subject, with one displayed
detection per sample. Major limb joints look broadly plausible at sheet resolution;
both feet are inside the image. Minor arm/posture changes are visible, especially
at sampled frames 121 and 162; unsampled frames between them are not validated.

Codex records this supplied account in `manual_observations.json` in the association
notes directory; this update is not an independent reinspection. These observations
are sample-level qualitative review, not full-video identity verification, measured
joint accuracy or confirmed temporal labels. Original execution-time review manifests
and verification reports retain their historical pending state; this note records
the subsequent review separately from predictions.

## Outputs and review record

- `outputs/P001_SE01_V001_poses_pt640_20261005/`: poses.jsonl, summary.json, run_info.json.
- `outputs/P001_SE01_V001_target_20261005/`: target_poses.jsonl, selection_summary.json.
- [Target-review sheet](../outputs/P001_SE01_V001_target_20261005/review/reviewed_selection_sheet.jpg)
  and eight full-resolution PNGs in the same directory.
- `outputs/P001_SE01_V001_association_notes_20261005/`: seed_candidates.json,
  seed_candidates_frame_000000.png, seed_verification.json, verification.json and manual_observations.json.

The historical temporal-preparation inventory remains unchanged as a snapshot:
its missing SE01 full-pose/target stages are now completed by these new outputs.
All four imported clips now have full-pose extraction and target outputs; this does
not establish final labels, joint accuracy or full-video identity ground truth.
SE01 eight-sample assistant review is recorded above; full-video identity/joint accuracy
and temporal labels remain unverified. M2 IN_PROGRESS.

## SE04 descriptive confirmation recorded separately

[User-confirmed approximate sequence](../outputs/P001_SE04_V001_temporal_notes_20261005/REVIEW_NOTES.md)
and `user_confirmed_sequence.json` preserve all ten rounded intervals exactly as
descriptions. No conversion to frame boundaries or final labels, no assumption that
turning intervals exclude stationary pauses. Exact boundaries remain pending.
All existing annotation files and SE02/SE03 occasion grouping remain unchanged.
No split assignment or training. Next: refine temporal boundaries from source playback
using PTS, recording uncertainty explicitly; the sheet review does not establish them.
