# P001_SE03_V001 — extraction and experimental target review

## Conversation sample review

The assistant review supplied in the conversation reports: all eight preview
boxes cover the lying subject, with no extra detections. Feet are near the left
image edge; leg skeletons look broadly plausible. Arm/shoulder/hip estimates vary
despite little visible posture change, notably frame 419; far-side arm occlusion
limits review. These are qualitative observations, not ground truth or accuracy.
Codex records this account separately from predictions in
`outputs/P001_SE03_V001_association_notes_20260929/manual_observations.json`.
It does not drive association or establish correctness outside the eight samples.

### Uploaded selected-target sheet review

The assistant reviewed the uploaded selected-target sheet in the conversation.
As supplied by the user, yellow boxes select the lying subject in all eight
displayed frames (0,84,168,252,336,419,503,587), with no visible target switch in
those samples. Arm/torso joint placement remains uncertain. **Full-resolution
frame 419 has not been reviewed by the assistant**; its sheet thumbnail is within
the sample review scope. This is qualitative sample review, not ground truth,
full-video identity validation or joint accuracy. Codex records the supplied
account, without independent reinspection in this update. The structured entry is
`selection_sheet_review` in `manual_observations.json`; earlier observations remain.

## Execution and seed

- Input: `data/raw/P001/P001_SE03/P001_SE03_V001.mp4`; provenance and same-occasion
  grouping with SE02 remain as documented in [import note](LYING_RECORDING_IMPORT_20260929.md).
- Existing local YOLO11n-pose PT640: CPU float32, batch=1, rect=False,
  conf=.25, IoU=.7, keypoint quality confidence=.5; no configuration changes.
- Extraction completed: **588 decoded/processed/written frames, 588 detections**;
  **0 zero-detection and 0 multiple-detection frames**, no corrupt frames.
  All 588 indices/PTS match the prior inspect timeline; no missing, duplicate or
  decreasing timestamps. PTS span 19.8245 s, time base 1/30000, last PTS 594735.
- Before selection, Codex independently viewed frame 0 rendered from saved
  extraction predictions. The sole D0 box corresponds to the visible subject
  lying on the bed. This seed-only check is separate from the supplied sample review.
  Evidence: `seed_candidates_frame_000000.png` and `seed_verification.json` in the notes directory.
- Existing causal IoU selection: seed 0/D0, min IoU=.1, margin=.05, max gap=.5 s.
  **1 seeded + 587 matched, 0 unresolved**, no gap ranges. These thresholds remain
  experimental. D0 on every selected row is an observed result, not a fixed identity rule.
- Existing review exporter decoded/aligned all 588 frames, displaying preview
  indices **0,84,168,252,336,419,503,587**. No zero/multiple/unresolved cases exist
  to add. All boxes and gate-based skeleton lines use existing drawing rules.

## Outputs and checks

- `outputs/P001_SE03_V001_poses_pt640_20260929/`: poses.jsonl, summary.json, run_info.json.
- `outputs/P001_SE03_V001_target_20260929/`: target_poses.jsonl, selection_summary.json.
- [Compact selected-target sheet](../outputs/P001_SE03_V001_target_20260929/review/reviewed_selection_sheet.jpg)
  and eight full-resolution PNGs in the same directory, including `frame_000419.png`.
  `unresolved_selection_sheet.jpg` explicitly shows no matching frames.
- Review manifest retains selected frames, timestamps, all people/raw joints/flags
  and association results. Its execution-time pending verdict is preserved;
  the subsequent assistant sheet review is recorded separately above.
- `outputs/P001_SE03_V001_association_notes_20260929/verification.json`: timeline,
  input hashes, all selected raw people/17 joints/independent flags, recomputed quality
  flags, image dimensions and eight-frame review coverage verified. Actual package
  versions and PT configuration match the preceding preview run.

Extraction and selection CLI exit 0. Existing Ultralytics `half` deprecation warnings
remain; actual dtype is float32. No algorithm or dependency edits, no threshold tuning.
Rejected keypoints remain in JSON; confidence/gate counts and successful association
do not establish joint accuracy or identity throughout the video. Preserve SE02/SE03
as the same recording-occasion group; labels and split remain pending. No training or
Pi benchmark. M2 remains IN_PROGRESS. Next: review full-size frame 419; arm/torso
joint placement remains uncertain after the sheet review.
