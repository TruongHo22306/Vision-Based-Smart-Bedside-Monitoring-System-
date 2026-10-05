# Sitting clip: extraction and experimental target selection

## Scope and manual sample observations

Input: `data/raw/P001/P001_SE02/P001_SE02_V001.mp4`, imported from
`WIN_20260928_11_33_49_Pro.mp4`. Original import/provenance remains in
[SITTING_RECORDING_IMPORT_20260928.md](SITTING_RECORDING_IMPORT_20260928.md).

The conversation's qualitative review reports the seated subject in all eight
preview samples. Frame 29 at 1.952067 s has an extra detection on hanging
background objects at the upper left, visually judged a false positive.
This is a supplied sample review, not automated ground truth or full-video
validation. It is recorded separately in
`outputs/P001_SE02_V001_association_notes_20260928/manual_observations.json`;
predictions remain unchanged and these observations do not drive association.

## Executed extraction and selection

- Existing PT640 pipeline completed on Windows: CPU/float32, batch=1,
  rect=False, conf=0.25, IoU=0.7, established local YOLO11n-pose weights.
- 204 decoded/processed/written frames, 214 detections; 0 zero-detection
  frames and 10 multiple-detection frames: 2, 5, 28, 29, 51, 52, 57, 61, 67, 121.
- Every extraction and selection index/PTS matches the existing inspect-video
  timeline. No missing, duplicate or decreasing timestamps; no corrupt frames.
  PTS span: 0 to 13.696366666666666 s, time base 1/30000.
- Before association, the assistant viewed frame 0 rendered from saved
  predictions: its sole D0 box encloses the visible seated subject. Seed evidence
  and its scope are saved in `seed_verification.json` in the notes directory.
- Existing causal IoU selection used seed frame 0/D0, min IoU=0.1,
  ambiguity margin=0.05, max elapsed gap=0.5 s. These remain experimental.
  Result: 1 seeded + 203 matched, 0 unresolved, no gap ranges.
  D0 on all 204 rows is an observed result, not a hardcoded identity rule;
  detection indices are frame-local. Raw selected joints and flags are unchanged.

## Outputs and verification

- Extraction: `outputs/P001_SE02_V001_poses_pt640_20260928/`
  (`poses.jsonl`, `summary.json`, `run_info.json`).
- Selection: `outputs/P001_SE02_V001_target_20260928/`
  (`target_poses.jsonl`, `selection_summary.json`).
- All ten multiple-detection frames, full-resolution PNGs and sheet:
  `outputs/P001_SE02_V001_pose_review_20260928/multi_detection_sheet.jpg`.
- Yellow selected boxes on all ten multi frames plus frame 0 and 203:
  `outputs/P001_SE02_V001_target_20260928/review/reviewed_selection_sheet.jpg`.
  The unresolved sheet explicitly has no matching frames. Boundary review found
  no knee/ankle boundary candidates under the existing criterion.
- Actual alignment, raw-value/flag preservation, image decoding and review
  coverage checks: `outputs/P001_SE02_V001_association_notes_20260928/verification.json`.
  No algorithm/dependency changes or new regression-test claims.

## Conversation review of uploaded sheets

The following is attributed to the assistant's review of uploaded sheets in the
conversation, supplied by the user for this update. Codex records the account;
it is not Codex's independent inspection. The earlier seed verification above
is a separate, previously recorded check.

- In all 12 displayed frames (0, 2, 5, 28, 29, 51, 52, 57, 61, 67, 121, 203),
  the yellow box selects the seated subject. No target switch is visible in
  these reviewed frames.
- Extra D1 detections at frames 2, 5, 28, 29, 51, 52, 57, 61, 67, 121 are
  visually judged background false positives on hanging objects at the upper
  left. D1 is a frame-local index, not a persistent identity or selection rule.
- Pose is qualitatively plausible at sheet resolution; overlapping limbs limit
  joint-level assessment. Gate counts are not accuracy.

This sample review is recorded in `manual_observations.json` under
`selection_sheet_review`, alongside the earlier eight-sample observations.
It is not ground truth or full-video validation; unreviewed frames and detailed
joint correctness remain unverified. Original export manifests remain unchanged
as execution-time artifacts; the subsequent conversation review is recorded here.
Association and confidence do not establish pose accuracy or full-video identity correctness.
No final temporal labels, split assignment, training or Pi benchmark was performed.
M2 remains IN_PROGRESS. Next: inspect full-resolution frames with overlapping limbs
before making joint-level judgments.
