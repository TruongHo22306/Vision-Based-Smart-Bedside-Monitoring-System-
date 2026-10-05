# P001_SE04_V001 — full extraction and experimental target selection

## Frame 80/81 finding and preflight

The earlier discrepancy's exact cause remains unverified. It was not reproduced
in local files: both sheets label frame **80 at 5.392133 s**, while frame 81 is
5.456133 s. The [mapping audit](PREVIEW_MAPPING_AUDIT_20261005.md) independently
decoded the video and reproduced both sheets byte-identically; it found no local
mapping/rendering defect or stale local sheet. The separately reported display
was not supplied for byte comparison; do not assert a cache/upload/reading cause.

Before this extraction, all **12** sample indices, actual PTS-derived times,
filenames and PNG SHA-256 values were rechecked against `samples.csv`, `poses.json`
and the inspection timeline. The CSV hash matches `run_info.json`; hashes of
original and reproduced sheets still agree. Evidence is saved in
`outputs/P001_SE04_V001_association_notes_20261005/mapping_preflight.json`.

## Supplied assistant review, separate from predictions

The conversation reports the assistant reviewed the new reproduced sheets:
all 12 displayed indices/timestamps agree, including frame 80 at 5.392133 s;
each pose sample contains one detection covering the subject. Joint accuracy
is unverified. This account is saved as `manual_observations.json` in the notes
directory. It concerns the earlier twelve-sample sheets, not the new target-review
sheet or all video frames. It is not ground truth and does not affect association.

### Subsequent assistant review: selection sheet and full-size frame 66

The user supplied the assistant's inspection of the uploaded 13-frame selection
sheet and full-size frame 66. Yellow boxes select the visible walking subject in
all 13 displayed samples. At frame 66 (**4.448100 s**), D1 covers a dark right-edge
background region and is visually judged a false positive; selected D0 is the
subject. Overlapping limbs and boundary proximity still limit joint review.

This is recorded in `manual_observations.json` under `selection_sheet_review`,
alongside the earlier twelve-sample review. Codex records the supplied account;
it is not an independent reinspection in this update. Observations are not
full-video ground truth or measured joint accuracy. D0/D1 are frame-local indices.

## Actual execution and checks

- Existing `extract-poses` on `data/raw/P001/P001_SE04/P001_SE04_V001.mp4`:
  **439 decoded/processed/written frames**, **440 detections**, **0 zero-detection**
  frames, **1 multiple-detection frame: 66 (4.4481 s)**. No corrupt frames or errors.
- All 439 extracted and selected index/PTS/time-base/presentation/relative/delta
  fields match the inspection timeline. No missing/duplicate/decreasing timestamps.
  Time base 1/30000, PTS 0 to 886103; span 29.536766666666665 s.
- Existing local YOLO11n-pose PT640, CPU float32, batch1, rect=False,
  conf=.25/IoU=.7/max_det300; quality confidence=.5. Package versions and actual
  configuration match the existing preview. No model/dependency changes.
- Codex viewed the saved-prediction overlay at frame 0 before selection: D0 encloses
  the visible subject in a white shirt/dark shorts. `seed_verification.json` records
  the evidence and scope; no assumption that D0 is always the subject.
- Existing `select-target`: seed0/D0, min IoU=.1, margin=.05, max gap=.5 s unchanged.
  **1 seeded + 438 matched**, **0 unresolved**, **no unresolved/gap ranges**.
  At frame66, selected D0 IoU=.9328693763, runner-up IoU=0. D0 on all rows is an
  observed result, not a persistent tracking ID or a selection override.
- All selected raw people and 17 keypoints exactly match source predictions;
  independent finite/confidence/boundary/quality flags were checked. Rejected joints
  are retained unchanged. Association does not establish joint accuracy.
- Existing target-review utility sequentially decoded/aligned all 439 frames and
  exported **13 full-resolution PNGs**: 0,40,66,80,120,159,199,239,279,319,358,398,438.
  All preview and exceptional frames are present; no zero/unresolved cases to add.
  Image decoding/dimensions and all review input hashes passed.

## Outputs and next review

- `outputs/P001_SE04_V001_poses_pt640_20261005/`: poses.jsonl, summary.json, run_info.json.
- `outputs/P001_SE04_V001_target_20261005/`: target_poses.jsonl, selection_summary.json.
- [13-frame target-review sheet](../outputs/P001_SE04_V001_target_20261005/review/reviewed_selection_sheet.jpg).
- [Frame66, two detections](../outputs/P001_SE04_V001_target_20261005/review/frame_000066.png).
  Review manifest and full-size PNGs accompany the sheet; unresolved sheet explicitly
  reports no matching frames. Execution-time pending verdicts in original manifests
  are preserved; the subsequent assistant review is recorded separately above.
- `outputs/P001_SE04_V001_association_notes_20261005/`: mapping_preflight.json,
  manual_observations.json, seed evidence and verification.json.

Extraction and selection CLI exit 0; existing Ultralytics `half` deprecation warning
remains, actual dtype float32. No algorithm/threshold changes or new test claims.
No final temporal labels, split assignment or training. Keep SE02/SE03 grouped;
P001_SE04 remains the user-confirmed new occasion. M2 IN_PROGRESS.
Final labels and dataset split remain pending after this sample review. The exact
cause of the earlier reported 80/81 display discrepancy is still not confirmed;
only the correct local mapping and identical reproduced sheets are established.
Next: review the original video for temporal annotation boundaries before assigning labels.
