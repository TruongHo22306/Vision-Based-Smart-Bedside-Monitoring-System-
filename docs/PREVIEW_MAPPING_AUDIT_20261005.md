# P001_SE04_V001 — audit of reported frame 81/80 discrepancy

## Finding

The reported discrepancy is **not reproduced in the current local artifacts**.
Codex opened both original contact sheets: their third sample reads **Frame 80,
t=5.392133 s**. All 12 labels and their image mappings were checked. Independent
sequential decoding and exact-byte reproduction of both sheets support this result.

| Source frame index (0-based) | PTS, time base 1/30000 | Presentation time (s) |
|---|---:|---:|
| 80 | 161764 | 5.392133333333334 |
| 81 | 163684 | 5.456133333333334 |

No local display-label defect, stale current local sheet or incorrect frame mapping
was found. The exact uploaded/displayed copy reported as saying 81 was not supplied
for byte comparison. Its cause remains unverified: do not claim cache, stale upload
or a reading mistake as established fact. No correction or inference rerun is needed
for the verified local files. No code defect was found; no regression test was added.

## Actual checks, all 12 samples

- Read the inspection timeline, preview `samples.csv`, PNG paths/hashes,
  pose `poses.json`/`run_info.json`, and both rendering functions.
- Sequentially decoded the imported original video: all **439** frame indices,
  PTS, time bases, presentation/relative times and deltas match the inspection CSV.
- Samples: **0,40,80,120,159,199,239,279,319,358,398,438**. Every saved raw PNG
  matches the decoded source frame pixel-for-pixel, with correct filename/index/time.
- The input CSV hash matches `run_info.json`; each PNG hash matches the corresponding
  `poses.json` source hash, source path, frame index and actual timestamp.
- Re-exported raw images and raw sheet into a new audit directory using the existing
  exporter. Every regenerated PNG has the same SHA-256 as its original.
- Re-rendered annotations using the saved predictions and unchanged display threshold;
  each annotated PNG is byte-identical to its original. No YOLO/model execution.
- Rebuilt the pose sheet with the existing renderer. **Both complete JPEG sheets are
  byte-identical to the originals**, including all captions, image cells and counts.
- Raw sheet SHA-256: `140911a388e0ae7bf4a1ca1c1a2a9acabb666b4e12f66cde727a8189e9a8a6f7`.
- Pose sheet SHA-256: `7229e62dfafdaaeef596ada09dfb82bc5f79ede6d3aaa112d899027f98e8d6b4`.

## Evidence paths and separate qualitative review

- `outputs/P001_SE04_V001_mapping_audit_20261005/audit.json`: all 12 row checks,
  source hashes, adjacent frame timings and sheet comparisons.
- [Reproduced raw sheet](../outputs/P001_SE04_V001_mapping_audit_20261005/reproduced_raw/contact_sheet.jpg).
- [Reproduced pose sheet](../outputs/P001_SE04_V001_mapping_audit_20261005/reproduced_pose/pose_contact_sheet.jpg).
  These are verified reproductions, not corrected predictions or replacement artifacts.
- `outputs/P001_SE04_V001_mapping_audit_20261005/manual_observations.json`: records
  the assistant qualitative review supplied in this conversation: all 12 pose samples
  detect the visible subject, with no apparent extra background detection; frame 120
  is near the upper boundary with overlapping limbs; frame 319 has crossed legs.
  Joint accuracy is unverified. This account is separate from the label/mapping audit,
  not ground truth or final temporal annotation.

Original artifacts, code, thresholds, dependencies, weights and PROJECT_PLAN are
preserved. Mapping is verified for these local samples; full-video extraction was
not run. Labels/split remain pending; SE02/SE03 grouping is unchanged. M2 IN_PROGRESS.
Next: compare the verified raw sheet with the exact copy previously seen as frame 81
if that display discrepancy persists.
