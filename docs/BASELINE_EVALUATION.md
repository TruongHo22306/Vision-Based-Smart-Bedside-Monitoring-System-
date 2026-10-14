# Frozen baseline v002: confirmed-only evaluation

Entry point: `python -m patient_monitoring.baseline_evaluation --package PACKAGE.json --annotations ANNOTATIONS.csv --output-dir NEW_DIR`.
Existing dependencies only; no inference. New output directory required; failures produce an incomplete evaluation.json with errors.

## Annotation first

Package: `outputs/evaluation_v002_20261015_v001/annotation_review/index.html`.
Open locally: original source videos, source timestamps/frame lookup and source-only sheets; no predicted labels or skeletons.
`annotations_proposed.csv` and empty `annotations_confirmed.csv` reuse `docs/segment_annotations_template.csv`.
Do not change prior drafts. Save a new annotation version after human review.
The assistant inspected 63 stills (9 per clip), **not continuous playback**. Proposed whole-clip postures are questions for confirmation, not evidence of uninterrupted activity.
Transitions, brief pauses, unseen intervals and occlusion boundaries remain pending; no guessed exact boundary is promoted.
The user's favorable review of prediction videos is stored separately in user_feedback.json. It provides no confirmed intervals.
Prior prediction exposure cannot be undone; source-only annotation reduces further bias, but is not blinded independent review.

## Existing schema and state mapping

- Confirmed known class: `annotation_status=reviewed`, `label_certainty=clear`, `segment_kind=activity`, one of standing/sitting/lying/moving, with annotator/reviewer IDs and timestamps.
- Proposed: `draft` activity, never scored. Candidate whole-clip ranges can have uncertain certainty until confirmed.
- Uncertain: `needs_review`, uncertain/transition intervals, unknown activity. Never force a new uncertainty class.
- Confirmed excluded: reviewed unknown/no_person/uncertain/transition or uncertain-certainty rows; explicit reason retained. Excluded from the four-class task, not erased.
- Uncovered frames are unannotated, not implicit negatives. No automatic promotion; empty confirmed template is intentional.

Intervals are `[start_frame_index,end_frame_index_exclusive)`; seconds relative to first valid source PTS.
Internal ends equal next frame PTS; video end is last PTS + decoded source duration. Rounded display seconds are not accepted as exact boundaries.
Evaluator validates count, video identity, PTS/time_base/time, order, bounds, overlap and boundary seconds (1 microsecond tolerance for CSV decimals).
Package pins timeline and prediction SHA-256 plus frozen config hash. Timeline durations were independently decoded from source and compared to saved export mapping.
Strict validation rejects missing predictions rather than silently reducing the denominator.

## Metrics and denominators

Rows of confusion matrix: confirmed standing/sitting/lying/moving. Columns: the same four plus predicted unknown.
Each confirmed frame contributes once; startup and unresolved-target unknown predictions are included as incorrect.
Two separate results: frame counts and seconds weighted by each decoded source frame duration, not nominal FPS.
Duration totals exclude genuine timestamp gaps and report those gaps separately; no resampling or imputed durations.
For each class, precision = TP / all predictions of that class on confirmed known-class frames;
recall = TP / all confirmed truth frames of that class (including predicted unknown);
F1 = 2TP / (truth support + prediction support). Each result exposes numerator and denominator.
Zero denominators give null and `zero_denominator`, never zero or NaN by convention.
Prediction coverage = non-unknown predictions / all confirmed known-class frames.
Overall accuracy = correct predictions / all confirmed known-class frames, including unknown as incorrect.
The duration-weighted versions use seconds in both numerator and denominator. No numerical accuracy/matrix is emitted without confirmed known-class labels.
Annotation coverage and excluded-from-metric duration/reasons are reported separately for proposed, uncertain, confirmed-excluded and unannotated regions.
Metrics describe only the confirmed subset; partial labeling cannot establish whole-video accuracy. Coverage is not accuracy.

## Commands (PowerShell from project root)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_baseline_evaluation.py
.\.venv\Scripts\python.exe -m patient_monitoring.baseline_evaluation --package outputs/evaluation_v002_20261015_v001/evaluation_package.json --annotations outputs/evaluation_v002_20261015_v001/annotation_review/annotations_confirmed.csv --output-dir outputs/evaluation_v002_confirmed_run_NEW
```

Keep P001_SE05/SE06/SE07 in recording occasion P001_SE05_SE06_SE07; no split assigned.
Frozen v002 config SHA-256: `ec8a41657c61207951957584fac50a9b6c68f29a7c25ae0716abaedc2068bbbc`.
These are one person's home recordings, not evidence for unseen subjects or hospital conditions. No tuning/training; M2 remains IN_PROGRESS.
