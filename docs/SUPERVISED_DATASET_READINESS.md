# Supervised dataset v001 — readiness, not training

## Scope and provenance

Run: `outputs/supervised_action_dataset_v001_20261015/`. Saved features only;
v002/v003 frozen, no inference, fitting, annotation edits or split assignment.
`dataset/windows.jsonl` is the candidate supervised dataset; `inventory.json`,
`rejections.jsonl`, `summary.json`, `manifest.json` preserve support and hashes.
Root `annotation_inventory.json` scans all 16 segment-annotation CSVs, including
empty files and old drafts. Canonical sources are the four-clip confirmed CSV
and the SE06 interior CSV. Draft/proposed copies never become labels.

Eligibility requires reviewed, clear, known-class activity plus reviewer/time
provenance and an explicitly listed direct-confirmation source in inputs.json.
The four whole clips were explicitly confirmed by user playback. The assistant
did not independently establish whole-video labels. The earlier SE06 interior
evaluation used assistant-selected subsets of approximate user feedback. Its
`reviewed` status means usable for that scoped evaluation, not independently
confirmed exact boundaries. **This stricter supervised dataset excludes all
SE06 segments**, retaining them in the inventory; no annotations are promoted.
The input source list is a manually reviewed provenance allowlist, not a label
inferred from subject IDs, video names, recording intent or baseline predictions.

## Actual label inventory

All rows below are P001, occasion **P001_SE05_SE06_SE07**. Seconds are from verified
source PTS and final frame duration. Each direct-confirmed row is one segment.

| Video | Label/tier | Half-open seconds | Duration | Windows |
|---|---|---|---:|---:|
| SE05_V001 | standing, direct | [0,18.6418333333) | 18.641833 | 71 |
| SE05_V002 | standing, direct | [0,15.5537333333) | 15.553733 | 59 |
| SE05_V005 | sitting, direct | [0,21.6819) | 21.681900 | 83 |
| SE07_V001 | lying, direct | [0,20.9458333333) | 20.945833 | 80 |
| SE06_V001 | moving, preliminary | [2,2.9760666667) | 0.976067 | 0 |
| SE06_V001 | standing, preliminary | [5.0401,6.9921666667) | 1.952067 | 0 |
| SE06_V001 | moving, preliminary | [10.0002666667,10.9763) | 0.976033 | 0 |

Direct confirmed: **76.8233 s / 1448 frames**, four videos, three classes,
one person and one occasion. Moving preliminary support is only **1.9521 s in
two separate segments**, each shorter than 1 s; even if their provenance were
later confirmed, neither fits the fixed window. Never concatenate these spans.
SE06 other 9.777533 s remain pending. No labels are inferred for that remainder.

SE05_V003/V004 have features but no confirmed intervals. SE01–SE04 also have
features (284/204/588/439 frames) but no confirmed activity segments; SE04 drafts
and qualitative pose reviews are excluded. Pilot01 (415 frames) remains
development-only, no confirmed activity segments, recording occasion unknown.
Root `unlabeled_inventory.json` records their paths/hashes and feature availability.
SE01 and SE04 are separate known occasions; **SE02/SE03 stay together**, as do
**SE05/SE06/SE07**. Per-video recording_group_id is not an occasion identifier.

## Window construction and rejection policy

Experimental fixed settings: duration **1.0 s**, stride **0.25 s**; no optimization.
For each eligible segment [a,b), endpoints are a+1+k*0.25 while endpoint<=b.
Window [endpoint-1,endpoint) is causal and never concatenates segments/videos.
Grid anchoring uses annotation start only to select supervised examples, never as
a model feature. No extra off-grid endpoint is added at a segment's final tail.
The first window includes startup frames; insufficient short histories are not
filled. Actual PTS/time_base use exact rational arithmetic, not nominal FPS.

Each whole window must have source display support and selected seeded/matched
targets. Internal reseeds, invalid temporal_pair/reference, unresolved targets,
invalid timestamps or source interval >0.5 s reject it. The 0.5 s guard reuses
the saved feature continuity convention; it is not a classifier change.
An unmarked identity switch cannot be discovered from these files alone.

Static features use the known frame display interval intersected with the window,
holding the observed value without interpolation. Source final frame duration
is preserved. Features at/after the window end are never read for summaries.
Backward hip velocities use their full [previous PTS,current PTS] support only
when entirely inside the window. A velocity depending on a pre-window frame is
excluded, even if its current frame is inside. This conservatively loses some
edge support. Missing joints do not break target identity by themselves.

293 scheduled eligible windows accepted; **0 scheduled windows rejected** for
track/coverage problems. Seven SE06 annotation regions are separately rejected
at eligibility (not seven attempted windows); two are also shorter than 1 s.
Unannotated videos generate no candidate grid. Boundary tails shorter than one
stride remain unused: standing covered union34.0s, sitting21.5s, lying20.75s,
total **76.25 s** versus76.8233s labeled. Windows overlap, so their summed293s
is not labeled duration or independent sample size.

## Feature subset and missingness

Seven scalar channels, no requirement for all17 joints:

| Channel | Units / meaning |
|---|---|
| left/right knee angle | degrees, existing valid hip-knee-ankle geometry |
| torso inclination | degrees relative to upward image vertical |
| hip x/y backward velocity | image-width/image-height fractions per second |
| box width/height | fractions of image width/height |

Each channel exports duration-weighted mean and population standard deviation,
valid flag, valid observation count, valid duration/coverage and missing reasons.
Mean=sum(value*support)/sum(valid support); std uses the same denominator.
No valid support => mean/std null, coverage0, valid=false. Partial coverage keeps
the observed summary and an explicit coverage indicator; there is no minimum
feature-coverage filter or missing-value zero fill. Zero observed motion remains
a legitimate zero. No imputer, scaler, normalization fit or smoothing is run.
Future preprocessing must be fitted on a separately established training group.

All five static channels are valid on every direct-confirmed frame; hip velocity
is valid on1444/1448 frames (each video's first frame lacks a predecessor).
Window static coverage is100% within floating-point rounding; temporal coverage:
SE05_V00186.4–95.987%, V00286.4–94.42%, V00586.4–96%, SE07_V00194.397–99.203%.
`verification.json` gives exact per-video and per-segment counts and denominators.

IDs, source paths, PTS and frame lists are provenance outside `model_inputs`.
Do not feed these, label, segment boundaries, absolute recording time, context,
or missing-reason text into a model. Numeric summaries/validity/coverage are the
documented inputs. Box size and image velocities depend on perspective, camera
angle, scale and pose jitter; none independently proves locomotion or accuracy.

## Split feasibility and next collection

**No feasible four-class group-disjoint split now.** All directly labeled windows
belong to one occasion; moving has0 windows, and there is only one participant.
More overlapping windows cannot resolve either problem. SE01–SE04 and the later
batch have already informed baseline development; do not advertise them as an
independent test. Keep all same-occasion derivatives together. Pilot stays
development-only. No split is assigned by this dataset exporter.

Next: obtain reviewed walking spans comfortably longer than1s (e.g. several
continuous5–10s stretches) without expanding the existing short labels. Record
approach/retreat and sideways walking; mark pauses, turns with uncertainty and
occlusions separately. Collect all four activities across independent occasions,
including standing arm adjustments, with fixed camera/full-body visibility and
actual subject/session/occasion metadata. Follow the plan toward5–8 participants
and>=2sessions per person; select held-out subjects/groups before tuning. Neither
these collection suggestions nor intended activities are automatic labels.

## Reproduction and verification

From the project root, choose a new output directory:

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring.supervised_dataset outputs/supervised_action_dataset_v001_20261015/inputs.json --output-dir outputs/supervised_action_dataset_rerun
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_supervised_dataset.py -v
```

Existing output directories are refused. Failures leave incomplete summary.
Actual checks:12 focused tests +21 feature +8 evaluator +33 baseline tests PASS
(74 total). One initial assertion used exact float equality and was corrected to
tolerant comparison; it was not a feature/data defect. 2482 batch feature rows
match source timeline indices/PTS/time_base; seven local source hashes verified;
all293 windows rechecked against confirmed bounds and feature coverage. Protected
files are audited against a pre-task990-file hash snapshot; STATUS is the sole
intentional existing-file change. Baseline rules/configurations remain frozen.
