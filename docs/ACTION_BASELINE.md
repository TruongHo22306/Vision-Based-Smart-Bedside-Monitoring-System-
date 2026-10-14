# Experimental offline action baseline v001

## Scope and inputs

`action-baseline` reads saved `per_frame_v001` features only. No pose inference, video decoding,
training, annotation confirmation or split assignment. This is an initial rule baseline, not a
validated activity detector. M2 remains IN_PROGRESS; this does not satisfy M4 evaluation gates.

Actual schema inspected for SE01–SE04: 284/204/588/439 rows. Torso inclination, both knee angles
and pixel box dimensions are valid under saved masks on every row. Hip/box velocity is invalid
on the first row only. Full 17-joint normalization is available on 284/0/51/235 rows respectively,
so requiring all joints would unnecessarily reject SE02. Validity is not accuracy.

Six feature fields are used, without changing their saved quality masks:

| Saved feature | Window statistic | Units |
|---|---|---|
| torso_inclination_deg | Time-weighted mean | Degrees from image upward vertical, 0–180 |
| left_knee_angle_deg, right_knee_angle_deg | Separate time-weighted means | Degrees; straight = 180 |
| box_size_px | Time-weighted mean of pixel width / pixel height | Dimensionless aspect |
| hip_image_velocity_per_second | Norm of time-weighted mean 2D vector | Image width/height fractions per second |
| box_size_image_velocity_per_second | Norm of time-weighted mean 2D vector | Image width/height fractions per second |

The aspect is computed from pixel dimensions, not width/height fractions (which would include
the image aspect ratio). No normalized joint coordinates, face joints, filename, video ID,
recording context, annotations or draft intervals are used by the rule decision. IDs/source
metadata are copied for provenance and used only to prevent windows crossing source boundaries.

## CLI and configuration

PowerShell, in the project root:

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring action-baseline --help
.\.venv\Scripts\python.exe -m patient_monitoring action-baseline outputs/P001_SE01_V001_features_v001_20261007/features.jsonl --output-dir outputs/P001_SE01_V001_action_baseline_v001_20261008
```

The example output already exists after the recorded run; reruns require a NEW directory.
Apply the same command to SE02, SE03 and SE04 with their verified feature files. Optional
`--config PATH.json` accepts a JSON object overriding these defaults; unknown keys are errors.
The complete effective configuration and its hash are always saved. Do not tune by clip.

| Configuration key | Default |
|---|---:|
| window_seconds | 1.0 |
| minimum_coverage | 0.8 |
| minimum_intervals | 3 |
| max_gap_seconds | 0.5 |
| upright_max_deg | 30 |
| lying_min_deg / lying_max_deg | 60 / 120 |
| lying_min_box_aspect | 1.2 |
| sitting_max_knee_deg / standing_min_knee_deg | 135 / 155 |
| quiet_hip_speed / moving_hip_speed | 0.015 / 0.03 |
| quiet_box_speed / moving_box_speed | 0.015 / 0.03 |

All defaults are **unvalidated heuristics**, chosen once before inspecting predicted label counts.
They are not fitted to the four recordings and are not a deployment configuration.

## Causal window and missing values

At source time t, use only observations ending in **(t − window_seconds, t]**. Time is computed
as exact rational PTS × time_base, checked against the saved presentation timestamp (1e-9 s
numeric tolerance). Never use nominal FPS, processing speed or future frames.

For each adjacent pair a,b, its contribution duration is the intersection of (ta,tb] with the
window. Static features contribute the right-endpoint observed value, only when both endpoints
are valid. Velocity contributes the saved backward-difference vector at b, whose source gate
already validates both endpoints. This is an interval-weighted statistic; there is no resampling,
interpolation or filling of missing coordinates. A masked interval contributes neither value
nor duration; zero accumulator initialization never substitutes a missing observation.

Divide the weighted sum by **valid covered duration**, not the nominal window duration.
Coverage itself is valid covered duration / **full configured window duration**, including
startup. Each of the six channels must have coverage >=0.8, >=3 contributing intervals and a
valid current observation. Thus startup normally yields unknown until at least 0.8 seconds.
No evidence is recovered from a rejected mask. Invalid aggregates remain null with a reason;
their duration/count diagnostics remain visible. Raw selected input values/masks are copied.

The motion statistic is ||sum(v × duration) / covered_duration||. It measures net image change,
not average absolute speed, physical speed, gait or proof that feet stepped. Opposite movement
within the window may cancel. Hip translation and box growth/shrinkage provide complementary
cues for sideways motion and approaching/receding, respectively.

Clear history at a video/source/stream/image-size change, nonadjacent index, unresolved target,
reseed/track break, changed explicit track identifier, invalid saved temporal pair/reference,
invalid/nonincreasing timestamp or adjacent dt > max_gap_seconds. dt equal to the limit is allowed.
The saved feature temporal barriers remain authoritative even if this configuration allows a
larger gap. Invalid targets/times do not seed a new window; otherwise the current row starts
a fresh segment after a barrier. Missing individual features reduce coverage without treating
them as zero. They never create derivatives bridging a missing observation. Only a trailing
history plus one boundary predecessor is retained; window size is temporal, not a frame count.

## Rules and conflicts

All comparisons below use the valid window statistics. Insufficient evidence returns unknown
with `insufficient:<feature>`. The rules keep posture and motion evidence separately:

1. Posture evidence: lying if torso is within [60,120] degrees and box aspect >=1.2;
   sitting if torso <=30 and **both** knees <=135; standing if torso <=30 and both knees >=155.
   Otherwise posture evidence is null; intermediate or disagreeing knees are not forced.
2. Motion evidence: high if hip >=0.03 **or** box >=0.03; quiet if hip <=0.015 **and** box <=0.015;
   otherwise indeterminate. Both channels need sufficient coverage even though high uses OR.
3. High motion with sitting/lying evidence returns unknown (`motion_posture_conflict`).
   Other high motion with upright torso yields moving, including intermediate knee geometry
   during steps. High motion without upright support returns unknown.
4. Quiet motion uses the posture result, or unknown if posture is ambiguous. Indeterminate
   motion always returns unknown (`motion_threshold_band`), even if posture looks clear.

Moving is a temporal activity hypothesis; standing/sitting/lying describe relatively quiet
postures under these rules. A high-motion upright person therefore outputs moving rather than
standing. No smoothing, label holding, future lookahead, majority vote or automatic annotation
promotion is applied. At turns/pauses the result may be unknown, lag or alternate; turns are
not an additional activity class. `unknown` is an operational state, not a new annotation class.

## Outputs and diagnostics

Every input row, including missing/unresolved cases, yields one `predictions.jsonl` row with
original video/frame/PTS/time_base/timestamps and metadata, prediction/reasons, posture/motion
evidence, selected input values/masks, per-channel supporting statistic and coverage, reset
reason, config hash and input SHA-256. Complete configuration, Python/code/output hashes and
no-inference/no-label provenance are in `provenance.json`. Existing directories are refused.
Failure writes incomplete status/errors and retains any partial predictions for inspection.

`summary.json` contains label counts (denominator: **all written rows**), unknown reason counts
(multiple reasons can occur on a row), and every adjacent-row label change with original PTS.
Transitions include entry/exit from unknown and transitions after barriers; they are prediction
diagnostics, not confirmed activity boundaries. Initial label is not counted as a transition.

## Actual SE01–SE04 diagnostic run (2026-10-08)

Outputs: `outputs/P001_SE0{1,2,3,4}_V001_action_baseline_v001_20261008/`, each containing
`predictions.jsonl`, `summary.json`, `provenance.json`. All four runs completed with the same
default config hash; no thresholds were changed after viewing these results.

| Video | Rows | standing | sitting | lying | moving | unknown | Label transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| SE01 | 284 | 195 | 0 | 0 | 38 | 51 | 19 |
| SE02 | 204 | 0 | 192 | 0 | 0 | 12 | 1 |
| SE03 | 588 | 0 | 0 | 334 | 0 | 254 | 253 |
| SE04 | 439 | 13 | 0 | 0 | 393 | 33 | 13 |

Unknown reasons (grouped by rows; startup rows each list all six insufficient channels):

| Video | Insufficient startup coverage | Motion threshold band | Motion/posture conflict |
|---|---:|---:|---:|
| SE01 | 12 | 39 | 0 |
| SE02 | 12 | 0 | 0 |
| SE03 | 24 | 191 | 39 |
| SE04 | 12 | 21 | 0 |

Transitions are listed with exact frame/PTS/time in each summary, including transitions to/from
unknown. SE02 first changes unknown→sitting at frame12. SE01's38 moving predictions and SE03's
253 transitions deserve timed video review. Do not silently call these accuracy errors based
on source filenames or qualitative sample review. SE04's393 moving predictions do not confirm
any draft interval or resolve its uncertain pauses/turns.

Verification: 15 new tests, full121-test suite PASS; `pip check` PASS; CLI help and all four CLI
runs succeeded. All1,515 rows preserve source metadata, timestamps and selected feature masks.
9,090 per-channel window statistics independently recomputed from the saved source features.
467 pre-existing data/model/output/plan/dependency-declaration files have unchanged SHA-256.
Report: `outputs/action_baseline_v001_audit_20261008/verification.json`; initial protection map
in the same directory. No model inference or training was performed.

Commands actually used:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_action_baseline.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
foreach ($session in @('01','02','03','04')) { $videoId = "P001_SE${session}_V001"; .\.venv\Scripts\python.exe -m patient_monitoring action-baseline "outputs/${videoId}_features_v001_20261007/features.jsonl" --output-dir "outputs/${videoId}_action_baseline_v001_20261008"; if ($LASTEXITCODE -ne 0) { throw "Baseline failed: $videoId" } }
```

## Limitations and next review

Image geometry depends on camera tilt, viewpoint, scale, perspective and occlusion. Bent knees
do not prove sitting; image-horizontal torso/box does not prove lying. Camera movement, box
jitter, posture shifts or target switches can imitate locomotion; head-on gait, occluded legs,
turns with little displacement and opposing motion can be missed. Temporal aggregation delays
changes. Unknown reasons and valid masks are not calibrated uncertainty or joint accuracy.

Only one person and four development recordings have been processed. There are no confirmed
temporal evaluation labels, accuracy/F1 results, held-out test set or Pi measurements here.
Review errors against original timed video and complete annotation independently before
evaluating or choosing thresholds. Keep SE02/SE03 grouped; no split is assigned by this task.
