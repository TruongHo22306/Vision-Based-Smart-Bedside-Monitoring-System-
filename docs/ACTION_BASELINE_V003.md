# Experimental v003 locomotion revision — development analysis

V002 is frozen and unchanged. This is a separate experimental command:

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring.action_baseline_v3 PATH_TO_FEATURES.jsonl --output-dir NEW_DIRECTORY
```

Inputs are existing per_frame_v001 features, including individually masked joints derived from saved target poses. No pose inference, new model or dependency. New landmark-window measurements are stored in each prediction's `landmark_motion`; old features are never edited. The pilot lacked features, so they were derived from its saved targets into a new directory.

## Measured diagnosis versus interpretation

SE05_V002 is user-confirmed standing with arm/body adjustments. All178 false moving frames take v002 `upright_with_temporal_motion` because **box speed alone exceeds0.03 image-fraction/s**. Hip speed remains below0.03 throughout those frames. SE06's20 unknown moving-interior frames all take `motion_threshold_band`, with valid support, hip below0.015 and box between0.015 and0.03. This is a rule/input mismatch, not missing pose evidence or an identified timestamp/coverage implementation bug.

`outputs/action_v003_development_20261015_v001/diagnostic_rows.jsonl` preserves every affected row's original PTS, branch, hip/box values, distances to thresholds and new measurements. `diagnosis_summary.json` has measured ranges. The existing saved v002 decisions/support were replayed exactly for all11 clips with saved v002 outputs before comparison.

Hypothesis: arm/body adjustments alter detector box extent without locomotion; far-away walking produces small image-normalized motion. These are plausible explanations consistent with source review and the measured branch behavior, not causal proof or evidence that keypoints are accurate. Landmarks can also jitter, and torso scale can change with posture.

## One shared revision: body transport AND lower-body change

For upright cases only, replace the old box/hip OR rule with the following evidence conjunction. Preserve v002 sitting/lying decisions, its geometry thresholds and its existing six-channel support/coverage prerequisites. Box values no longer directly trigger moving or motion-band unknown in the changed branch; box validity remains an inherited prerequisite. No source ID, filename, recording intention or annotation enters a decision.

1. Use a trailing1s window of actual rational PTS, keeping only observed samples at/after current time minus1s. Require a continuous valid landmark span at least0.8s and at least3 consecutive intervals. Do not interpolate the window boundary.
2. Require saved-valid hip midpoint H, shoulder midpoint S, positive torso scale L=distance(H,S), and both ankles' saved-valid body-normalized coordinates. Every sample in the landmark window must be valid; any missing landmark clears that window. Do not rethreshold confidence.
3. For oldest/current endpoints a,b, dt=PTS_b-PTS_a and Lmean=(L_a+L_b)/2:
   - hip speed = distance(H_b,H_a)/(Lmean*dt), torso lengths/s;
   - shoulder speed = distance(S_b,S_a)/(Lmean*dt), torso lengths/s;
   - scale speed = abs(log(L_b/L_a))/dt, inverse seconds.
   Body transport candidate requires **both** hip and shoulder speed>=0.10, **or** scale speed>=0.05. Translation addresses lateral movement; scale change is a tentative approach/retreat proxy, not depth reconstruction.
4. At each sample compute ankle separation D=(left ankle-right ankle)/torso scale. Hip centering cancels. Lower-body range R=sqrt((max D_x-min D_x)^2+(max D_y-min D_y)^2) across the same past window. Require R>=0.15 alongside body transport for `moving`.
5. With v002 standing geometry, no transport and R<=0.10, return `standing`. Otherwise return `unknown` with an explicit reason. Missing evidence never becomes zero/standing. Hip movement alone, box change alone or leg movement alone cannot trigger walking.

These round thresholds (10% torso/s, 5% log-scale/s, 15% torso range and10% quiet range) are **unvalidated heuristics selected after development diagnosis**, not fitted or optimal thresholds. One configuration was run once on all12 processed clips; no grid search, per-video settings or adjustment after observing v003 evaluation results.

The range is lower-body-change evidence, **not gait-cycle detection**. It does not prove actual steps or distinguish turning/leg adjustment from walking. The remaining standing false positive at SE05_V002 frame217 demonstrates this limitation: torso scale change and ankle range cross both gates despite user-confirmed no walking. Ankle range depends on view, occlusion and jitter; torso scale depends on pose/perspective. Camera motion is not estimated. Pure depth movement and partial-body cases remain uncertain.

## Causality, gaps and delay

Reuse v002 resets for source/video changes, invalid/nonincreasing PTS, nonconsecutive frames, reseed, track breaks/IDs, unresolved targets, saved temporal barriers and gaps>0.5s. Also clear the extra window on any required landmark missing/invalid. No future-frame lookahead, smoothing, debounce, label hold or automatic reseeding.

Added lookahead and explicit debounce delay:0s. A new uninterrupted valid span>=0.8s is required after a missing landmark, potentially increasing unknown time. Earliest sample-based readiness is the first frame meeting that span and inherited coverage gates; it is not a guaranteed onset latency. A trailing1s range can retain earlier foot changes for up to1s, and endpoint displacement may cancel on turns. Exact activity onsets are not annotated, so empirical onset/offset delay cannot be measured here; first detections inside the evaluation subsets must not be called onset delay.

## Results and validation

See [development report](../outputs/action_v003_development_20261015_v001/DEVELOPMENT_REPORT.md), per_video_comparison.json and the two separate evaluation directories. All4,412 source rows preserved across12 clips; one configuration hash. Exact same immutable confirmed four-clip annotations and SE06 interior subsets were used. Unknown counts as incorrect; no startup exclusion. No accuracy is inferred for unannotated clips or pilot.

SE05_V002 false moving178→1 (12.049467→0.064s), but unknown25→106. SE06 moving recall38/58→48/58; standing49/58→40/58. Nine prior correct SE06 standing frames become unknown. All v002 sitting/lying labels are unchanged across12 clips. This is a tradeoff, not an unqualified improvement or a recommended replacement.

10 new tests cover box-only/hip-body-only/leg-only rejection, lateral/depth candidates, saved masks, missing/gap/reseed/track resets, irregular PTS, causal prefixes, identifier invariance, scale normalization, non-upright preservation and invalid configuration. Together with existing baseline/features/evaluator tests:62 PASS; pip check PASS.

SE05_V002 and SE06 are explicitly development data for this revision, alongside previously used SE01–04/pilot. No independent test/generalization claim, training, split assignment, deployment or M2 completion. Same recording-occasion grouping is retained. Frozen v002 remains available.
