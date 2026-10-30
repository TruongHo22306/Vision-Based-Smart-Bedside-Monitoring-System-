# Low-rate 3-second development experiment ? 2026-10-30

Status: exactly ONE training run complete. Development/resubstitution only; no held-out split, live integration, Pi package, pose inference or dependency change. Frozen original classifier, baselines, strict mode, data/annotations/PROJECT_PLAN untouched.

## Configuration locked before sampling/training
Config SHA-256: `1cd7e002e3fe32a249ada07ef838fc056e6c99b71fb7a3fafe3a5edc8abfb1d9`; preparation outputs/low_rate_experiment_preparation_v001_20261030/. New hypothesis, not validated parameters: causal [end-3,end), stride.25s, gap<=1.5s, latest observed sample age<=1.25s, >=3 observations INSIDE window and observed span>=1.5s. Features recomputed with new separate1.5s gap; no modification to strict. A contributing frame may begin before window start for duration weighting, but training rejects observations outside eligible segment. Derivative support must lie wholly inside window.

Sparse source PTS schedules:.5,.75,1s and fixed repeating [.5,.75,1,1.25]s; first source observation at/after grid, skip missed slots. These are proxies, not measured Pi timestamp traces. No duplicated/interpolated pose. All intervening original missing-target/reseed barriers excluded, including hidden dropped frames. Saved identity used as an optimistic sampling diagnostic, NOT evidence sparse association will retain identity. This conservative offline barrier audit uses saved source-track history.

Seven channels: left/right knee angle and torso inclination(degrees), hip x/y velocity(image fraction/s), box width/height(image fraction). New35-column schema:14mean/std +7coverage +14separate mean/std missing flags. Mean requires>=2valid observations and coverage>=.5; std additionally>=3. Insufficient evidence remains null before training-only median imputation. A single difference does NOT become zero variability. All-missing training-column fallback0 and constant-scale1 are explicit persisted preprocessing policies; missing flags retained. No global fitting or row dropping.

## Support gate BEFORE training
Predeclared minimum20windows AND2distinct eligible segments per class. All passed. Each training3s support and contributing source observations lies in one eligible v005 segment; preliminary SE06 excluded. Walking interiors retain assistant-selected provenance.
|Class|Windows|Distinct segments|Eligible segment seconds|Occasions|
|---|---:|---:|---:|---|
|standing|1188|4|89.768|P001_SE05_SE06_SE07, P001_SE08_SE09|
|sitting|1043|3|76.918|P001_SE05_SE06_SE07, P001_SE08_SE09|
|lying|1169|3|85.878|P001_SE05_SE06_SE07, P001_SE08_SE09|
|moving|233|12|50.928|P001_SE10_SE11|

Segment seconds count each contributing eligible segment once, not sum of overlapping window durations. Sampling variants and overlapping windows are correlated, not independent samples. Same video/occasion retained as provenance, excluded from inputs. Moving only one occasion; one participant overall. No random split or independent performance claim.

## One fixed run
CPU float64 Linear35x4 (144parameters), seed42, threads1/interop1, full-batch Adam lr.03, default betas(.9,.999)/eps1e-8/weight_decay0, loss mean CE +.001/2*sum(weight?),1000steps, final weights; no early stop/search/retry.3633training rows. Objective1.441643 ->.013936 (logged pre-update at step1000). Saved final-step weights; reload logits exactly equal. Reported2.4481s elapsed covers optimization plus artifact evaluation/serialization section, NOT total sampling/preparation runtime. PyTorch emitted a scalar-conversion warning when logging loss; it did not fail or cause rerun. Support audit v001 failed before training due missing import, fixed; v002 audit passed. Only development_v001 contains TRAINING_STARTED.

## Same-timestamp comparison (include unavailable)
Full timelines use the SAME0.25s decision grid and terminal source support for both models within each cadence. Startup remains included. Denominators vary slightly between cadences because final sampled PTS differs; do not rank across cadences as identical timelines. Frozen strict uses original1s/.5s rules on the SAME sparse poses, freshly recomputed.
|Cadence|Decisions|Low-rate available|Strict available|Known-label decisions|Low-rate correct|Strict correct|Low-rate unknown|Strict unknown|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0.5|1840|1542|410|1289|1141|210|139|1067|
|0.75|1828|1523|0|1277|1125|0|139|1277|
|1.0|1828|1526|0|1279|1102|0|139|1279|
|irregular|1804|1290|0|1265|939|0|303|1265|

Label diagnostics use eligible label AT DECISION TIME (not only available windows). All known-label timestamps including startup/unresolved are retained; unknown is incorrect in correct/known ratios. Training requires entire3s support labeled, evaluation does not silently remove decisions whose trailing window extends outside the current eligible interval. Pending/excluded decision timestamps are not scored. This differs from prior sparse-replay whole-window labeling: comparison here recomputes strict on identical timestamps. Full true->predicted/unknown counts in confusions.json; this is development analysis, largely training-overlapping, not held-out accuracy.

## Coverage and missingness
|Cadence|Available windows|Hip vx median coverage|Hip vx mean missing|Hip vx std missing|
|---|---:|---:|---:|---:|
|0.5|1542|0.8320|16|16|
|0.75|1523|0.7467|21|24|
|1.0|1526|0.6667|12|1526|
|irregular|1290|0.7444|9|671|

Detailed per-channel mean/std missing counts and coverage in coverage_missingness.json; raw null vectors and masks in decisions.jsonl. At1s,3s window often has only2supported velocity differences, so std remains missing and is imputed using TRAINING median with explicit mask. Increased availability is not proof of preserved motion information. Irregular cadence has228insufficient_observed_span refusals, besides startup192 and unresolved94.

## Decision and limitations
Promising development availability only; DO NOT integrate live/package Pi yet. Next step is review existing wrong-label windows and source support/missingness, especially moving and irregular cadence, without tuning. Longer history introduces at least3s startup support plus grid/acquisition wait and inference time. At a grid deadline the next real processed observation may arrive later; this offline report is not sensor-to-screen latency. Mixed-posture transition windows can lag or disagree; no transition accuracy established.3s window does not create observations during gaps, and no output should be held across missing data.

Geometry depends on perspective/occlusion; sparse motion can alias walking, turns and adjustments. Model trained/evaluated on saved PT640 poses, not the live NCNN320 distribution. Power/undervoltage and live identity uncertainty are NOT simulated. No evidence of generalization to unseen participants/occasions/hospital settings.

## Verification and artifacts
6new tests +5sparse replay +12dataset regression tests =23PASS; pip checkPASS. Input annotation/timeline/target/training manifest and frozen classifier hashes checked before/after. No-future feature read, gap/barrier refusal, irregular selection, null std vs mean, training-only imputation and persistence verified. outputs/low_rate_development_v001_20261030/: config/schema, support, recomputed per-video/cadence features, decisions, coverage/missingness, confusion, model_state/preprocessing, loss_history and artifact_hashes. Existing versions preserved.

Executed once:
`.venv\Scripts\python.exe -m patient_monitoring.low_rate_experiment --config outputs/low_rate_experiment_preparation_v001_20261030/config.json --output-dir outputs/low_rate_development_v001_20261030 --train`
Do not rerun training without a new explicit task. M2 remains IN_PROGRESS.
