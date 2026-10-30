# Sparse development replay ? 2026-10-30

User-reported Pi low-rate run complete:76processed/39matched/1seed;63/75gaps>.5s;0available/149unavailable,117timestamp_gap. NCNN320 association1.5s, feature/window.5s. These are conversation evidence, not newly independently inspected Pi logs. No Pi execution here.

Method:16 videos from frozen dataset v005 inventory. Original saved target poses and quality masks, PTS/timebase verified against inspection timelines. Four anchored schedules select first observed frame at/after each due time; skip missed slots, no duplicate/interpolation. Actual gaps differ from nominal cadence. Recompute FeatureExtractor differences over selected actual dt, then existing make_window1s/stride.25/gap.5. Last sampled frame provides endpoint only; no invented tail support. Saved identities are retained to isolate temporal sampling: an optimistic identity-conditioned diagnostic, NOT proof strict sparse association works. Strict live/association, model/preprocessing/baselines untouched.

Every intervening original missing-target/reseed barrier is checked, including frames omitted by sampling. Conservative offline interval audit marks prior support unresolved when such a hidden barrier occurs. No joining separated tracks. This uses known saved-track history, not a claimed online implementation. v001 preserved but superseded: it marked the following temporal pair, yet could leave the preceding support interval available; v002 also excludes that span.

## Availability (all scheduled decisions; startup included)
| Nominal seconds | Available | Unavailable | timestamp_gap | unresolved | startup |
|---|---:|---:|---:|---:|---:|
|0.25|1674|170|0|106|64|
|0.5|410|1430|1262|104|64|
|0.75|0|1828|1663|101|64|
|1.0|0|1828|1664|100|64|

## Feature distribution/support
Unimputed raw aggregate columns; units: knee/torso degrees, hip velocity image fraction/s, box dimensions image fractions. Coverage is valid duration /1s. Nulls excluded only from numeric quantiles, never from missingness/coverage columns or model rows. JSON contains count/p10/median/p90 for all28columns; count difference exposes missingness. Training1183windows vs full-timeline available sparse windows have different class/video/support mixtures: these are UNPAIRED descriptive shifts, not pure causal sampling effect. At .75/1 there are no valid aggregated vectors, so all distribution quantiles null.

| Column | Training median | .25s median | .5s median |
|---|---:|---:|---:|
|hip_vx_per_s.coverage|0.944033|0.751967|0.500000|
|hip_vx_per_s.std|0.030646|0.012554|0.000000|
|torso_deg.std|0.798237|0.844138|1.026860|
|left_knee_deg.std|2.594912|2.216966|1.793511|
|box_height_fraction.mean|0.561477|0.568266|0.331230|

Velocity coverage falls from~.944 to~.752/.5; at .5s velocity std median0 reflects a single supported difference, not proven absence of motion. Fewer observations lose variability information. Available windows are survivor-biased. No imputation/scaler fit or threshold adjustment.

## Labels: eligible whole support only
v005 eligible confirmed static clips and authorized assistant-selected moving interiors only. Legacy direct flag does not imply independently exact user boundaries. Preliminary SE06 excluded. Support must fit wholly in one eligible half-open segment. Predictions outside labels are diagnostics only. Startup has no eligible full1s interval, so it remains in availability totals but not scored-label totals. Correlated overlapping windows/training-overlapping development, not independent accuracy. Unknown retained.

| Seconds | Eligible scored windows | Correct | Wrong known label | Unknown |
|---|---:|---:|---:|---:|
|0.25|1176|1171|5|0|
|0.5|1174|175|7|992|
|0.75|1162|0|0|1162|
|1.0|1164|0|0|1164|

## Recommendation, no changes implemented
Association lifetime alone cannot restore valid temporal support. Preserve UNAVAILABLE at slow cadence. Nominal2Hz is not robust: first-at/after sampling creates >.5s intervals (303 in this replay);4Hz proxy has margin and much greater availability, but it is not a Pi performance guarantee. Existing pipeline needs reliably <=.5s actual gaps, not merely a mean rate above2Hz.
If that cannot be achieved, a separate future low-rate feature/model design must explicitly represent sparse evidence, assess reduced velocity support and be validated on matched sampling distributions before any window/gap/preprocessing change or retraining. Static-posture-only geometry is a possible separate research path, not a replacement for moving recognition. No gap relaxation, window change, training, inference rerun, deployment or new recording request here.

Checks:5new sampling/geometry/gap/annotation tests +12dataset regression tests PASS; pip check PASS. Input manifest/output hashes and frozen classifier hashes verified before/after. Source pose index/PTS alignment checked for all16videos. Output recomputed feature rows retain source_frame_index/PTS/timebase; decisions and distributions saved per version. M2 remains IN_PROGRESS.

Command:
`.venv\Scripts\python.exe -m patient_monitoring.sparse_replay --dataset outputs/SE11_V002_annotations_dataset_v005_20261021/dataset --model-dir outputs/supervised_action_development_v001_20261021 --output-dir outputs/sparse_replay_v002_20261030`
Choose a new output path for any rerun; no overwrite.
