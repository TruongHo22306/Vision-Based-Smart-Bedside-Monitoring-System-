# Acquisition action development v001 (2026-11-03)

## Scope and freeze
Exactly ONE authorized training run completed. No split, tuning, retries, YOLO, annotation edits or live-default change. All figures are development/resubstitution diagnostics on one participant; overlapping windows and cadence variants are correlated. Moving support comes from one recording occasion. No generalization or calibrated-confidence claim.

Audit reverified 1,311 eligible windows (432 standing, 378 sitting, 423 lying, 78 moving), source/config/schema hashes, original annotation CSV provenance and timeline alignment. Entire training supports and velocity predecessors are inside eligible segments; hidden target-loss/reset barriers retained. SE06 preliminary labels excluded. Assistant-selected moving interiors retain their provenance despite legacy direct-confirmation fields.

Preparation v001 stopped BEFORE training due to Windows default text decoding of annotation notes. UTF-8-sig was made explicit in the new audit loader; no labels/bounds/source files changed. Successful immutable preparation is outputs/acquisition_training_preparation_20261103_v002/.

## Inputs and optimizer
35 columns from acquisition_features_v002: 14 means/stds, seven coverage values, 14 independent missing flags; exact frozen order in feature_schema.json. No ID/path/time/occasion/baseline inputs. Only eligible training rows fit median imputation and population centering/scaling; all-missing median fallback zero and constant scale one are explicit. Coverage and missing flags pass through unchanged.

Linear(35,4), CPU float64, seed42, intra/inter-op threads1, full-batch Adam lr0.03/betas(0.9,0.999)/eps1e-8/weight_decay0; objective cross entropy + 0.001/2 times squared weight norm; bias not regularized. 1,000 fixed steps, final weights. These optimizer settings match the previous low-rate experiment; feature semantics, row population and fitted preprocessing differ. Config was hashed before fitting: cda17dd4c70c1e10a424d951101c025142a22a31dd5fd6d7bc05f3873a36eb0d.

Training support by cadence (standing/sitting/lying/moving): .5s=158/138/155/29; .75s=105/92/103/19; 1s=80/70/78/14; irregular=89/78/87/16. Total 480/319/242/270.

## Actual results
Training/resubstitution confusion is diagonal 432/378/423/78; correct 1311/1311, unavailable0. Every class P/R/F1=1 with respective support/predicted denominators 432/378/423/78. This is not held-out accuracy.

Objective 1.4338921593 -> 0.0133700257; optimization only 0.7556094 seconds, saved state_dict 3013 bytes. Loss history retains every pre-update objective and final post-update objective. Torch emitted a scalar-conversion warning when logging loss; run stayed finite and completed, no retry.

## Replay on saved target poses
Shared pure/stream features and original/reloaded model agree exactly on all 2131 available decisions; all 2551 feature records re-audited. Known-point diagnostics below score only authorized annotation timestamps, including unavailable as incorrect. Replay windows need not lie wholly inside a label segment; this differs from training eligibility. No truth inferred for pending regions. Counts are decisions, not duration-weighted or independent frames.

|Cadence|All available/total|Known points|Correct|Wrong label|Unavailable on known|Correct/known|
|---|---:|---:|---:|---:|---:|---:|
|0.5|780/928|653|581|2|70|88.9740%|
|0.75|518/620|434|384|3|47|88.4793%|
|1.0|394/469|329|290|4|35|88.1459%|
|irregular|439/534|375|325|4|46|86.6667%|

### Per-class replay metrics
P=TP/predicted count; R=TP/actual known count (includes unavailable); F1=2TP/(actual+predicted). Zero denominators remain null. Full confusion and per-video/occasion training metrics: ../acquisition_action_development_20261103_v001/metrics.json.

|Cadence|Class|TP|Predicted|Actual|Precision|Recall|F1|
|---|---|---:|---:|---:|---:|---:|---:|
|0.5|standing|158|160|182|0.9875|0.8681|0.9240|
|0.5|sitting|138|138|156|1.0000|0.8846|0.9388|
|0.5|lying|155|155|173|1.0000|0.8960|0.9451|
|0.5|moving|130|130|142|1.0000|0.9155|0.9559|
|0.75|standing|105|108|121|0.9722|0.8678|0.9170|
|0.75|sitting|92|92|104|1.0000|0.8846|0.9388|
|0.75|lying|103|103|115|1.0000|0.8957|0.9450|
|0.75|moving|84|84|94|1.0000|0.8936|0.9438|
|1.0|standing|80|84|92|0.9524|0.8696|0.9091|
|1.0|sitting|70|70|79|1.0000|0.8861|0.9396|
|1.0|lying|78|78|87|1.0000|0.8966|0.9455|
|1.0|moving|62|62|71|1.0000|0.8732|0.9323|
|irregular|standing|89|93|105|0.9570|0.8476|0.8990|
|irregular|sitting|78|78|90|1.0000|0.8667|0.9286|
|irregular|lying|87|87|99|1.0000|0.8788|0.9355|
|irregular|moving|71|71|81|1.0000|0.8765|0.9342|

### Notable errors
All 13 wrong-label decisions are moving -> standing; none standing -> moving within scored points. Exact PTS are retained below. These are acquisition decisions, not confirmed error-duration intervals between samples.

|Video|Cadence|PTS exact|Seconds|
|---|---|---|---:|
|P001_SE10_V001|0.5|540013/30000|18.000433|
|P001_SE10_V001|0.5|278167/15000|18.544467|
|P001_SE10_V001|0.75|540013/30000|18.000433|
|P001_SE10_V001|0.75|7501/400|18.752500|
|P001_SE10_V001|1.0|540013/30000|18.000433|
|P001_SE10_V001|1.0|661457/30000|22.048567|
|P001_SE10_V001|1.0|115283/5000|23.056600|
|P001_SE10_V001|irregular|540013/30000|18.000433|
|P001_SE10_V001|irregular|7501/400|18.752500|
|P001_SE10_V001|irregular|667697/30000|22.256567|
|P001_SE11_V001|1.0|29|29.000000|
|P001_SE11_V002|0.75|484/15|32.266667|
|P001_SE11_V002|irregular|244/15|16.266667|

Missing velocity std occurs for each vx/vy column in 162/1311 training windows: 134/242 at 1s and 28/270 irregular; none in .5/.75 eligible windows. All other 12 numeric training columns are present. Separate missing flags and training-only medians are saved; missing std is not converted to observed zero variation. Full missingness.json retained.

Replay unavailable totals by cadence: startup 98/67/49/63; unresolved/invalid timestamp 50/35/26/32. Irregular schedule is simulated repeating .5/.75/1/1.25s mapped to real source PTS, not a measured Pi trace. Saved identity conditions replay; it does not verify live association. Training overlap and common participant/occasion prohibit an independent evaluation claim.

No direct numerical old/new-model comparison: old .25-second decision grid differs from acquisition-anchored endpoints. Raw accuracy or availability differences would mix schedules and populations.

## Verification and next step
23 focused tests PASS before training (vector/mask/schema, training-only fit, missing/constant columns, persistence/hash refusal, feature causality/gaps, strict/live-display regression). pip check PASS. Actual weights_only reload has exact logits and labels; 2131 offline/stream logits/labels match. All 34 protected old artifact/code hashes plus all new output hashes reverified after training.

Ready for a SEPARATE experimental Pi integration task with explicit new feature version/model pin, expiry/reset tests and measured timing; not ready to replace strict/low-rate v001 or claim Pi availability. Existing conditional latency-budget estimates are not Pi execution, and stale contributing channels still constrain TTL. No packaging/live switch in this task.

## Executed command (do not rerun for this task)
PowerShell, project root:
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m patient_monitoring.acquisition_action train --config outputs/acquisition_training_preparation_20261103_v002/experiment_config.json --output-dir outputs/acquisition_action_development_20261103_v001
```
Existing output directories are refused. Model/preprocessing/schema/config hashes are in frozen_manifest.json; all result hashes in output_hashes.json. Model SHA256 c9d4eb488d3f197ee98101e2f19d5b30e99871c25a6d237d37c9b0bc7329c29f. Wrong feature versions are refused even if dimension=35.
