# Acquisition features v002 - experimental, 2026-11-03

Implementation: src/patient_monitoring/acquisition_features.py. Shared pure aggregate(prefix,T) and streaming AcquisitionFeatures.step(selected_target,hidden_barrier=False); no classifier, no live-default wiring, no frozen-model compatibility. Source PTS/time_base and source frame index preserved. Live caller must supply host-acquisition PTS in the same clock domain as freshness checks; never use completion time as acquisition.

## Frozen experimental definition
Config/schema hashes in outputs/acquisition_features_v002_20261103/run_info.json.3s right-closed window (T-3,T], ending at a REAL observation. One decision on first acquisition reaching each .25s opportunity; missed opportunities skipped, never retroactively assign current input to an earlier decision timestamp. Minimum3 observations,1.5s observation span,3s continuous history, max adjacent gap1.5s. These are development choices, not optimized parameters.

Five geometry channels: left/right knee angle and torso inclination (degrees), box width/height (image fractions). Equal weight1 per valid actual observation in (T-3,T]. Mean=sum(x)/n, population std=sqrt(sum((x-mean)^2)/n). Coverage=valid observation count/all observed points in the window. No claim this is duration coverage; no future duration assigned to current pose.

Two velocity channels: hip image-fraction displacement per second from the unchanged FeatureExtractor backward differences. Include only real consecutive valid intervals [t_prev,t] wholly within [T-3,T], ending in (T-3,T]. Weight=dt seconds, mean=sum(dt*v)/sum(dt), population std=sqrt(sum(dt*(v-mean)^2)/sum(dt)). Coverage=sum(valid dt)/3s. Interval with left endpoint exactlyT-3 is included; an interval crossing the left edge is excluded, not trimmed/interpolated. Current derivative is included if valid. This explains why real near1Hz sampling may still give fewer than3 intervals.

Mean requires>=2 valid observations, coverage>=.5 and latest contributing valid observation age<=1.25s. Std additionally requires>=3 valid observations/differences; one difference is NOT evidence of zero variability. Raw moments are diagnostic; rejected vector values remain null with explicit mean/std missing flags. Per-channel latest_valid_timestamp is set only by valid numeric samples, never by a missing current sample. Coverage/quality indicators are not new posture evidence.

Vector order:14mean/std values,7coverage values,14mean/std missing flags =35. Same dimensionality as v001 does NOT mean same feature semantics/preprocessing. No imputation, global fit, scaling, score, label prediction or old-model inference is applied.

Window availability here means temporal gates pass AND at least one channel has usable mean; this is a feature-extraction status, not a validated classifier requirement. Other channels remain missing. expires_at=min(latest_valid_timestamp+1.25s over channels actually contributing a mean/std). freshness(record,now) fails before acquisition, for unavailable records, or after that deadline; exact equality allowed. Missing channels do not renew any used channel's support. A future model may not promote a fresh box alone to guaranteed posture validity; training/evaluation must retain these masks.

Missing/unresolved target, nonincreasing/invalid timestamp, seed/reset, hidden source losses, explicit identity/context break and gaps>1.5s cannot be bridged. Rebuild3s support after barriers. Offline hidden-loss audit checks EVERY saved source row between selected observations; it conditions on saved identity, not proof that a live tracker detects hidden losses. Adapter history is time-pruned plus capped256rows; high rates beyond retained support become unavailable rather than fabricating history. Pure aggregator rejects future rows.

## Actual replay
Command:
`$env:PYTHONPATH='src'; .venv\Scripts\python.exe tools/replay_acquisition_features.py --output-dir outputs/acquisition_features_v002_20261103`
Use a NEW output path for any future run. Existing path is complete and must not be overwritten.

16 saved development videos, true PTS selection at/after due time, geometry/velocities recomputed from selected target poses. Regular targets .5/.75/1s; irregular cycle .5,.75,1,1.25s is SIMULATED, not Pi timing. No interpolation/pose duplication. Pure/stream feature vector+channel parity checked on all2131 temporally valid windows.

|Cadence|Decisions|Feature available|Startup|Unresolved/invalid|Eligible labeled windows|
|---|---:|---:|---:|---:|---:|
|.5|928|780|98|50|480|
|.75|620|518|67|35|319|
|1|469|394|49|26|242|
|irregular|534|439|63|32|270|

Per-video features.jsonl/windows.jsonl preserve every selected observation and decision vector/channel support/masks. summary.json contains missingness/coverage/age populations per channel, per_video.json counts, training_support.json labels/occasion groups, input_hashes.json immutable inputs. Missing invalid rows remain represented; startup records have no vector.

Velocity std missing vx/vy respectively:6/780 at.5,6/518 at.75,174/394 at1,41/439 irregular. Mean velocity coverage .9223/.8866/.8501/.8526. Geometry box coverage1 in all temporally valid windows; knee/torso missingness remains explicit. No claim of joint accuracy.

1311 eligible windows:standing432 from4 segments,sitting378 from3,lying423 from3,moving78 from9. Group totals: SE05/06/07 standing158/sitting104/lying98; SE08/09 standing274/sitting274/lying325; SE10/11 moving78. No preliminary SE06 labels promoted. Entire3s support and ALL contributing source indices, including backward-pair predecessors, must lie in a single existing eligible segment; T must be strictly before its half-open annotation end. Draft/pending/excluded regions never become training labels. SE02/03 and occasion grouping retained; no split. These cadences/windows overlap and are correlated, not1311 independent recordings.

## Conditional Pi latency budget, NOT a Pi result
outputs/acquisition_features_audit_20261103/latency_budget_estimate.json uses the50 received support-result delays already saved by the prior Pi audit. Cartesian scenarios combine each available development feature window with each OLD processing/UI delay. They are not aligned pose replay or observations of the new implementation/model. New-model runtime is UNKNOWN. V001's processing time includes its own old feature/classifier work; results are a budget stress estimate only.

|Cadence|Scenarios|Fresh at estimated completion|Fresh at estimated UI|
|---|---:|---:|---:|
|.5|39000|38223|27292|
|.75|25900|25116|18072|
|1|19700|19200|13824|
|irregular|21950|21393|15356|

Denominator is AVAILABLE feature windows times50; startup/unresolved windows excluded from this conditional table and reported above. No accuracy or availability-improvement claim. Mean remaining support budget at acquisition is1.225-1.229s, but minimum .034s for irregular due to an older contributing channel. Reducing mandatory one-sample geometry lag does not eliminate expiry.

## Tests and readiness
43 focused tests PASS:9new acquisition tests plus34 frozen/strict/parity/expiry/reset regressions. pip check PASS. Frozen code/models/config/annotations/datasets/PROJECT_PLAN hashes remain unchanged. Dataset manifest/input hashes verified, new output hashes saved. No YOLO, model inference for replay, training, dependency changes or Pi execution.

Technically READY to prepare one separately configured DEVELOPMENT training experiment: all four classes have eligible support,35-column missingness/order defined, shared causal adapter and parity verified. NOT ready to replace frozen classifiers or deploy. Before an authorized training run, freeze a NEW experiment config and training-only preprocessing/missing-value policy for this schema; never reuse old preprocessing. Moving support is limited to one occasion/participant, so group-disjoint class-complete evaluation remains unavailable. No fitting or split was performed here.
