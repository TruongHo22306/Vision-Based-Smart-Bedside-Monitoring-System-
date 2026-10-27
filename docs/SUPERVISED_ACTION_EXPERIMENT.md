# Execution update: first supervised development run completed

The single run authorized on2026-10-21 completed. The preparation text below is retained
as historical design; its not-executed statements describe that earlier stage only.
Implementation: src/patient_monitoring/supervised_action.py; tests: tests/test_supervised_action.py.
Artifacts: outputs/supervised_action_development_v001_20261021/.
Report: outputs/supervised_training_implementation_audit_20261021/TRAINING_REPORT.md.
13new tests +12dataset tests PASS before real run; pip checkPASS.1000steps, CPUfloat64,
116parameters. Training/resubstitution1182/1183; not held-out accuracy. Frozen model,
preprocessing/schema/config hashes saved; no second run. Reload logits identical.

Actual command (existing output is protected; reproduction needs a NEW directory):
```powershell
.venv\Scripts\python.exe -m patient_monitoring.supervised_action train --config outputs/supervised_training_preparation_v001_20261021/experiment_config.json --output-dir outputs/supervised_action_development_v001_20261021
```
Next: new-occasion all4class source annotation and frozen-model evaluation, no refit/tuning.

---

# First supervised action experiment: preparation only

## State and exact next task
No classifier, preprocessing fit, training or inference has been run. No package changes.
Prepared artifacts: outputs/supervised_training_preparation_v001_20261021/.
Next implementation task: add `patient_monitoring.supervised_action` with a new-output-only
`train --config PATH --output-dir NEW_DIR` interface (PROPOSED, does not exist yet).
Implement schema validation/vectorization, training-only preprocessing, fixed linear model,
state persistence and diagnostics below; add focused tests; then run the single authorized
development experiment only after a separate training instruction. Do not alter datasets,
frozen v002/v003, pose pipelines, annotations or PROJECT_PLAN.

## Actual v005 representation
1183 windows: standing345, sitting297, lying333, moving208. Existing causal trailing1s,
stride0.25s; correlated overlaps are not independent samples. Source annotations differ:
static whole-clip user feedback versus assistant-selected moving interiors authorized by
user, with prior prediction exposure. Legacy direct_confirmed names do not erase this.

Each window has seven channels, each with mean, std, valid, valid_duration_seconds,
coverage, valid_observations and missing_reasons. Mean=sum(value*valid_seconds)/sum(valid_seconds);
std is weighted population standard deviation with that same denominator. Static values
have source display support intersected with window; backward velocities contribute only
when their full previous-to-current PTS interval lies inside it. No interpolation/smoothing.

|Channel|Units of mean and std|
|---|---|
|left_knee_deg, right_knee_deg|degrees; saved hip-knee-ankle geometry|
|torso_deg|degrees relative to upward image vertical|
|hip_vx_per_s, hip_vy_per_s|image-width / image-height fractions per second|
|box_width_fraction, box_height_fraction|fractions of image width / height|

Vector order is frozen in feature_schema.json:14 mean/std scalars, then7 coverage scalars,
then7 missing indicators (=not valid), total28. Only model_inputs allowlisted fields enter X.
Labels are y only. IDs, paths, frame/PTS clocks, occasion, clip context, baseline predictions,
observation counts, missing-reason strings and redundant durations are excluded from X.
No raw17-joint requirement or feature recomputation. Validate all rows; fail a malformed run
with row/reason, never silently drop it. Finite valid values, paired nulls, nonnegative std,
boolean masks, exact channel set, consistent coverage are required. Allow1e-9 numeric
coverage tolerance: observed max1.0000000000000007 is rounding; source remains unchanged.

Audit: no aggregate mean/std nulls among1183 windows. This does NOT imply all joints valid.
Knee coverage minima0.144033/0.2 (10partially covered windows per knee), torso0.566667
(7partial), hip velocity0.863967-0.992067 (all1183partial due to derivative support).
Boxes effectively full coverage. No preprocessing parameters have been fitted in this task.

## Proposed model and dependencies
Multinomial logistic regression: Torch Linear(28,4,bias=True),116parameters, CPU float64.
A small transparent model is sufficient for the first plumbing/learnability experiment;
no hidden layer, temporal smoothing, learned threshold or hyperparameter search.
scikit-learn is not installed/importable. Torch2.9.1+cpu and NumPy2.2.6 are installed;
Torch is already a declared dependency. No scikit-learn installation is needed. Existing
Linear/Adam APIs inspected locally; pip check PASS. Full package snapshot saved.

## Missingness and scaling, fit only on training rows
For the14 mean/std columns, estimate each observed training median and impute missing values
with that median in the derived matrix only. Retain7 fixed missing indicators for every
channel even if missingness was absent in training. If a whole training column is missing,
use explicit fallback0, record the condition, retain the column; never infer a median.
Then fit population mean/std on the imputed training matrix. Constant columns use scale1;
all-missing columns center0/scale1. Store all medians, fallback flags, centers and scales.
Coverage and missing indicators pass through unscaled; keep tolerated rounding excursions.
No window dropped, including all-missing rows; retain and separately count such cases.
No source nulls rewritten. Future transforms reuse saved parameters without refitting.
All training missingness flags currently0, so behavior on new missingness is not validated;
confidence/softmax is not calibrated certainty. No new abstention threshold in this design.

## Fixed development run (not executed)
Use all1183 eligible v005 windows as development training rows; no heldout score, CV,
random split, early stopping or tuning. Seed42 for Python/NumPy/Torch; deterministic
algorithms, CPU threads1/inter-op1 explicitly set and recorded in the future process.
Full-batch Adam lr0.01, betas(0.9,0.999), eps1e-8, weight_decay0, foreach/fusedFalse,
1000fixed steps. Objective: mean unweighted cross entropy +0.001/2*sum(W squared),
intercept not penalized. Use final step, not best training result. These choices are
unvalidated defaults; report loss/convergence behavior, do not silently rerun or adjust.
All windows equally weighted; overlap and unequal video lengths therefore affect fitting.
Train diagnostics: loss at first/every100/final step, resubstitution confusion matrix,
per-class P/R/F1 with denominators/null undefined, counts and per-occasion/video breakdown.
Metadata is used only for diagnostics. These are NOT held-out accuracy/generalization.
Output every window prediction for audit, never promote it to an annotation.

## Group-disjoint feasibility
|Occasion|Standing|Sitting|Lying|Moving|
|---|---:|---:|---:|---:|
|P001_SE05_SE06_SE07|130|83|80|0|
|P001_SE08_SE09|215|214|253|0|
|P001_SE10_SE11|0|0|0|208|
All6 nonempty two-way group partitions were audited; none has all4 classes on both sides.
Holding out SE10_SE11 removes moving from training; retaining it removes moving support
from any other heldout occasion. Static-only holdout is possible but cannot evaluate all4
classes. Never split one occasion, even across devices/videos. No split file is created.
Class is confounded with occasion: box size, coverage and viewpoint can encode context
even without explicit IDs. Missingness/scale choices cannot solve this data limitation.
Only one participant, all development data. A new occasion is needed for evaluation;
even that would evaluate new conditions for P001, not unseen patients/hospital use.

## Reproduction and implementation acceptance
experiment_config.json pins dataset bytes, dataset manifest/config, feature-schema hash,
seed/model/optimizer/preprocessing rules. config_hash.json records exact file SHA256.
Before a future run verify hashes, complete status, expected labels and schema ordering;
check class count, unique window keys and same-occasion grouping. Save actual source code,
package/Python versions, config, dataset/schema hashes and actual CPU/dtype/thread counts.
Persist model_state.pt (state_dict only), JSON preprocessing/schema/class order/config,
run_info, loss/diagnostics and prediction JSONL in a NEW directory; incomplete on failure.
Reload using explicit architecture and weights_only=True; refuse incompatible schema/hash,
round-trip test equality of predictions and close logits on saved examples. Do not pickle
an executable whole-model object. Cross-version bitwise reproduction is not guaranteed.
Focused future tests: exact28-column order; metadata changes cannot change X; null/mask
validation; training-only medians/scales, all-null/constant columns, unseen missingness;
no rows dropped; saved reload; schema/hash mismatch and no-overwrite. These tests are
planned, not claimed as implemented or run. No model metrics are available yet.

## New-occasion evaluation recording checklist
- Freeze trained weights, preprocessing, feature schema and configuration hashes BEFORE
  reviewing the new occasion. Reserve all its videos as one evaluation group; no tuning.
- Record actual subject/session/device/camera setup/occasion; do not invent registry entries.
  Use fixed camera, full-body margin, stable light and retained originals/timelines.
- Record10-20s each: standing still; standing with arm/body adjustments but no walking;
  sitting still and with arm movements; lying still and with voluntary adjustments;
  walking lateral and toward/away; walking with a pause then restart. All four classes
  must be present in this occasion, not separated into class-specific occasions.
- Keep transitions/turns/pauses/entry/exit explicitly annotated or uncertain. Record intent
  is context only. Annotate source video without predicted-label overlays where feasible,
  before inspecting model output; derive exact PTS bounds with the established policy.
- Preserve failed/unresolved cases and report scored versus excluded coverage. Never
  silently drop hard windows. No performance claims until confirmed annotations exist.
- If results drive any tuning, this occasion becomes development data; reserve another
  untouched occasion for evaluation. Collect more participants for unseen-person claims.
