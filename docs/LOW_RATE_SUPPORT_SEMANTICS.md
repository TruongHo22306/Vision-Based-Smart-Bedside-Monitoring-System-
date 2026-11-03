# Frozen low-rate support semantics audit - 2026-11-02

Inputs: the three summary.json, diagnostics.jsonl, timing_report.json files under D:/patient-monitoring-transfer/pi-scheduling-results. Read-only hashes and per-result reconstruction are saved in outputs/support_semantics_audit_20261102_v001/. Timing report recalculated exactly using the reporter. No missing log records.

## Measured findings
82 pose calls;143 historical available decisions.50 distinct received pose results have non-null support. All50 support timestamps equal the preceding processed acquisition in the same target epoch.48/50 already exceed1.25s at worker completion;49/50 exceed it at FIRST UI assessment. One initially fresh result later expires too (the repeated-UI summary therefore reports50 eventual expirations, not49). First-result assessments total79 including seed results, not50 or82. Only1 labeled submission/.050453831s clipped-valid duration; not accuracy or proof of screen visibility.

Same50-result population means:
- Previous-support to current acquisition: .697700555s.
- Current acquisition to worker completion: .744855613s (pre-start .092379635; predict .547738291; classifier .095590784; other .009146902).
- Support age at completion:1.442556167s.
- Completion to UI:.291918537s; support age at UI:1.734474704s.

Frame186: support age1.242726228s at completion,1.877028805s at UI. Frame291:1.180762153s ->1.199546169s, only first-UI fresh result. Even zero completion-to-UI delay could save at most one additional received result under the current measured timestamps:48 were already expired. Faster UI alone is not the primary solution for this run. Worker-start age for all82 calls averages.085169232s; residual shutdown frame count1, not a growing queue. No cross-run speed advantage is inferred.

## Exact contract and why the previous sample is used
1. live_low_rate.py:71 schedules ends E<=latest acquisition T, at .25s grid. If E<T, using current numerical pose at T would look ahead relative to E. Backdating the current frame to E is prohibited.
2. low_rate_experiment.py:38-45 uses observations from times[:-1], start<=t<E, and calls make_window(rows[:-1],times,...). The latest timestamp supplies a known right bound for the prior observation; its numeric features are not passed.
3. supervised_dataset.py:102-140 selects rows with t_i<E and t_(i+1)>start, then assigns geometry duration [t_i,t_(i+1)) intersected with [start,E). At E=T, the current geometry has zero forward duration inside the window and no observed subsequent endpoint. It cannot have positive weight under this contract.
4. Velocity is already a valid causal backward difference (p_i-p_(i-1))/(t_i-t_(i-1)); its interval is [t_(i-1),t_i]. Mathematically the derivative ending at T could be used when E=T, with no future information. BUT the common row-selection step excludes row i=T before the velocity-specific support calculation. Existing training therefore also omits that derivative. Example1Hz [0,1,2,3], E=3: current v3 is valid but only v1/v2 enter, so2 velocity observations, std missing under the frozen3-observation requirement. Including v3 changes coverage/mean/std and missing flags; this is not a UI-only fix.
5. live_low_rate.py:82 records the timestamp of the last numerical observation actually used. Current pose is used for identity/continuity gating, which does not make the classified numeric window current. Advancing support to T without changing the vector would falsify freshness.

This is a shared training/offline/live lag/conservative aggregation contract, not evidence of wrong live sample indexing or an expiry error. A derivative-endpoint design correction could be legitimate for a NEW feature version, but cannot be represented as a parity-preserving patch to the frozen input semantics. Merely changing < to <=, counting a zero-duration current observation, inventing a next timestamp, or moving the window to wall-clock completion changes weighting/coverage/missingness or fabricates support. No such edit made.

## Preferred next proposal (NOT implemented/trained/authorized for production)
Define a separately versioned acquisition-anchored causal representation:
- Decision window ends at the real latest acquisition T, (T-3s,T], evaluated at the first observed pose meeting the .25s scheduling opportunity; never publish before its computation finishes or rewrite old grid decisions.
- For the existing five geometry channels (knees, torso, box width/height), compute observation-weighted mean/std over valid actual observations including T. Explicitly change coverage to observed valid-count / observed-count for geometry; retain counts and missing flags. This is a new weighting/coverage contract even if the vector remains35 columns.
- For hip vx/vy, aggregate valid BACKWARD intervals wholly contained in the window, including the interval ending at T, weighted by their actual durations. Do not extrapolate a current position or interpolate a missing interval. Std requires enough real differences, not one difference treated as zero variation.
- Keep masks, gap1.5s, age1.25s, min3observations and causal reset barriers explicit; record latest valid timestamp PER channel. A current invalid/unused pose must not refresh support just because it arrived. Availability/freshness policy for required evidence must be frozen in the new specification before training.
- First produce an isolated support/missingness audit from saved development poses and real/simulated acquisition schedules, with no fitting. Audit eligible labeled training support and full annotation/track containment. Only after review/authorization fit NEW preprocessing and a NEW model version; do not reuse the frozen model as if the new vectors were equivalent. Keep existing classifier and strict path as references. No random splits or independent accuracy claim.

This proposal attacks the mandatory one-sample lag; it does not guarantee classification accuracy or that processing finishes before TTL. A separate incremental/cached implementation might reduce compute with identical vectors, but it cannot make the current geometry contribute under the frozen forward-duration convention. No justified semantics-preserving incorporation of the latest observation was found, so runtime is unchanged and no new Pi run is requested.

## Actual checks
25 tests PASS:3 new semantic boundary tests,5 frozen offline/live parity tests (vector/missing flags/logits/labels),4 display tests and13 async/reset/expiry/accounting tests. Synthetic tests show changing current geometry or its valid backward velocity does not change the frozen vector, while changing prior contributing geometry does; irregular grid excludes future observation. Three-file timing report equality,50 previous-support mappings,48completion/49UI expiry counts all verified. Existing runtime/model/preprocessing/config/PROJECT_PLAN/dependencies and input logs hashes preserved. No YOLO, retraining, TTL/gap change or Pi execution.
