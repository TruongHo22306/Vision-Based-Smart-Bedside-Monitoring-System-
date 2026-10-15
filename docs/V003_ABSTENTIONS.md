# V003 abstentions: exact implementation and bounded development review

Frozen configuration SHA-256: `9cd5684ef52a6660c6ac64d004983909e98feebc8fc7b7fdc6a428c9a6c303e3`.
Implementation: `action_baseline_v3.py`, `BaselineV3.step` and `landmarks`; inherited support: `action_baseline.py`, `Baseline.step/decide`; non-upright policy: `action_baseline_v2.py`.
This task changes no rules, thresholds, models or dependencies. V002 remains frozen reference; v003 experimental.

## Equations, units and ordering actually implemented

Required for the new motion window: valid left/right hips (indices11/12), shoulders(5/6), ankles(15/16). H=(hipL+hipR)/2 and S=(shoulderL+shoulderR)/2 in pixels; L=norm(S-H)>1e-9pixels. Saved feature valid flags and ankle passes_quality_gate must pass; finite values required. Body coordinates are (joint-H)/L. Saved joint quality originally requires finite confidence>=0.5 and strictly interior image coordinates; no rethresholding here.

Keep samples with t>=now-1s, where t=PTS*time_base. All samples must form an uninterrupted valid sequence. Need endpoint span dt>=0.8s and at least3intervals. No interpolation at the trailing boundary. With a,b the oldest/current samples, Lbar=(L_a+L_b)/2:

```
vH = norm(H_b-H_a)/(Lbar*dt)       # torso lengths/second
vS = norm(S_b-S_a)/(Lbar*dt)       # torso lengths/second
vL = abs(log(L_b/L_a))/dt          # 1/second
D_i = (ankleL_i-ankleR_i)/L_i     # torso lengths; hip centering cancels
R = sqrt((max D_x-min D_x)^2 + (max D_y-min D_y)^2)
T = (min(vH,vS)>=0.10) OR (vL>=0.05)
G = R>=0.15
```

Inherited v002 prerequisites are unchanged: six support channels (torso inclination, two knee angles, box aspect, hip velocity, box-size velocity), each current value valid, >=0.8s covered in the trailing1s window and >=3 contributing intervals. Static support uses time-weighted right endpoint samples with both endpoint masks; velocity support uses the norm of the time-weighted vector mean. Thus knees(13/14) and valid box also remain indirect requirements. Missing inherited support returns the v002 unknown reason before applying the new locomotion rule.

Inherited geometry: upright inclination<=30degrees; sitting max(knee angles)<=135degrees and upright; standing min(knee angles)>=155degrees and upright; lying60<=inclination<=120degrees and box width/height>=1.2. V002 defines motion high if hip OR box speed>=0.03 image-fraction/s, quiet if both<=0.015, otherwise indeterminate. That old motion remains diagnostic in the result, not the deciding signal in the changed upright branch.

Exact decision order:

1. Compute v002 result/support and the new landmark window.
2. If v002 posture evidence is sitting/lying, return its decision unchanged, even if the new landmark window is invalid.
3. If any inherited support is invalid, or torso support exceeds30degrees, return v002 unchanged.
4. Otherwise invalid landmark window => unknown (`insufficient_contiguous_landmark_evidence`).
5. Ready window with T AND G => moving.
6. Otherwise standing geometry AND NOT T AND R<=0.10 => standing.
7. Otherwise unknown (`locomotion_evidence_inconclusive`).

There is deliberately an ambiguity band0.10<R<0.15, but unknown is not limited to that band: T without G, or G without T, also causes abstention. The new window does not infer actual steps. Body/box movement, scale change and joint confidence are not accuracy or proof of locomotion.

## Reset conditions and latency

The landmark deque clears on every inherited reset: segment start; source/video/stream/dimension change; invalid or nonincreasing PTS; invalid/nonconsecutive frame index; unresolved target; reseed/explicit track break; changed track/association identifier; invalid saved temporal_pair or reference discontinuity; actual gap>0.5s. It also clears whenever required landmark data, target or time is invalid. A current valid row starts a fresh window after a reset; an invalid row is not skipped to bridge the gap.

No future-frame lookahead or debounce is added. **This does not mean zero decision latency**: initial/rebuilt evidence requires>=0.8s, potentially longer when data remain invalid; a1s trailing ankle range retains earlier changes. Exact onset/offset latency cannot be measured with available approximate/interior annotations.

## Findings: does resetting block posture?

- SE05_V002: all94 post-startup unknowns have ready landmark windows AND standing geometry. No missing joints/reset explanation. They fail the locomotion/quiet rule:35 have no transport and ankle range in the ambiguity band;26 have lower-body range>=0.15 without transport;33 have transport without sufficient lower-body range (18of these have quiet ankles). All33transport flags here come from torso-scale speed, because hip speed<0.10. These are implemented predicates, not proof the subject moved.
- SE06 standing interiors:18unknowns, all ready, no landmark missing/reset. The9formerly correct frames are [158,166) and [167,168):8have T but not G;1has no T but ankle range>0.10. Their ankle range0.062748–0.128088, hip speed0.099223–0.119807, shoulder speed0.128297–0.154285. This regression is threshold conjunction/quiet-policy behavior, not rebuilding.
- SE05_V003:19unknown frames have missing right ankle;8of those also miss left ankle (19unique frames, not27). All19already lack inherited knee support.54unknowns have a rebuilding landmark window, of which51also lack inherited support. Only3frames [194,197),13.088267–13.296300s, have valid inherited standing geometry but are additionally blocked by landmark rebuilding; spans0.608/0.672033/0.752033s, starting at frame185.

The3frame block is required by the implemented continuous locomotion window. It is not logically necessary to compute static standing geometry, so the design does couple the activity output to temporal locomotion evidence. No extra accidental reset is demonstrated by these records. Relaxing that coupling would be a new rule change and was not done. Sitting/lying are explicitly exempt; inherited missing support can still block them. Source causation (jitter, posture adjustment, perspective or occlusion) cannot be proven from saved coordinates alone.

## Outputs and development interpretation

`outputs/v003_abstention_review_20261015_v001/EXPERIMENT_NOTE.md` summarizes reason-duration totals. `UNKNOWN_INTERVALS.md` contains every contiguous half-open unknown interval and original PTS time; `abstention_rows.jsonl` preserves per-frame evidence. Mutually exclusive active reasons and overlapping diagnostics are reported separately: do not add overlapping counts.
Two full-clip side-by-side videos only: SE05_V002 and SE06_V001, source pixels repeated left/right with saved labels, reasons and window readiness. They contain no new inference. Both are video-only copies; originals remain unchanged. Source integer PTS/time_base/frame duration are verified by decoding each output.

Previously measured development results: false moving178->1 on SE05_V002 but unknown25->106; SE06 moving38/58->48/58, standing49/58->40/58. SE05_V003 unknown85->139:61new unknowns offset by7old unknowns becoming known, not simply54new abstentions. Its labels remain unconfirmed, so this is coverage diagnosis, not accuracy. Reduced false positives do not establish overall superiority or generalization. No independent test, annotation changes, new recording request, tuning or deployment. M2 remains IN_PROGRESS.
