# Low-rate Pi display audit ? 2026-10-31

Source (read only): D:/patient-monitoring-transfer/pi-live-results/pi_live_low_rate_review_20261030_170810/. See pi_log_replay/audit.json for exact hashes and reconstructed timing_replay.json for all 36 candidate frames.

## Findings
Complete log: 122 processed frames; 259 decisions = 94 available, 165 unavailable; omitted records 0. The old frame log has 10 labels (7 standing, 3 moving), 112 null. These are pre-telemetry/pre-final-expiry states, not proof that 10 labels were visible on screen.

94 historical available decisions belong to 37 processed-frame batches. Only 36 batches end in an available decision. Frame 20 ends at support_end=9.5 s with insufficient_observed_span, after three earlier available decisions (8.75, 9, 9.25 s). Rejecting that older label is correct. All decision grids are increasing and the last end is within [acquisition-0.25s, acquisition]. No wrong latest-decision selection or grid lag was found.

The newest numeric sample in all 36 candidate windows is the preceding processed frame: the current sample closes the duration support and is excluded numerically by the frozen offline [start,end) aggregation. This is the trained temporal convention, not an off-by-one that can be changed locally. Acquisition timestamps are unchanged. Wall-time support age = (current acquisition - last supporting acquisition) + current frame processing age. It is NOT merely decision timestamp age or latest_age measured at the historical window end.

|Old frame state|Count|Support age min / median / max (s)|
|---|---:|---|
|label present|10|0.860705 / 1.105298 / 1.218271|
|display_support_expired|26|1.283201 / 1.388567 / 1.855115|

All 36 reconstructed expiry classifications match; 0 mismatches. Example processed frame 12: scheduled end 3s, nominal latest_age .782308s, previous-support lag at acquisition .839549s plus processing age .824179s = 1.663728s > 1.25s. Correct expiry despite historical availability. For expired frames, support lag at capture .469?.973s, median .720394s; classifier aggregation .021850?.089978s, median .064386s. Mean subsequent pose predict .542804s; this is not total latency. Telemetry current undervoltage/throttling present in 16/18 samples; 2 only historical. These samples do not quantify its causal performance impact.

112 null frame reasons: awaiting seed29; startup8; expired26; insufficient span1; no detections3; unresolved-target continuity1; unresolved/invalid timestamp4; gap timeout2; association ended36; low IoU2. No accuracy claims or ground truth in this analysis.

## Minimal instrumentation corrections
- Legacy computed_host_ns was assigned BEFORE classifier forward. Now compute_started_host_ns is separate and computed_host_ns records completion; decision/acquisition timestamps stay unchanged.
- Legacy frame label was recorded BEFORE telemetry and a second expiry check. Preserve that legacy record; add one display_assessment per processed frame after telemetry/decision logging and before UI submission, with selected decision, completion, support timestamps and age.
- Add display_summary: processed assessments, fresh-label frames, UI-label submission frames, non-display reason counts. Headless readiness is separate from UI submission. Counts persist beyond bounded-log truncation. No claim of physical screen refresh or dwell time. Missing/reset/expired states never restore a historical label.
- Classifier selection, frozen 35 inputs, model/config, TTL1.25s, gap1.5s, and strict behavior unchanged.

## Checks and limits
29 focused tests PASS: 4 display timing/reset/headless tests, 5 offline/live parity tests, 9 strict/live regressions, 5 association tests, 6 performance tests. Saved-log audit checks all 122 frame / 259 decision counts, ordering, support-source mapping and 36 expiry decisions. Logs contain no raw poses: this replays timing and display eligibility, not original feature extraction/classification. Exact legacy completion and actual render times cannot be reconstructed; early frame time gives a lower bound, so all 26 expired cases remain expired at rendering. Original inputs are read-only and their hashes preserved.

## Decision / next step
No model-selection or scheduling fix is justified. Keep rejecting expired labels. This synchronous UI also replaces the label with processing_fresh_frame during every next predict, reducing visible dwell time even for fresh labels. Do not increase TTL/gaps or substitute completion time for capture time. A concrete next software experiment would separate UI refresh from synchronous pose work with one bounded result slot and the same wall-time expiry check, then measure label-visible dwell time; this does not guarantee more available classifications. To make more labels eligible, the existing pipeline must finish within 1.25s minus the support-to-current-capture lag. No such threading/performance change is made here. No Pi rerun, YOLO, training, model/config or dependency change.
