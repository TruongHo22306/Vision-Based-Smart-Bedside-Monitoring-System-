# Pi async UI audit - 2026-11-02

Read-only source: D:/patient-monitoring-transfer/pi-live-results/pi_live_async_ui_90s_001. New audit: outputs/async_ui_audit_20261102_v001/audit.json and frame_age_decomposition.json. Input hashes recorded. Deployed async source hash matches the pre-edit local module exactly.

## Evidence and decomposition
Complete run, no omitted logs:147 processed frames,143 first-result assessments (includes control/seed results; not143 unique pose calls),319 UI updates;161 historical available decisions;11 labeled submissions/.765561425s clipped-valid submission duration.69 initial result assessments have available model support:11 fresh/58expired. All69 source-age sums and TTL outcomes reproduced with0 mismatches.66 distinct results eventually expire across92 repeated UI updates; do not mix these with58 expired on first assessment. UI submission is not physical display/accuracy.

Mean seconds across the same69 result assessments (additive; medians are not additive):
|Component|Mean|Median|
|---|---:|---:|
|Previous numeric support -> current acquisition|.570014|.538940|
|Acquisition -> worker process start|.176175|.143808|
|model.predict|.433552|.438770|
|Other pose conversion/serialization|.006838|.005963|
|Classifier feature/window aggregation + forwards|.097980|.100368|
|Remaining completion overhead|.000337|.000190|
|Completion -> UI assessment|.341031|.431585|
|Total support age|1.625926|1.679071|

Formula: support_age = sample_gap + pre_inference + pose + classifier + residual + completion_to_UI. The previous sample closes [start,end) numeric support exactly as trained. Sample gap is the largest single mean component; sample gap+predict alone average1.003566s, leaving only.246434s of the1.25s budget before other costs. Classifier is not the dominant component. Legacy pre_inference combines capture IPC, main-thread polling, pending-slot residence and scheduling: separate queue waiting cannot be measured exactly. Completion_to_UI includes result logging/telemetry/publication and scheduling/render-loop wait, NOT pure UI rendering time. Do not invent more precise subdivisions.

## Worker freshness and initial prediction
Frame0 timestamp265784894209ns, first predict13.931815s; next processed frame106 timestamp279705158024ns, start age.071626s. It did NOT wait13.93s. Frame107 subsequently starts.324513s after acquisition, frame114 .406404s. Legacy worker takes the newest UI-submitted slot, not necessarily newest frame still in capture queue: the UI can be delayed in rendering/native calls/IPC. Logs do not record candidate queue IDs at each start, so cannot prove which unseen newer frame was then available.

Minimal scheduling correction: worker is now the sole consumer of capture.newest(), called only when ready for work; UI never reads capture or feeds the inference slot. It retains only the processed image/result and initially shows an unavailable placeholder. Source retrieval is outside the control lock; reset during retrieval discards that frame and applies control first. One worker/model, unchanged source timestamps/feature semantics. New source-fetch/process/predict/classifier/publication/receipt/render/event/log timings expose the remaining delay. Newest means newest delivered by the existing capture queue, not a guarantee of newest sensor exposure or zero IPC latency. Existing latest-slot API remains for regression tests; no model setting change.

## UI tick gap
The maximum1.135731392s is between265797197149 and266932928541ns: first camera image display while first predict is running, before any seed or label. It is not the13.93s prediction duration. Median adjacent tick gap=.145323281s, despite requested20ms sleep. Legacy loop also performs a blocking capture.newest(timeout=.05) on UI thread, plus paint/imshow/waitKey, logging and thread scheduling. Lazy GUI initialization and native/GIL contention are plausible contributors, not measured causes. No per-stage legacy timestamps exist, so exact attribution is impossible from these files. The fix removes capture IPC from UI; it does NOT prove removal of GIL or GUI stalls. Stage logs now distinguish render/events/logging and publication-to-receipt. No latency/availability improvement has been measured.

## Frame accounting
Old equation676captured -147processed -334capture-overwritten -66capture-dropped -1drained =128. This omitted125 inference-slot replacements and1 frame cleared at reset. Correct residual from OLD logs is2, not128 queued frames. At most one capture-queue and one inference pending frame exist. Legacy shutdown did not count pending deletion explicitly, so2 cannot be split conclusively; they are a shutdown residual, not evidence of growing backlog. Dropped/overwritten RESULT objects already refer to processed frames and must not be subtracted again.

New accounting records pre-processing reset/stop/fetch-epoch drops separately, preserves legacy_unprocessed_residual, computes capture_accounting_residual, and leaves any negative discrepancy visible. Strict accounting unchanged. No old summary is rewritten.

## Verification / next step
42 focused tests PASS, including2 new regressions: source read occurs only when worker ready (simulated long first predict gets latest106), and128->2 accounting without double-counting results.22 async/display/parity tests PASS under Windows .venv-ncnn; both pip check PASS. Existing reset, queue bounds, fake full UI frame/seed alignment, camera failure/cleanup, TTL, frozen parity and strict regression retained. Model/config/feature/window/PROJECT_PLAN/dependencies unchanged; no Pi/YOLO/retraining.

The architecture cannot remove the previous-sample support lag without changing trained semantics, which is prohibited here. Do not increase TTL, extend labels or run the same90s trial again as if it were a new experiment. Next justified diagnostic: a short controlled profiling of native predict versus main-thread GUI/event cadence, using the new publication/render milestones (and identical backend settings), to determine whether a process boundary is warranted. Moving inference to a process is only a hypothesis until scheduling/GIL costs are isolated; it will not erase original support age. This task does not implement that further architecture or request new recordings.
