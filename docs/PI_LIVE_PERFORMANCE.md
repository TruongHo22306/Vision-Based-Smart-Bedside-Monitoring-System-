# Pi live performance diagnostic increment ? 2026-10-27

Prerequisite: previous pi-live-update-20261027 applied with frozen classifier. This increment adds no model/dependency. No Pi execution by Codex. Original live code snapshot remains docs/PI_LIVE_FREEZE.json; this increment has separate checksums. Default Windows/PT640 behavior and model parameters unchanged; optional explicit --imgsz320 changes pose resolution only for this diagnostic.

## User-reported evidence, not locally measured
V4L2 camera0 640x480/~15capture frames/s. PT640 headless complete/exit0:17processed/892captured, mean reported predict3.469095s, processing0.277246/s; all awaiting explicit seed. Classifier aggregation was not tested. Separate Python process Torch2.9.1+cpu reports intraop4/interop4; not proof of counts in live. Postrun65.7C and0x50000: historical undervoltage and throttling bits; current flags clear at that sample only, not evidence about intervals between samples. No causal attribution of slowdown yet.

## Initialization trace and thread policy
live_demo.run -> load_runtime -> import supervised_action (definitions only) -> load_bundle -> CPU float64 Linear + weights_only state_dict + eval. train() is NOT called. supervised_action.py:216 thread setters exist inside train(), not load_bundle. A focused real classifier-load test patches both thread setters to fail; loading succeeds. No added thread setter, env thread override or optimizer.

Four actual current-process Torch intraop/interop/version/monotonic snapshots in summary.json: before_classifier_load, after_classifier_load, after_yolo_initialization, after_first_predict. Stages not reached remain absent on failure. First predict lazily initializes Ultralytics predictor, so import/YOLO constructor is not the last potential library change. Installed Ultralytics select_device contains its own CPU thread setter; observations, not separate-process assumptions, will show live values. No assertion that it caused the reported Pi slowdown.

## Timings and telemetry
--imgsz defaults640, accepts positive integer multiples of32; PT320 is explicit. PREDICT_CONFIG copied per run; CPU/batch1/rectFalse/halfFalse/conf.25/IoU.7 unchanged. --backend means camera capture backend, NOT PT/NCNN. Input source remains PIL RGB preserving aspect ratio; OpenCV capture geometry unchanged.

first_predict_seconds is first successful model.predict wall time, including its lazy setup/warmup. subsequent_predict_count/sum/mean/min/max use every later successful call, exclude first. Timer includes Ultralytics preprocessing/postprocessing, excludes PIL conversion, saved-joint conversion, telemetry, classifier and UI. Existing pose_seconds/latest_mean_pose_seconds retain their older broader conversion+predict+quality timing; do not compare them as identical definitions. Per-frame predict_phase/predict_seconds and Ultralytics speed milliseconds are logged separately. No artificial warmup frames or dropped first output.

Processed timestamps remain host acquisition read-return monotonic_ns. Summary counts/mean/max gaps and gaps>0.5s, per-frame exceeds_continuity_limit. Equality0.5s allowed. Target last-accepted gap>0.5s ends association; feature/window barriers unchanged. Adjacent gaps<=0.5s do not guarantee a valid track. Classifier windows still1s/stride.25 with original coverage/imputation. PT320 cannot be assumed fast enough or accurate enough; headless remains unseeded/UNAVAILABLE and cannot validate classifier availability.

Telemetry reuses benchmark command/parsing semantics in stdlib-only live_diagnostics: vcgencmd measure_temp/get_throttled sequentially, timeout1s each. Samples before capture, after first predict, at most once per5s thereafter after predict, after capture cleanup. Outside predict timer, but included in live throughput and may contribute gaps. Log<=10MB, omitted count explicit, only latest telemetry plus count in summary; no growing memory buffer. Unavailable/partial/errors are explicit; current/historical flags separate. No claim about unsampled periods.

YOLO_CONFIG_DIR and MPLCONFIGDIR are absolute directories created and write-tested before load_runtime/Ultralytics imports. CLI starts a fresh process; already imported libraries in an embedding application may have cached old settings. Existing warning dedup remains narrow. A zero elapsed monotonic clock tick now reports null summary throughput rather than division by zero.

## Windows PowerShell
```powershell
scp -4 -O D:\patient-monitoring-transfer\pi-live-performance-20261027\pi-live-performance-update-20261027.tar.gz D:\patient-monitoring-transfer\pi-live-performance-20261027\pi-live-performance-update-20261027.tar.gz.sha256 truong@patient-pi4.local:~/
```

## Pi terminal: one command per step, stop on errors
```bash
cd ~
sha256sum -c pi-live-performance-update-20261027.tar.gz.sha256
mkdir pi-live-performance-stage-20261027
tar -xzf pi-live-performance-update-20261027.tar.gz -C pi-live-performance-stage-20261027
cd ~/pi-live-performance-stage-20261027
sha256sum -c UPDATE_SHA256SUMS
python3 tools/pi_live_performance_update.py apply --staging . --project ~/patient-monitoring-pi4
cd ~/patient-monitoring-pi4
.venv/bin/python -m pip check
.venv/bin/python -m patient_monitoring.live_demo --help
timeout --signal=INT --kill-after=10s 180s .venv/bin/python -m patient_monitoring.live_demo --camera 0 --backend v4l2 --imgsz 320 --weights models/yolo11n-pose.pt --model-dir outputs/supervised_action_development_v001_20261021 --output-dir outputs/pi_live_pt320_diagnostic_001 --duration 60 --no-display
```

Updater verifies exact file hashes, backs up every replaced file before replacement and writes receipt under backups/pi-live-performance-*. Five-file increment; no source video, models, .venv or outputs. Choose new staging/output suffix if already present. 60s is capture-loop budget after model loading; current predict/telemetry/shutdown can overrun it. External180s ceiling/10s kill grace bounds hangs; timeout exit124 is not successful completion. No raw video. Review summary.json and diagnostics.jsonl, especially snapshots, first/subsequent timing, gaps and telemetry before changing anything.

## NCNN is separate work
pose_benchmark supports local *_ncnn_model metadata/hash verification, explicit predictor setup and net.opt thread/precision snapshots. live_demo currently enforces the original PT file hash and YOLO PT loading. Accepting an NCNN directory needs explicit live loader/config validation, CPU/runtime option reporting and parity checks. Torch thread counts would not describe NCNN execution. This update does not integrate or switch to NCNN.

Local checks:6 new diagnostic tests +9 live +4 camera +5 Pi preparation regression tests; pip check. Synthetic camera/YOLO in diagnostic integration test; real frozen classifier load, no camera or pose inference and no Pi measurement. Previous failed test fixtures retained in conversation only; final tests must pass. M2 remains IN_PROGRESS.
