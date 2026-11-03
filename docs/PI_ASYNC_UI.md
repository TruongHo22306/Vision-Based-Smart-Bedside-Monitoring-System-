# Pi low-rate asynchronous UI (2026-10-31)

Opt-in `--async-ui --classifier low-rate`. Strict remains the default synchronous path; combining strict with --async-ui is rejected. Existing model/schema/preprocessing, 3s window, .25s schedule, 1.25s support age, 1.5s feature gap and separately configured association gap are unchanged. No extra model copy, imports/install/downloads, or training.

## Ownership and timing
Models load once using the existing initialization. One worker thread exclusively calls pose predict, target/feature/classifier code and telemetry. Main thread owns OpenCV UI and polls the existing bounded capture process. Latest-only slots: one running inference, one pending frame, one result, one pending control; overwritten counts and discarded epochs are explicit. Log remains capped at 10MB and is thread-safe. UI loop waits .02s plus rendering/event time; this is not a real-time guarantee. Python/GIL, CPU contention and OS scheduling can delay refresh.

UI deliberately shows a PROCESSED FRAME, never overlays old poses on a new camera image. Image age is host monotonic now minus that image's original acquisition timestamp; not sensor-to-screen latency. Before a current-epoch result exists, raw camera frames are displayed without boxes, labels or seedable detections.

Result envelope includes source frame ID/timestamp, completion timestamp, target_epoch, matching image/pose and current latest-decision state. The last grid decision wins, including unavailable; no search backward for an older available decision. Every UI update checks original support age using monotonic time. Receiving/redisplaying a result never restarts TTL. Labels can remain only until the unchanged support deadline while the next inference runs. No frame interpolation or feature/window changes.

Digits0-9 seed the detection on the exact processed image currently displayed. R resets. Both increment UI target_epoch immediately, clear displayed target/history eligibility and clear pending jobs/results. Worker serially clears engine history/applies the exact saved detection without rerunning pose. Results finishing under an old epoch are discarded. A rapid later control supersedes a pending control; no automatic seed/re-entry. Internal engine epoch and external target_epoch are logged separately. Gap limits still apply to the seed's ORIGINAL timestamp; seeding an old picture does not refresh its age.

Q/Esc/window close or --duration ends UI, closes its windows, stops accepting work and joins the one in-flight inference (10s limit), then the existing capture cleanup runs. A native model call cannot safely be killed as a Python thread: if it does not return, summary is incomplete, async_worker_stopped=false; use the external timeout below. No claim of clean shutdown in that case. Worker and camera failures propagate to incomplete status.

## Diagnostics
- decision_counts: model/window available/unavailable, including work completed under a subsequently discarded epoch. Historical availability is NOT live label validity.
- display_summary: first assessment of each received current-epoch result; separate from repeated refreshes.
- async_ui: label_submission_updates, fresh_label_submitted_duration_seconds, support_expired_results, non_display_reasons, mean/max processed image age, max UI tick gap. Duration integrates monotonic time between submitted label updates, clipped at original support deadline, next update/reset or UI close. It does NOT prove screen refresh, human perception or actual visible dwell.
- frame/decision/ui_update records: frame IDs, source/decision/completion/support timestamps, target epoch, image/support age, reasons, predict and aggregation timing. First/subsequent model.predict statistics remain separate. Stale-epoch and overwritten jobs/results have separate counters. No raw video recording.
- Legacy latest_sample_count/mean rings are synchronous-only; use async_ui image-age statistics and per-frame timings for this path.

## Actual local checks / limitations
40 focused tests PASS (11 async + 29 display/parity/strict/association/performance); 20 async/display/parity tests PASS in Windows .venv-ncnn. Both pip check PASS. Full loop with fake camera/predict verifies main-thread UI, exactly one inference thread, image/pose alignment and explicit seed frame. These are not webcam or Pi tests. No real pose inference was executed. Protected model/config/feature/window files unchanged.

The new UI does not make support newer and does not resolve the audited 26 expired labels by design. Availability may remain unchanged or worsen under UI CPU load. It may improve UI responsiveness and allow a still-valid label to remain submitted during inference; measure on Pi rather than asserting improvement. Known undervoltage remains a measurement limitation.

## Transfer / apply (one line per step)
Requires prior NCNN/low-rate live update and existing classifier artifacts. Package contains code/tests/docs only, no weights or environment. Do not overwrite an existing staging/output directory; choose a new suffix if needed.

Windows PowerShell:
```powershell
scp -4 -O D:\patient-monitoring-transfer\pi-async-ui-20261031\pi-async-ui-update-20261031.tar.gz D:\patient-monitoring-transfer\pi-async-ui-20261031\pi-async-ui-update-20261031.tar.gz.sha256 truong@patient-pi4.local:~/
```
Pi terminal:
```bash
cd ~
sha256sum -c pi-async-ui-update-20261031.tar.gz.sha256
mkdir pi-async-ui-stage-20261031
tar -xzf pi-async-ui-update-20261031.tar.gz -C pi-async-ui-stage-20261031
cd ~/pi-async-ui-stage-20261031
sha256sum -c UPDATE_SHA256SUMS
python3 tools/pi_async_ui_update.py apply --staging . --project ~/patient-monitoring-pi4
cd ~/patient-monitoring-pi4
.venv-ncnn/bin/python -m pip check
PYTHONPATH=src .venv-ncnn/bin/python -m unittest discover -s tests -p test_live_async_ui.py -v
PYTHONPATH=src .venv-ncnn/bin/python -m unittest discover -s tests -p test_live_low_rate_classifier.py -v
```
Updater verifies every payload before making changes, backs up all existing destinations before replacement, verifies backups, then replaces files individually. Receipt in backups/pi-async-ui-*/update_receipt.json; incomplete receipts require inspection, not a blind retry.

On the Pi desktop terminal (interactive UI requires desktop; plain SSH is insufficient):
```bash
PYTHONPATH=src timeout --signal=INT --kill-after=10s 180s .venv-ncnn/bin/python -m patient_monitoring.live_demo --async-ui --classifier low-rate --camera 0 --backend v4l2 --pose-backend ncnn --imgsz 320 --experimental-association-gap 1.5 --weights models/ncnn_trial_320/yolo11n_pose_320_ncnn_model --model-dir outputs/low_rate_development_v001_20261030 --output-dir outputs/pi_live_async_ui_90s_001 --duration 90
```
90s main loop after model/capture initialization; cleanup can add time. External timeout exit124 is not success. Select the visible subject explicitly with its digit; R resets; Q/Esc stops. Compare actual summary/diagnostics before claiming availability or UI improvements. No Pi execution has been performed here.

## 2026-11-02 source audit correction (not in the 20261031 archive)
See ASYNC_UI_AUDIT_20261102.md. Async worker now consumes capture directly when ready, UI no longer polls capture/feeds a second frame slot. Startup displays unavailable placeholder until a processed image arrives. Source-fetch/publication/UI-stage timing added. Raw-frame drops at reset/stop are accounted separately from discarded results; old unprocessed_on_stop128 contained126 known frame drops. Existing archive remains unchanged; do not assume it contains this correction. No new Pi run or improvement claim.
