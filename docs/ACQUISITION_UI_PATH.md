# Acquisition-v002 Pi UI path audit 2026-11-04

Read-only inputs: D:/patient-monitoring-transfer/pi-acquisition-results/{summary.json,diagnostics.jsonl}. Hashes in analysis.json. Run complete,141 processed; log_omitted_records0. No inference/Pi run during audit.

Matched FIRST UI receipt by (target_epoch,frame_id), not unrelated summary populations:101 support-bearing results =67 fresh at UI +23 fresh at completion but expired at UI +11 expired already at completion. Summary completion91fresh includes one not in this received support population; do not subtract aggregate counts to infer matching. 102 historical available decisions are not102 live eligible labels. Full matched timestamps/deadlines: matched_results.json.

For23 lost-in-transit results: completion->publication mean8.020ms; publication->receipt mean531.269ms,median563.531ms,p95 685.712ms,max700.789ms; receipt->assessment mean0.019ms. Across101 support results publication->receipt mean212.600ms. Worker publication overhead is not the main delay in this sample.

Intersect the23 publication-to-receipt intervals with logged UI phases: render0.132164s; waitKey envelope3.800697s; after events to next loop8.285511s. These are time overlaps, not proof that a named OpenCV call consumed CPU. All325 UI updates: render mean46.857ms,p95 453.125ms; waitKey envelope mean144.007ms,p95 620.708ms; logging mean1.122ms,p95 4.028ms,max9.015ms. The tail includes getWindowProperty, key handling, logging, unconditional20ms sleep and OS/GIL contention; old logs cannot separate them. Do not attribute its ~360ms mean overlap to20ms polling alone. No measured mutex-contention or scheduler trace exists.

Code evidence: each loop sleeps20ms even if a result was published during render/waitKey, and publication did not notify UI. Narrow correction: Condition notification on publication and interruptible bounded idle wait with result/error/stop predicate; pending results skip wait. Timeout remains<=20ms and clips to current label deadline. No model, feature, preprocessing, TTL, gap, target or strict synchronous path change. Added tail timing for window-property and actual idle wait; future traces can separate these from scheduler overrun. Main-thread render/waitKey remain; no claim this fixes the full531ms delay.

Tracking is separate: frames285/290 rejected best IoU .094612/0 at reference ages.672240/1.342307s (minimum remains.1); frame293 times out at1.746517s >association1.5s.11 subsequent association_ended frames require explicit reseed. These unresolved barriers correctly clear features. Expired support in101 matched results must not be counted as track loss. Recorded target counts:17 awaiting seed,110 matched,2 low_iou,1 gap_timeout,11 ended.

37 focused tests PASS: new publication wake/pending-result no-sleep regression plus queue/old-epoch/reset/expiry/camera failure/shutdown, acquisition parity, low-rate and strict tests. Model/config/deadline unchanged. Pi has not run this patch. Check results on one bounded60s run, then compare same-receipt populations using deadline at completion and UI, lost-in-transit fraction, publication->receipt median/p95, render/waitKey/property/idle timings, fresh-label duration per seeded observation time and tracking/shutdown separately. Compare under similar actions/conditions; counts alone and different track durations are not controlled improvement evidence. If property/waitKey dominates, this patch may have little effect; use new measurements before another architecture change.

## Transfer and apply (one line per step)
Windows PowerShell:
```powershell
scp -4 -O "D:\patient-monitoring-transfer\pi-acquisition-ui-20261104\pi-acquisition-ui-update-20261104.tar.gz" "D:\patient-monitoring-transfer\pi-acquisition-ui-20261104\pi-acquisition-ui-update-20261104.tar.gz.sha256" truong@patient-pi4.local:~/
```
Pi Terminal, stop the previous demo first; staging/output must be new:
```bash
cd ~
sha256sum -c pi-acquisition-ui-update-20261104.tar.gz.sha256
mkdir pi-acquisition-ui-stage-20261104
tar -xzf pi-acquisition-ui-update-20261104.tar.gz -C pi-acquisition-ui-stage-20261104
cd ~/pi-acquisition-ui-stage-20261104
sha256sum -c UPDATE_SHA256SUMS
python3 tools/pi_acquisition_ui_update.py apply --staging . --project ~/patient-monitoring-pi4
cd ~/patient-monitoring-pi4
PYTHONPATH=src:tests .venv-ncnn/bin/python -m unittest test_live_async_ui test_live_acquisition -q
```
Updater verifies incoming files, backs up every replaced file before replacement and writes backups/pi-acquisition-ui-*/update_receipt.json. Inspect any incomplete receipt rather than blindly retrying. Prior acquisition-v002 installation required; no model rebundled.

Run on Pi desktop (not ordinary SSH without GUI):
```bash
PYTHONPATH=src timeout --signal=INT --kill-after=10s 150s .venv-ncnn/bin/python -m patient_monitoring.live_demo --async-ui --classifier acquisition-v002 --camera 0 --backend v4l2 --pose-backend ncnn --imgsz 320 --experimental-association-gap 1.5 --weights models/ncnn_trial_320/yolo11n_pose_320_ncnn_model --model-dir outputs/acquisition_action_development_20261103_v001 --output-dir outputs/pi_acquisition_ui_60s_001 --duration 60
```
Select detection with0-9; R explicit reset; Q/Esc stop. Do not automatically reseed. Same model/pose/TTL/gaps as preceding run; record user actions and seed/reset times. Compare received support-bearing results by epoch/frame, completion-fresh -> UI-expired fraction (baseline23/90), publication-to-receipt p50/p95 (baseline all101:51.350/599.292ms; lost23:563.531/685.712ms). Report startup/track loss separately. Improvement needs lower delay/lost fraction without stale labels, unexpected identity changes or shutdown regression; short runs with different tracking/action distributions cannot prove causality. Return original summary.json and diagnostics.jsonl, including new ui_tail_timing. No Pi execution claimed.
