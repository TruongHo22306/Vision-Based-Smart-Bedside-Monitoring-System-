# Startup/backend update

2026-10-26: camera0 DSHOW probe and headless live capture+pose smoke succeeded;
MSMF camera0 blocked inside VideoCapture and timed out. Historical failure notes below
are retained. Explicit backend option is now available; no automatic fallback.
Next command (NEW output directory):
```powershell
.venv\Scripts\python.exe -m patient_monitoring.live_demo --camera 0 --backend dshow --weights models/yolo11n-pose.pt --model-dir outputs/supervised_action_development_v001_20261021 --output-dir outputs/live_dshow_manual_001
```
Lightweight sequential probe:
```powershell
.venv\Scripts\python.exe -m patient_monitoring.camera_probe --camera 0 --output-dir outputs/camera0_probe_manual_001
```
See outputs/camera_startup_diagnosis_20261026/REPORT.md for stage timings and limits.
New startup_stages distinguish worker/import/open/first-read timeout10s per stage.
No UI/seed/activity classification test was performed in the headless smoke.

---

# Live webcam development demo

## Start / initialize / stop
From D:\patient-monitoring-pi4, with a NEW output directory:
```powershell
.venv\Scripts\python.exe -m patient_monitoring.live_demo --camera 1 --weights models/yolo11n-pose.pt --model-dir outputs/supervised_action_development_v001_20261021 --output-dir outputs/live_webcam_manual_001
```
Camera index1 is only a requested index, NOT verified USB identity. Use the actual USB index on your machine; the program does not guess from filenames or silently fall back to another camera. Optional `--duration 30` stops after30seconds from capture startup (one in-flight inference may finish first). Existing outputs are refused. No video/images are recorded by default.

In the OpenCV window, inspect the visible boxes labeled D0,D1,... (frame-local indices).
Press the corresponding digit0-9 to explicitly seed THAT displayed image/detection. No person is selected by default. Yellow box/green skeleton is the selected subject. Keys select only indices0-9; do not assume D0 always means the subject. A missing/invalid seed is rejected and requires another explicit selection.
Press r to clear association, selected label and all feature/window history, then choose a new displayed D index. Press q or Escape to stop; closing the window or Ctrl+C also stops. Keep keyboard focus on the OpenCV window. A digit is itself an explicit reseed/reset; there is no silent reacquisition.

Headless smoke (no seeding or activity decisions, never chooses a person automatically):
```powershell
.venv\Scripts\python.exe -m patient_monitoring.live_demo --camera 1 --weights models/yolo11n-pose.pt --model-dir outputs/supervised_action_development_v001_20261021 --output-dir outputs/live_webcam_smoke_NEW --duration 8 --no-display
```

## Reused logic and temporal contract
Existing verified YOLO11n pose weights are required locally by SHA256; no downloads or package installs. PT640 CPU config is unchanged: batch1, rectFalse, halfFalse, conf0.25, IoU0.7. PIL RGB input follows existing extraction convention. quality_people preserves the current flags; skeleton edges require both existing passes_quality_gate endpoints. Boxes/quality are not accuracy evidence.
TargetAssociator uses IoU0.1, margin0.05, max elapsed gap0.5s. Reference advances only on accepted matches. After timeout association remains ended; explicit user seed is required. FeatureExtractor reuses actual processed-frame deltas and the same0.5s continuity guard. Camera frames overwritten before processing are not invented or interpolated. Slow processing beyond0.5s may therefore terminate tracking; no threshold was relaxed for the demo.

Capture timestamps are time.monotonic_ns immediately after read returns in a separate capture process. They are HOST acquisition timestamps, not verified sensor exposure timestamps. Frame indices enumerate processed samples within each explicitly reset epoch; capture sequence numbers are logged separately. Reset changes epoch/origin and removes all prior temporal evidence.

The classifier uses the frozen28-column vectorizer, stored medians/scales and weights_only state load. Frozen manifest is verified before and after a completed run. No fitting, smoothing, confidence threshold or new activity rule. Every epoch schedules trailing1s windows every0.25s from the seed timestamp. Before1s: UNAVAILABLE. Live display support for a processed sample is its acquisition time through the next processed sample acquisition; no last-frame duration is guessed. Decisions are computed only once a later acquisition establishes the window end. Features at/after the window end are excluded, and backward derivatives keep their entire support within the window, matching training aggregation on identical supports.

History is bounded to256processed samples and approximately2s, retaining a boundary sample. Local history indices are rebased ONLY for make_window continuity checks; original processed indices are recorded separately. If support has been evicted, a window is unavailable. Long processing stalls record skipped decision-slot counts instead of building an unlimited backlog. No windows bridge unresolved targets, invalid temporal pairs or reseeds.

The display shows the latest available decision only after its support end. A rejected/unavailable decision or current lost/broken track clears the label. When capture waits longer than the existing0.5s gap bound without another processed frame, UI clears the held label as waiting_for_fresh_processed_frame. During a synchronous in-flight pose call UI may pause until it returns; this is an offline-development-grade demo, not a hard realtime UI. This display behavior is separate from the model's four-class output.

## Bounded capture, diagnostics and cleanup
One queue slot holds the latest capture. Replacing a pending frame increments overwritten_pending; a feeder race that prevents insertion increments dropped_incoming; consumer-side discarded older frames increment drained_older. Any remainder at shutdown is unprocessed_on_stop. No growing frame queue. CAP_PROP_BUFFERSIZE=1 is best effort and may be ignored by the camera backend; hidden driver buffering is not measured. Queue capacity does not prove sensor freshness.
Capture owns VideoCapture and releases it in finally. Stop asks the child to finish, waits2s, then terminates a blocked backend if necessary and reports forced_capture_termination. Main closes queue/UI/logs. A backend forcibly terminated cannot run its Python release-finally; OS process teardown is the fallback, not a claimed graceful device release.

summary.json and diagnostics.jsonl contain frame acquisition times, capture sequence, processed deltas, selection status, decisions, gaps, drops, pose/classifier-aggregation durations and frame age. JSONL is capped at10MB, then omitted-record count grows; no raw frames or whole pose arrays saved. Measurement rings retain latest256 timings. Summary distinguishes capture rate (first/last successful read timestamps) from processing throughput (processed count/capture-start elapsed). Pose time includes its predict preprocessing/postprocessing. Classifier-aggregation time includes windows/vectorization/model calls, not just matrix multiplication. Host read-return-to-pre-display age is NOT sensor-to-screen latency; exposure, backend buffering and actual display refresh are unmeasured. No nominal FPS is substituted.

## Actual checks and limitations
9live/replay tests +7offline-prediction +21feature +11target tests PASS (48total); pip checkPASS. Tests cover explicit seed/timestamp rejection, irregular deltas, offline/live aggregation/vector/logit parity on identical support, future-feature exclusion, gap termination/reset, bounded history/queue/log, failed camera release and normal stop release. Existing frozen artifacts were not changed.
Actual smoke: camera index1, requested8s, no-display. OpenCV reported Camera index out of range;0captured/processed. Summary outputs/live_camera1_smoke_20261026/summary.json is incomplete with camera-open/read error. Capture process exited with no forced termination. No real-camera pose/activity rate, frame-age measurement or interactive UI/seed verification is claimed. The small later UI-wait/close handling change passed live tests but could not be camera-verified. Camera index0/other devices were not probed or substituted.

Single-person IoU can still associate incorrectly; explicit seeding is not an identity guarantee. Current classifier is trained on one participant with occasion confounding and overlapping windows. Softmax scores are uncalibrated. This is a development demonstration, not patient monitoring validation, Pi deployment or fall detection.
