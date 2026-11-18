# Acquisition-v002 experimental Pi live integration

Explicit --classifier acquisition-v002 requires --async-ui --pose-backend ncnn --imgsz 320. Strict remains default; low-rate v001 is unchanged, no fallback. Existing local NCNN320 and .venv-ncnn required. No training, new dependencies, model conversion/download or new pose inference on saved datasets.

## Frozen semantics
New loader pins the five classifier bundle hashes and original acquisition feature implementation hash. It calls the same weights_only CPU float64 model loader, vectorizer, preprocessing and AcquisitionFeatures used offline; no fitting or thread setters. Shared CONFIG is immutable: causal (T-3,T], actual acquisition endpoint T, opportunity stride .25s (skip missed slots, no historical backdating); >=3 observations, span>=1.5s, history3s, max gap1.5s, per-channel TTL1.25s, coverage>=.5 and existing mean/std rules. Missing current values cannot refresh old channel support. UI uses the earliest original deadline among valid used channels; no TTL renewal at completion/publication/receipt. Softmax scores are uncalibrated.

Association remains independent: default .5s, explicit experimental option1.5s in the command below. IoU/margin and ambiguity/expiration/no-auto-reseed unchanged. Missing target, track break, reset/reseed clear feature history. Main-thread UI retains only the corresponding processed image and poses; it shows image age. One inference worker takes newest capture, bounded slots; target epoch discards stale work. Explicit reset clears UI immediately. This does not prove identity retention or Pi availability.

## Observability
Bounded diagnostics retain actual acquisition, predict start/end, classifier start/end, completion, publication, receipt and UI assessment clocks. Decision records contain raw vectors, missing flags, channels and per-channel support/deadlines/reasons plus model/config hashes. result_freshness_at_completion counts fresh/expired/unavailable snapshots (including seed-control snapshots, before epoch/overwrite drops); result_freshness_at_first_ui_assessment counts received snapshots once. decision_counts is historical availability, not display eligibility. async_ui fresh_label_submitted_duration_seconds is clipped at original deadline/reset, not proof of human observation. UI refreshes recheck deadlines; latest unavailable never falls back to an older available decision. Tracking, queue replacement, epoch discard, capture accounting and shutdown remain logged. Publication/receipt details are measurable for received results; overwritten results lack a UI timestamp. Check log_omitted_records before treating truncated logs as full coverage.

## Windows PowerShell: transfer only
```powershell
scp -4 -O "D:\patient-monitoring-transfer\pi-acquisition-v002-20261103\pi-acquisition-v002-update-20261103.tar.gz" "D:\patient-monitoring-transfer\pi-acquisition-v002-20261103\pi-acquisition-v002-update-20261103.tar.gz.sha256" truong@patient-pi4.local:~/
```

## Pi Terminal: one line at a time
Existing project and prior live/scheduling updates required. Stop any running demo first. Use a fresh staging directory; mkdir refuses an existing one. Updater verifies the exact allowlist and every incoming hash, backs up all replaced files with verified hashes before replacing, then writes backups/pi-acquisition-v002-*/update_receipt.json. On an incomplete receipt inspect the error, do not blindly retry. No videos, YOLO weights, .venv or old classifier bundles are included.
```bash
cd ~
sha256sum -c pi-acquisition-v002-update-20261103.tar.gz.sha256
mkdir pi-acquisition-v002-stage-20261103
tar -xzf pi-acquisition-v002-update-20261103.tar.gz -C pi-acquisition-v002-stage-20261103
cd ~/pi-acquisition-v002-stage-20261103
sha256sum -c UPDATE_SHA256SUMS
python3 tools/pi_acquisition_update.py apply --staging . --project ~/patient-monitoring-pi4
cd ~/patient-monitoring-pi4
.venv-ncnn/bin/python -m pip check
PYTHONPATH=src .venv-ncnn/bin/python -c "from patient_monitoring.live_acquisition import load_acquisition; m,p,s,c=load_acquisition('outputs/acquisition_action_development_20261103_v001'); print('Loaded',s['version'],s['dimensions'],c)"
PYTHONPATH=src:tests .venv-ncnn/bin/python -m unittest test_live_acquisition test_live_async_ui -q
```

## 90-second UI on Pi desktop
Run in the Pi desktop terminal; ordinary SSH without GUI access cannot show the UI. Output directory must be new. 90s is loop budget after initialization, cleanup adds time; external timeout124 is not a successful complete run.
```bash
PYTHONPATH=src timeout --signal=INT --kill-after=10s 180s .venv-ncnn/bin/python -m patient_monitoring.live_demo --async-ui --classifier acquisition-v002 --camera 0 --backend v4l2 --pose-backend ncnn --imgsz 320 --experimental-association-gap 1.5 --weights models/ncnn_trial_320/yolo11n_pose_320_ncnn_model --model-dir outputs/acquisition_action_development_20261103_v001 --output-dir outputs/pi_live_acquisition_v002_90s_001 --duration 90
```
0-9 explicitly selects the detection on the displayed processed frame; R resets target/history, Q/Esc stops. Startup or expired support must show UNAVAILABLE. Return summary.json and diagnostics.jsonl for timing/availability analysis. Known supply/throttling limitation remains; no performance/accuracy claim, no Pi execution during packaging.

## Local verification (Windows, not Pi)
56 focused tests PASS in .venv, including new adapter parity/version/expiry/missing/reset tests, async epoch/queue/camera-failure/shutdown, strict regression and updater backup/tamper checks. 27 tests PASS in .venv-ncnn including actual frozen model load/forward; both pip checks PASS. 32 protected old hashes and all completed training output hashes unchanged. No retrain, YOLO/camera or Pi run. Evidence: outputs/acquisition_live_integration_20261103_v001/verification.json.
