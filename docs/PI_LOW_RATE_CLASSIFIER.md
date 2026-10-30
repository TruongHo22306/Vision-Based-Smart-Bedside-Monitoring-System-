# Explicit experimental low-rate classifier UI ? 2026-10-30

This update integrates the saved Linear35x4 model without refitting, training, pose inference or package changes. Strict stays default: --classifier strict with the original28-input directory. Explicit --classifier low-rate requires the separate low-rate artifact directory; neither loader falls back. Pi execution NOT performed here.

## Shared logic and pinned artifacts
live_low_rate.load_low_rate pins SHA256 of original model_state.pt,preprocessing.json,schema.json,config.json in code. It validates35columns/order,4-class order standing/sitting/lying/moving, finite medians/scales/weights; torch.load(weights_only=True),CPUfloat64. Serialized training config hash9dcea4c1... differs from original preparation hash1cd7e002... only because serialized whitespace differs; deployment pins the actual saved artifact bytes. Original artifacts never edited. No training setters executed by import or load.

LowRateEngine calls low_rate_experiment.recompute and window/apply directly. No alternative window/normalization implementation:3s trailing window,stride.25,>=3observations,span>=1.5s,gap<=1.5s,latest sample age<=1.25s,mean>=2valid observations+coverage.5,std>=3. Values read from pinned config. Nulls/missing flags preserved. Capture/acquisition timestamps monotonic host times, not sensor exposure.

Association is separate: default strict.5s; use explicit --experimental-association-gap1.5 for this diagnostic. IoU/margin and expiry behavior unchanged; no automatic person switch/reseed. Selected rows are recomputed in bounded history, max256frames and window+2gap+stride=6.25s retention; strict remains~2s. Missing/rejected rows/barriers prevent bridging; reset clears every target/temporal state. Real events between processed frames are unknowable: offline saved identity is not equivalent to live identity guarantee.

Decision support timestamps are separate from computed_host_ns and latest_arrived_host_ns. No numerical feature at/after nominal window end is read; a following acquired timestamp closes support, as offline. Compute waits until that timestamp has arrived. Old grid slots outside retained support are counted as skipped, never filled. Every new unavailable decision clears label; lost target/invalid temporal pair clears immediately.

Display expiry uses the timestamp of the newest supporting pose and pinned max_latest_age_seconds, evaluated against actual monotonic clock before rendering and while waiting for capture. Delayed computation/telemetry may therefore produce an available MODEL decision but UNAVAILABLE DISPLAY; this is logged separately in frame label/reason. Low-rate UI shows UNAVAILABLE/processing_fresh_frame during synchronous pose inference to avoid leaving the preceding label visible. UI remains synchronous and may not respond immediately; no sensor-to-screen latency guarantee. Header LOW-RATE EXPERIMENTAL; softmax is uncalibrated model score, not accuracy.

Bounded diagnostics10MB: mode, model/config hashes, observations/span/latest_age/max_support_gap, per-channel masks/coverage/vector/logits, availability reason, label, timing, association scores/reference age. Summary aggregates decisions independent of log truncation. Fixed Performance gaps_over_0_5s remains a strict-reference diagnostic; frame exceeds_continuity_limit uses the selected classifier's gap. The loader does not change dependencies or precision of existing NCNN pose.

## Local verification
5new tests: exact/tolerance1e-12 offline/live vector/logit/mask/label parity for regular/irregular sequences including bounded history trimming; missing detection/reset; expired support; nonincreasing timestamps; schema/class/hash mismatch; no future feature reads.24strict/live/performance/camera regression tests PASS. Same5new tests also PASS inside Windows .venv-ncnn; frozen35-input classifier loads and forwards there. No YOLO/camera/UI or Pi test claimed. Report outputs/low_rate_live_integration_20261030/.

## Transfer ? Windows PowerShell
```powershell
scp -4 -O D:\patient-monitoring-transfer\pi-low-rate-classifier-20261030\pi-low-rate-classifier-update-20261030.tar.gz D:\patient-monitoring-transfer\pi-low-rate-classifier-20261030\pi-low-rate-classifier-update-20261030.tar.gz.sha256 truong@patient-pi4.local:~/
```

## Apply ? Pi terminal, stop on any error
Prior NCNN/live/performance updates and existing pose weights required. No install or weight download. Updater validates all payload hashes, backs up all destinations BEFORE replacing anything, receipt under backups/pi-low-rate-classifier-*; no video/env/pose weights/old reports bundled.
```bash
cd ~
sha256sum -c pi-low-rate-classifier-update-20261030.tar.gz.sha256
mkdir pi-low-rate-classifier-stage-20261030
tar -xzf pi-low-rate-classifier-update-20261030.tar.gz -C pi-low-rate-classifier-stage-20261030
cd ~/pi-low-rate-classifier-stage-20261030
sha256sum -c UPDATE_SHA256SUMS
python3 tools/pi_low_rate_classifier_update.py apply --staging . --project ~/patient-monitoring-pi4
cd ~/patient-monitoring-pi4
.venv-ncnn/bin/python -m pip check
PYTHONPATH=src .venv-ncnn/bin/python -c "from patient_monitoring.live_low_rate import load_low_rate; m,p,s,c=load_low_rate('outputs/low_rate_development_v001_20261030'); print(m.weight.device,m.weight.dtype,s['dimensions'],p['classes'],c['window_seconds'])"
PYTHONPATH=src .venv-ncnn/bin/python -m unittest discover -s tests -p test_live_low_rate_classifier.py -v
```

## UI: run in Pi desktop terminal (ordinary SSH has no desktop)
```bash
PYTHONPATH=src timeout --signal=INT --kill-after=10s 180s .venv-ncnn/bin/python -m patient_monitoring.live_demo --classifier low-rate --camera 0 --backend v4l2 --pose-backend ncnn --imgsz 320 --experimental-association-gap 1.5 --weights models/ncnn_trial_320/yolo11n_pose_320_ncnn_model --model-dir outputs/low_rate_development_v001_20261030 --output-dir outputs/pi_live_low_rate_classifier_001 --duration 60
```
Use new output suffix each run.0?9 select a VERIFIED displayed detection; R reset, Q/Esc stop. When processing_fresh_frame is displayed, wait for detected boxes before selecting.60s loop budget may overrun during inference/cleanup; external180s timeout/10s kill grace bounds hangs; exit124 is not successful completion. No raw recording. Headless --no-display remains unseeded and does not validate classification. To return to strict use --classifier strict plus original supervised_action_development_v001_20261021 directory; do not point strict at low-rate model.

This is development only, one participant, correlated training windows, moving one occasion. NCNN320 pose differs from PT640 training; undervoltage/throttling affects performance; longer association risks identity errors. No held-out accuracy or clinical reliability claim. Observe summary.json/diagnostics.jsonl for availability vs display expiry, not just apparent recognition. M2 stays IN_PROGRESS.
