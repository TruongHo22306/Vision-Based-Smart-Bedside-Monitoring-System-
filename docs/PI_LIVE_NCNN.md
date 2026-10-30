# Explicit NCNN live integration ? 2026-10-28

Prerequisite: previous live + live-performance updates already applied. No dependency installation or model export/download. Default pose backend remains pt, imgsz640; camera --backend remains independent (DSHOW on Windows, V4L2 on Linux). NCNN requires --pose-backend ncnn --imgsz320 and the existing local directory. Wrong backend/path/metadata/shape is an error, never PT fallback.

live_pose.validate_pose reuses benchmark model_identity: three nonempty local files, unambiguous param, task pose, batch1,17x3 keypoints, matching input size; hashes all files and a canonical artifact-map digest. PT still requires original verified hash. NCNN load reuses load_local_model and explicit predictor.setup_model; no PT .to/.float calls on exported models. Offline/autoinstall-disabled environment set before imports. Existing absolute writable config directories are retained; benchmark library_settings is created before loading. No device index substitution.

OpenCV BGR -> PIL RGB -> same Ultralytics predict contract; original image coordinates returned and quality_people keeps every detection/17 raw keypoints plus masks. Conf.25/IoU.7/batch1/rectFalse/halfFalse unchanged. Explicit target selection/reset, gap0.5s, feature/window semantics, bounded capture, telemetry and UNAVAILABLE are unchanged. NCNN setup is done during load; first predict still includes backend warmup. Therefore first-predict timing is not equivalent to PT lazy setup timing; subsequent timings retained.

Summary records explicit pose_backend, artifact hashes/export metadata, package versions, runtime settings after load/first predict, NCNN Net options (including num_threads and available precision flags). Torch counts describe Torch host/classifier, NOT NCNN threads. Input float32/export quantize32 is not proof of every NCNN operation's precision. Classifier stays CPU float64, frozen weights/preprocessing/schema/config, weights_only load; no fit.

User-reported Pi PT320:106processed, subsequent mean0.496116s,34/105 host timestamp gaps>0.5s; sampled undervoltage/throttling. Only current supply available. Continue software integration; measurements are power/thermal-condition dependent, not optimal performance or backend superiority. No automatic gap relaxation or reacquisition. NCNN speed/availability on Pi remains unverified by this task.

Local Windows verification:30 tests:3 backend +6 performance +9 live +4 startup +8 existing NCNN tests. Existing .venv-ncnn ncnn1.0.20260526/ultralytics8.4.152 loads frozen classifier CPU float64. One saved-image1280x720 NCNN320 smoke detects1person/17keypoints and returns original shape. This is NOT a Pi/webcam test or accuracy measurement. Runtime options include FP16 capabilities; no per-op precision claim. See outputs/live_ncnn_integration_20261028/. No new model/dependency/annotation changes.

## Windows PowerShell
```powershell
scp -4 -O D:\patient-monitoring-transfer\pi-live-ncnn-20261028\pi-live-ncnn-update-20261028.tar.gz D:\patient-monitoring-transfer\pi-live-ncnn-20261028\pi-live-ncnn-update-20261028.tar.gz.sha256 truong@patient-pi4.local:~/
```

## Pi terminal: stop on any error, one command per line
```bash
cd ~
sha256sum -c pi-live-ncnn-update-20261028.tar.gz.sha256
mkdir pi-live-ncnn-stage-20261028
tar -xzf pi-live-ncnn-update-20261028.tar.gz -C pi-live-ncnn-stage-20261028
cd ~/pi-live-ncnn-stage-20261028
sha256sum -c UPDATE_SHA256SUMS
python3 tools/pi_live_ncnn_update.py apply --staging . --project ~/patient-monitoring-pi4
cd ~/patient-monitoring-pi4
.venv-ncnn/bin/python -m pip check
.venv-ncnn/bin/python -c "import sys;sys.path.insert(0,'src');from patient_monitoring.live_pose import package_versions,validate_pose;print(package_versions());print(validate_pose('models/ncnn_trial_320/yolo11n_pose_320_ncnn_model','ncnn',320))"
.venv-ncnn/bin/python -c "import sys;sys.path.insert(0,'src');from pathlib import Path;from patient_monitoring.live_demo import load_runtime,verify_live_manifest;from patient_monitoring.supervised_action import load_bundle;load_runtime();p=Path('outputs/supervised_action_development_v001_20261021');verify_live_manifest(p);m,_,_=load_bundle(p);print(m.weight.device,m.weight.dtype)"
PYTHONPATH=src timeout --signal=INT --kill-after=10s 180s .venv-ncnn/bin/python -m patient_monitoring.live_demo --camera 0 --backend v4l2 --pose-backend ncnn --imgsz 320 --weights models/ncnn_trial_320/yolo11n_pose_320_ncnn_model --model-dir outputs/supervised_action_development_v001_20261021 --output-dir outputs/pi_live_ncnn320_diagnostic_001 --duration 60 --no-display
```

Classifier compatibility check MUST pass in Pi .venv-ncnn; local Windows success does not prove Pi load. No install workaround is automatic. Updater validates checksums, backs up replaced files before replacement, receipt under backups/pi-live-ncnn-*. Six code/test/doc files; weights, classifier artifacts, videos, environments and old outputs excluded. New output/staging suffix required if already present. Existing pose_benchmark included to ensure exact reused loader version.

60s loop budget plus current predict/shutdown; external180s limit returns124 on timeout, not success. Headless remains unseeded, so classifier aggregation/identity tracking are not exercised. For interaction use a Pi desktop terminal, remove --no-display and choose a new output directory. Digits0-9 explicitly select displayed detection, R resets target/history, Q/Esc stops. No silent reseed after timeout. Share summary.json and diagnostics.jsonl; do not interpret processing throughput as sensor FPS or accuracy.
