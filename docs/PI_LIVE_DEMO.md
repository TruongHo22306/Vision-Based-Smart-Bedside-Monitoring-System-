# Raspberry Pi live development preparation ? 2026-10-27

Status: prepared/tested locally on Windows only. No Pi execution, installation or camera/UI test in this task. Pi user-reported: Pi4 8GB, Bookworm aarch64, Python3.11.2. Windows user reports DSHOW camera1,3604 processed frames, approximately14.49 processed/s, clean shutdown; explicit D0 and four activities appeared correct. This is qualitative user feedback, not accuracy or Codex independent review. Prior Windows working code hashes are in PI_LIVE_FREEZE.json; association of those files to that user run is not independently established.

## Fixed behavior and blockers

Explicit `--backend v4l2 --camera 0` asks OpenCV for CAP_V4L2 index0 (normally /dev/video0); no camera/backend fallback. DSHOW/MSMF/auto retained. Camera process stays lightweight with spawn, one pending frame, overwrite/drop counts, monotonic host read-return timestamps (not sensor exposure). Stage-specific worker-start/import/open/first-read timeouts remain10s; stop waits2s then terminate/join2s. Capture backend may buffer internally despite the buffer request. No nominal FPS substitution.

PT YOLO11n-pose CPU640 batch1 rect=False half=False, conf=.25 IoU=.7; same local weight hash. No NCNN/precision/size change. Classifier remains CPU float64 Linear28x4 with frozen preprocessing; no fit. Exact repeated Ultralytics half-to-quantize WARNING is printed once per run and subsequent identical warnings counted; other warnings/errors remain visible. Inference argument half=False is unchanged.

Critical low-rate audit (implemented rules, not new thresholds):
- TargetAssociator elapsed since last accepted target >0.5s ends association permanently until explicit seed/reset; equality allowed. Min IoU .1, ambiguity margin .05 unchanged. Even identical boxes at1.041947089s spacing time out in synthetic replay.
- FeatureExtractor adjacent processed interval >0.5s invalidates differences; unresolved/seed/source breaks are barriers.
- make_window uses1s trailing support at0.25s grid, rejects any contributing frame span >0.5s, unresolved frame, or track break. It requires complete temporal support, not a fixed frame count. Coverage is duration-weighted per channel; minimum feature coverage is NONE (not80%). Missing channels use frozen training medians and explicit missing flags; sparse valid support is NOT evidence of reliability.
- Earlier Pi PT640 median predict1.041947089s / throughput0.9627 per second is incompatible with continuous association if repeated in live processing. Processing overhead also matters; >2 processed/s is necessary in the uniform-spacing case, NOT sufficient (every accepted-target gap must satisfy the limit). This is a blocker for useful continuous classified live operation under unchanged settings, not for a bounded capture/pose diagnostic. Do not repeatedly reset to manufacture continuity.
- One-second acquisition support and0.25s scheduling do not imply low latency. Computation uses newest available processed frames and actual timestamps; no interpolation. During synchronous inference UI cannot update. Labels clear on invalid target/window/temporal break and when waiting for a fresh frame >0.5s. Slow UI updates can remain on-screen during computation; this is not a real-time clinical display.
- Headless mode intentionally remains UNAVAILABLE/awaiting_explicit_seed; no automatic D0. It measures capture/pose and logs timings only, not live classifier accuracy. Desktop digit0?9 selects that displayed detection, r clears target/history, q/Esc stops. Lost association requires explicit reseeding, no automatic person switch.

## Transfer from Windows PowerShell (one command per line)

```powershell
scp -4 -O D:\patient-monitoring-transfer\pi-live-update-20261027\pi-live-update-20261027.tar.gz D:\patient-monitoring-transfer\pi-live-update-20261027\pi-live-update-20261027.tar.gz.sha256 truong@patient-pi4.local:~/
```

## Pi terminal: verify, stage, back up/apply

Use a fresh staging directory. No remote commands have been run by Codex. Archive contains required code closure (some unchanged supporting modules ensure compatibility with earlier Pi updates), minimal original classifier artifacts, tests and documentation. It excludes pose weights, videos, .venv and historical outputs. Manifest is the original full training manifest; live validates its five inference artifact hashes and load_bundle validates schema/config plus model hashes. Training logs need not be deployed. No dependencies are installed by the updater.

```bash
cd ~
sha256sum -c pi-live-update-20261027.tar.gz.sha256
mkdir pi-live-stage-20261027
tar -xzf pi-live-update-20261027.tar.gz -C pi-live-stage-20261027
cd ~/pi-live-stage-20261027
sha256sum -c UPDATE_SHA256SUMS
python3 tools/pi_live_update.py apply --staging . --project ~/patient-monitoring-pi4
cd ~/patient-monitoring-pi4
.venv/bin/python --version
uname -m
.venv/bin/python -m pip check
.venv/bin/python -c "import cv2,torch,numpy,PIL,av,ultralytics; print('cv2',cv2.__version__,'torch',torch.__version__,'numpy',numpy.__version__,'Pillow',PIL.__version__,'av',av.__version__,'ultralytics',ultralytics.__version__); print(cv2.getBuildInformation())"
.venv/bin/python -c "from pathlib import Path; from patient_monitoring.live_demo import load_runtime,verify_live_manifest; from patient_monitoring.supervised_action import load_bundle; load_runtime(); p=Path('outputs/supervised_action_development_v001_20261021'); verify_live_manifest(p); m,_,_=load_bundle(p); print(m.weight.device,m.weight.dtype)"
.venv/bin/python -c "import hashlib; from pathlib import Path; p=Path('models/yolo11n-pose.pt'); h=hashlib.sha256(p.read_bytes()).hexdigest(); assert h=='869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0'; print(h)"
ls -l /dev/video0
.venv/bin/python -m unittest discover -s tests -p test_pi_live.py -v
```

Stop on any failed checksum/apply/environment check. Updater validates exact allowlist and incoming hashes before replacement, backs up ALL existing destinations before first replacement, verifies backups, atomically replaces each file and records receipt under backups/pi-live-*. No automatic rollback; on failure inspect receipt before continuing. Original PROJECT_PLAN, dependencies and existing results are untouched. Do not copy Windows .venv or install x86/+cpu Windows pins on ARM. GUI requires OpenCV with GUI support plus a local Pi desktop session; ordinary SSH does not provide an interactive desktop.

## Bounded diagnostics (SSH is sufficient)

No thread variable needs to be set; existing thread settings are retained. Run this command directly (60s loop budget after model load; timeout is an external180s safety ceiling, exit124 means incomplete external timeout, not success). Output directory must not exist; choose a new suffix each run. No raw video is recorded.

```bash
timeout --signal=INT --kill-after=10s 180s .venv/bin/python -m patient_monitoring.live_demo --camera 0 --backend v4l2 --weights models/yolo11n-pose.pt --model-dir outputs/supervised_action_development_v001_20261021 --output-dir outputs/pi_live_headless_001 --duration 60 --no-display
```

## Pi desktop terminal: interactive diagnostic

```bash
.venv/bin/python -m patient_monitoring.live_demo --camera 0 --backend v4l2 --weights models/yolo11n-pose.pt --model-dir outputs/supervised_action_development_v001_20261021 --output-dir outputs/pi_live_desktop_001 --duration 60
```

Select a verified displayed D using0?9; r reset; q/Esc stop. Summary/diagnostics record stages, actual processed intervals, target status, decisions, overwritten frames, host age and timing. A complete headless run is not classifier availability. Review measured intervals/UNAVAILABLE before any future explicit decision about a separate low-rate design; this task does not change rules or training distribution. Pi power/cooling unverified; earlier current undervoltage/throttling observations remain relevant. No deployment-performance claim.
