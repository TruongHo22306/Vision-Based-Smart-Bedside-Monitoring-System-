# Scoped YOLO11n-pose NCNN trial

M2 remains IN_PROGRESS. No model replacement, tracking, action recognition or Pi execution in this task.

## Evidence and RGB/BGR audit

User reports Pi 4 8GB / Bookworm aarch64 / Python 3.11.2; PT CPU float32,
batch=1, rect=False, 3 warmups and 30 measured frames. At 640: predict throughput
0.96272884545/s, median 1.041947089 s. At 320: 2.67687469585/s, median 0.3741457125 s.
One run per size, different initial temperatures; preliminary, not a controlled backend comparison.
Power supply and cooling remain unverified. Record conditions; this does not block the trial.

Audited installed Ultralytics **8.4.152**, not an assumed older implementation:
`pose_extraction.py` and `pose_benchmark.py` pass PyAV `frame.to_image()` (PIL RGB).
`data/loaders.py:LoadPilAndNumpy._single_check` converts PIL RGB to BGR;
`engine/predictor.py:BasePredictor.preprocess` converts BGR to normalized RGB.
`pose-preview` also passes PIL RGB. No mismatch found, no channel correction applied,
and no old baseline invalidated for color handling. A synthetic red-dominant RGB round-trip test
exercises these installed functions. A future NumPy source must be BGR, not an unconverted RGB array.
This agrees with the official [input contract](https://docs.ultralytics.com/modes/predict/#inference-sources).

## Export actually performed on Windows

`tools/export_ncnn_trial.py` requires the exact existing M2 weights SHA-256
`869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`.
It refuses the project `.venv`, missing/wrong weights and existing output directories;
automatic installs/model downloads are disabled. It exports a temporary copy and removes that copy.
Original weights are never modified. A failed export retains `export_info.json` with incomplete status.

The installed exporter `engine/exporter.py:export_ncnn` calls `utils/export/ncnn.py:torch2ncnn`.
Requirements are NCNN and **pnnx==20260526**; the installed code pins PNNX for its documented
20260704 inference-segfault issue. See [official exporter](https://docs.ultralytics.com/reference/utils/export/ncnn/),
[NCNN integration](https://docs.ultralytics.com/integrations/ncnn/), and [PNNX](https://github.com/pnnx/pnnx).
Do not substitute the latest exporter recipe for this audited version.

Export succeeded: imgsz=320, batch=1, device=cpu, quantize=32, PNNX `fp16=0`, input `[1,3,320,320]f32`.
Environment `.venv-ncnn` has separate writable site-packages and a `.pth` that reuses baseline project
packages read-only; it is an overlay, not a portable/self-contained environment. No baseline package was installed/upgraded.
Added NCNN 1.0.20260526 and PNNX 20260526; subsequently completed declared dependencies with
portalocker 4.4.0, tqdm 4.70.1, colorama 0.4.6. Python stayed 3.10.11; Torch 2.9.1+cpu,
Ultralytics 8.4.152 and other baseline versions stayed unchanged. Both environments pass `pip check`.

Local model: `models/ncnn_trial_320/yolo11n_pose_320_ncnn_model/`.
`models/ncnn_trial_320/export_info.json` records source/artifact SHA-256, exporter-code hash,
all versions at export, settings and precision caveat. Artifact bundle includes this report and model files,
but no original PT weights or intermediate tracing weights. It does not contain an executable environment.

FP32 weight export does **not** guarantee all NCNN runtime operations use float32.
NCNN has hardware-dependent default FP16 options. We leave these defaults unchanged and record
`net.opt` flags, `num_threads`, CPU device and float32 input separately. Torch thread counts describe
host work, not NCNN execution. Windows observed NCNN num_threads=8 and Vulkan=False;
FP16 packed/storage/arithmetic enabled by default. These observations are not Pi settings or timings.

## Benchmark and comparison contracts

`benchmark-pose --weights LOCAL_ncnn_model --imgsz 320` reuses the existing measured frame prefix,
PTS, conf=.25, IoU=.7, batch=1, rect=False, warmup exclusion, telemetry and no-overwrite rules.
The NCNN Net loads during model-loading timing. PT behavior remains as before.
Directory inputs require the three unambiguous runtime files `model.ncnn.param`, `model.ncnn.bin`,
`metadata.yaml`, batch=1/pose/17x3 metadata and matching export imgsz. Other image sizes require
a separate explicitly requested export; no automatic re-export. Runtime file hashes and a canonical
combined model hash are recorded. Input changes or failures produce incomplete reports.

`compare-pose-backends` defaults to the first 3 sequential frames (maximum 30), identical PIL RGB
pixels and prediction settings for both backends. It exports `comparison.json`, no images.
Matching uses greedy descending box IoU>=0.5, one-to-one, array-index tie break; this is comparison-only,
not tracking or an inference-threshold change. All detections, unmatched indices, raw 17 joints,
confidences and quality flags remain represented. Each match reports box IoU and per-joint pixel
dx/dy/distance plus confidence pairs. Means divide by finite joint pairs only; null means unavailable.
Low-confidence/partial-body joints remain in the difference report. Differences measure consistency,
not accuracy, visibility or quality; exact numerical equality is neither required nor expected.

Windows checks: 41 focused pose tests pass; NCNN smoke 1 warmup/2 measured frames complete.
`outputs/ncnn_smoke_windows_20260923/` records telemetry unavailable and actual backend options.
`outputs/ncnn_consistency_windows_20260923/comparison.json`: frames 0,1,2 have 1,1,2 detections
on both backends; all 4 boxes match, IoU>=0.99999909, maximum joint difference 0.000367 pixels.
Indices/PTS match M1.1. These tiny Windows checks are neither Pi benchmarks nor accuracy validation.

## Windows PowerShell: transfer, one line at a time

The two bundles are outside the source; do not overwrite a previously received archive on Pi.
Stop after any failed command. All following Pi commands are instructions, not executed by Codex.

```powershell
Set-Location D:\patient-monitoring-transfer\pi-ncnn-update-20260923
scp -4 -O pi-ncnn-update-20260923.tar.gz pi-ncnn-update-20260923.tar.gz.sha256 "truong@patient-pi4.local:~/"
Set-Location D:\patient-monitoring-transfer\pi-ncnn-artifacts-20260923
scp -4 -O pi-ncnn-artifacts-20260923.tar.gz pi-ncnn-artifacts-20260923.tar.gz.sha256 "truong@patient-pi4.local:~/"
```

## Terminal on Pi: verify and update with backups

Run each line separately; continue only after success. Fresh staging and model destinations required.

```bash
cd "$HOME"
sha256sum -c pi-ncnn-update-20260923.tar.gz.sha256
mkdir "$HOME/pi-ncnn-update-20260923" && tar -xzf pi-ncnn-update-20260923.tar.gz -C "$HOME/pi-ncnn-update-20260923" --keep-old-files --no-same-owner
cd "$HOME/pi-ncnn-update-20260923" && sha256sum -c UPDATE_SHA256SUMS
python3 "$HOME/pi-ncnn-update-20260923/tools/pi_ncnn_update.py" apply --project "$HOME/patient-monitoring-pi4"
cd "$HOME"
sha256sum -c pi-ncnn-artifacts-20260923.tar.gz.sha256
mkdir "$HOME/pi-ncnn-artifacts-20260923" && tar -xzf pi-ncnn-artifacts-20260923.tar.gz -C "$HOME/pi-ncnn-artifacts-20260923" --keep-old-files --no-same-owner
cd "$HOME/pi-ncnn-artifacts-20260923" && sha256sum -c NCNN_SHA256SUMS
test ! -e "$HOME/patient-monitoring-pi4/models/ncnn_trial_320" && mv -T -n "$HOME/pi-ncnn-artifacts-20260923/ncnn_trial_320" "$HOME/patient-monitoring-pi4/models/ncnn_trial_320"
cd "$HOME/patient-monitoring-pi4"
```

Updater validates its exact 10-file allowlist/checksums, backs up **all** existing files before replacing
any, then verifies each replacement. Receipt/backups: `backups/pi-ncnn-<UTC>-<id>/`.
An incomplete receipt means stop and inspect; backup copies remain available. No dependency/data/weights/output/PLAN changes.

## Terminal on Pi: isolated runtime and trial

Keep the existing `.venv` unchanged. The helper creates a new local overlay using that interpreter
and imports its packages read-only, as on Windows. If `.venv-ncnn` already exists, inspect it and
choose a new `.venv-*` name consistently; never clear or recreate an existing environment.
Binary-wheel-only dry-run must succeed on the actual aarch64 Python before installation.
No Torch source build or system Python changes; ARM64 runtime compatibility remains unverified locally.

```bash
.venv/bin/python tools/create_ncnn_env.py --output-dir .venv-ncnn
.venv-ncnn/bin/python -m pip install --dry-run --only-binary=:all: --no-deps ncnn==1.0.20260526 portalocker==4.4.0 tqdm==4.70.1 colorama==0.4.6
.venv-ncnn/bin/python -m pip install --only-binary=:all: --no-deps ncnn==1.0.20260526 portalocker==4.4.0 tqdm==4.70.1 colorama==0.4.6
.venv-ncnn/bin/python -m pip check
.venv/bin/python -m pip check
.venv-ncnn/bin/python -m patient_monitoring compare-pose-backends data/raw/pilot01.mp4 --pt-weights models/yolo11n-pose.pt --ncnn-model models/ncnn_trial_320/yolo11n_pose_320_ncnn_model --imgsz 320 --frames 3 --output-dir outputs/pi_ncnn_consistency_run01
.venv-ncnn/bin/python -m patient_monitoring benchmark-pose data/raw/pilot01.mp4 --weights models/ncnn_trial_320/yolo11n_pose_320_ncnn_model --imgsz 320 --warmup 3 --frames 30 --output-dir outputs/pi_ncnn_320_run01
.venv-ncnn/bin/python -m patient_monitoring benchmark-pose data/raw/pilot01.mp4 --weights models/yolo11n-pose.pt --imgsz 320 --warmup 3 --frames 30 --output-dir outputs/pi_pt_320_comparison_run01
```

Let temperature settle to comparable levels before each benchmark; record power/cooling/load/order.
Use new run names for repeats. Do not interpret predict-call throughput as webcam or full-pipeline FPS.
No speedup conclusion from Windows or a single Pi pair with different starting conditions.
If wheel resolution or runtime fails, keep incomplete reports and report the blocker; do not claim Pi inference success.

## Optional reproducible export, not required for the transferred artifact

These Pi export commands have **not** been executed. Use the same isolated overlay, never `.venv`.

```bash
.venv-ncnn/bin/python -m pip install --dry-run --only-binary=:all: --no-deps pnnx==20260526
.venv-ncnn/bin/python -m pip install --only-binary=:all: --no-deps pnnx==20260526
.venv-ncnn/bin/python -m pip check
.venv-ncnn/bin/python tools/export_ncnn_trial.py --weights models/yolo11n-pose.pt --output-dir models/ncnn_trial_320_pi_export
```

The exporter requires audited Ultralytics 8.4.152 and PNNX 20260526; do not upgrade baseline dependencies to bypass a failure.
