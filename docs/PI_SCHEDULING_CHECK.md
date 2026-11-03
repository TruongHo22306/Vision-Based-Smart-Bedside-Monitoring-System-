# Pi scheduling/accounting check - 2026-11-02

This packages the completed scheduling/accounting fix without further runtime edits. Existing low-rate model/config/schema/preprocessing, TTL1.25s, feature gap1.5s, association gap1.5s and dependencies remain unchanged. Requires the previous low-rate and async UI updates on Pi. Code/tests/docs only; no weights, videos, environment or old outputs.

User feedback: UI is easier for selecting the person; qualitative conversation feedback only. User has NOT tried R. There is no reported reset failure. Automated reset tests are separate from user hardware testing. No new Pi run was performed while preparing this package.

## Windows PowerShell
```powershell
scp -4 -O D:\patient-monitoring-transfer\pi-scheduling-20261102\pi-scheduling-update-20261102.tar.gz D:\patient-monitoring-transfer\pi-scheduling-20261102\pi-scheduling-update-20261102.tar.gz.sha256 truong@patient-pi4.local:~/
```
## Pi terminal: verify, backup, apply
Stop any running live demo first. Use a NEW staging directory; if it exists choose a new suffix, do not overwrite.
```bash
cd ~
sha256sum -c pi-scheduling-update-20261102.tar.gz.sha256
mkdir pi-scheduling-stage-20261102
tar -xzf pi-scheduling-update-20261102.tar.gz -C pi-scheduling-stage-20261102
cd ~/pi-scheduling-stage-20261102
sha256sum -c UPDATE_SHA256SUMS
python3 tools/pi_scheduling_update.py apply --staging . --project ~/patient-monitoring-pi4
cd ~/patient-monitoring-pi4
.venv-ncnn/bin/python -m pip check
PYTHONPATH=src .venv-ncnn/bin/python -m unittest discover -s tests -p test_live_async_ui.py -v
.venv-ncnn/bin/python -m unittest discover -s tests -p test_report_live_scheduling.py -v
```
The updater verifies all incoming checksums, backs up every replaced file before replacement, checks backup hashes and writes backups/pi-scheduling-*/update_receipt.json. Inspect an incomplete receipt rather than blindly retrying. Model/config hashes are verified by the existing runtime loader.

## 60-second UI check on Pi desktop
Interactive UI needs the Pi desktop terminal (plain SSH without GUI access is insufficient). Output must not exist.
```bash
PYTHONPATH=src timeout --signal=INT --kill-after=10s 150s .venv-ncnn/bin/python -m patient_monitoring.live_demo --async-ui --classifier low-rate --camera 0 --backend v4l2 --pose-backend ncnn --imgsz 320 --experimental-association-gap 1.5 --weights models/ncnn_trial_320/yolo11n_pose_320_ncnn_model --model-dir outputs/low_rate_development_v001_20261030 --output-dir outputs/pi_live_scheduling_60s_001 --duration 60
```
0-9 selects the actual detection on the displayed processed image. Q/Esc stops. R is optional explicit reset, not a workaround for a reported defect. 60s is the main-loop budget after initialization; cleanup adds time. External timeout exit124 is not a successful complete run.

## Produce a NEW post-run report (no inference)
```bash
.venv-ncnn/bin/python tools/report_live_scheduling.py outputs/pi_live_scheduling_60s_001 --output-dir outputs/pi_live_scheduling_report_60s_001
```
Return summary.json, diagnostics.jsonl and the new timing_report.json for analysis. Original logs remain unchanged. The report distinguishes missing timestamps and incomplete/truncated logs instead of estimating unavailable timings.

Report definitions:
- Frame age at worker start: process_started_host_ns minus host_acquired_ns, all logged pose calls. It is host acquisition age, not sensor latency.
- Predict: recorded model.predict call, first versus subsequent reported separately.
- Classifier: classifier_completed_host_ns minus classifier_started_host_ns; includes association/features/windows and classifier forwards, not just Linear execution.
- Publication delay: published_host_ns minus completed_host_ns (includes worker logging/telemetry).
- Wait for UI: ui_received_host_ns minus published_host_ns; also report receipt-to-assessment separately. These are not pure render duration.
- Publication/UI timings use each received pose result ONCE; seed-only/control results and repeated UI refreshes are excluded from that population. Overwritten/discarded results without receipt cannot supply this measurement.
- Decision counts remain historical model availability; display_summary contains first received-result assessments, including seed results; async_ui contains repeated update counts, support expiry and fresh-label submission duration. None proves a human saw the label or that it was accurate.
- Accounting shows capture replacements/drops, async pre-processing discards, corrected residual and legacy residual separately; result drops already processed are not subtracted twice. Shutdown includes worker stopped, capture alive/exitcode and forced termination.

No claim of improved support freshness/availability before this check. Compare actual timing populations and first/subsequent predict separately. Existing 20261031/earlier archives and all previous outputs are preserved.
