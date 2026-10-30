# Experimental low-rate association ? 2026-10-29

Strict default remains association0.5s. Explicit --experimental-association-gap1.5 selects an unvalidated1.5s reference lifetime; accepted CLI range >0.5 to2s is an experimental safety bound, not a calibrated identity guarantee. It tolerates the reported0.675s next-frame gap while bounding stale reference age. IoU>=.1 and margin>=.05 remain unchanged. Ambiguous/missing/low-IoU observations never advance reference; expiry permanently ends association until user reseeds. Longer absence can admit a different person with overlapping box: this is NOT reliable identity tracking. No auto-seeding.

FeatureExtractor and make_window stay at0.5s,1s window/0.25s schedule. No preprocessing/model change, interpolation or stale-label holding. Accepted geometry can exist with invalid temporal differences; these are independent states. Missing/invalid gaps reject affected windows. Reset clears temporal history. Existing synchronous inference can delay UI refresh; this update does not make UI asynchronous.

User evidence (conversation, not complete log independently verified):89processed/~1.46/s,69/88gaps>.5;9successful D0 seeds; received178unavailable decisions=129unresolved_or_invalid_timestamp+25timestamp_gap+24startup. Beginning of logs truncated; counts cannot establish full-run event coverage. Example seed->next frame0.675s expires under strict mode. Undervoltage persists; available supply unchanged, performance limitations remain.

Diagnostics: every processed frame includes association status/reason/reference age/reference index/best and runner IoU/separation/candidates plus feature temporal_pair validity/reason/delta. Seed/reset events remain explicit; seed-generated decisions now also logged (previously returned but omitted by UI handler). Summary decision_counts available/unavailable and unavailable_decision_reasons count ALL generated decisions including seed startup; skipped grid slots counted separately, not as classifier outputs. Counters persist across resets; history_reset_events includes reseed resets but excludes construction. Log cap10MB and omitted count retained, so summary may include events beyond retained log. No available prediction implies valid identity or accuracy.

## Local checks and saved-data evidence
5new tests: slow consistent matching, strict expiry, ambiguity retaining reference, disappearance/expiry/reset, option bounds.9live+6performance regressions pass. Synthetic repeated saved pose at0.675s yields matches but invalid temporal pairs and no available windows.
Saved SE10 development poses subsampled sequentially using original PTS at intervals>=0.675s:53frames. This is a timing proxy, NOT reconstruction of Pi truncated logs. Initial detection is a test fixture, not new visual identity verification. Strict:1seed/1timeout/51ended,153unavailable. Experimental1.5s:1seed/2matched/2low-IoU/1timeout/47ended,153unavailable. Reasons strict startup4/timestamp_gap3/unresolved146; experimental startup4/timestamp_gap9/unresolved140. Both0available. Thus sampling remains a blocker even while the first matches survive; actual motion also defeats IoU. No promise of improved classification. Full indices/PTS/source hash: outputs/live_low_rate_20261029/replay.json.

## Windows PowerShell
```powershell
scp -4 -O D:\patient-monitoring-transfer\pi-live-low-rate-20261029\pi-live-low-rate-update-20261029.tar.gz D:\patient-monitoring-transfer\pi-live-low-rate-20261029\pi-live-low-rate-update-20261029.tar.gz.sha256 truong@patient-pi4.local:~/
```

## Pi terminal, after prior NCNN/live updates; stop on errors
```bash
cd ~
sha256sum -c pi-live-low-rate-update-20261029.tar.gz.sha256
mkdir pi-live-low-rate-stage-20261029
tar -xzf pi-live-low-rate-update-20261029.tar.gz -C pi-live-low-rate-stage-20261029
cd ~/pi-live-low-rate-stage-20261029
sha256sum -c UPDATE_SHA256SUMS
python3 tools/pi_live_low_rate_update.py apply --staging . --project ~/patient-monitoring-pi4
cd ~/patient-monitoring-pi4
.venv-ncnn/bin/python -m pip check
```

Use a terminal ON THE PI DESKTOP for this bounded UI experiment (normal SSH has no GUI). Choose a NEW output suffix on every run:
```bash
PYTHONPATH=src timeout --signal=INT --kill-after=10s 180s .venv-ncnn/bin/python -m patient_monitoring.live_demo --camera 0 --backend v4l2 --pose-backend ncnn --imgsz 320 --experimental-association-gap 1.5 --weights models/ncnn_trial_320/yolo11n_pose_320_ncnn_model --model-dir outputs/supervised_action_development_v001_20261021 --output-dir outputs/pi_live_low_rate_001 --duration 60
```
Digits0?9 seed the displayed detection; R clears history/target, Q/Esc stops. Omit --experimental-association-gap to restore strict default.60s capture-loop budget may overrun during current inference/cleanup; timeout124 is not success. Updater verifies hashes and backs up all replaced files before replacement, receipts under backups/pi-live-low-rate-*. No models/data/dependencies bundled. No Pi run in this task; inspect summary and diagnostics separately for matches vs available classifier windows. M2 IN_PROGRESS.
