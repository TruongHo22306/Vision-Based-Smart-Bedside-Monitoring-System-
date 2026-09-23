# Pi NCNN results and eight-image visual review — 2026-09-23

## Received Pi files: completed, with observed current power/throttle flags

Read directly, without modifying files, from `D:/patient-monitoring-transfer/pi-ncnn-results-20260923/`:
`pi_ncnn_consistency_run01/comparison.json` and `pi_ncnn_320_run01/{benchmark_summary.json,frame_timings.jsonl,telemetry.jsonl}`.
This task inspected received results; it did not connect to Pi or execute any new Pi benchmark.
File SHA-256 and audit evidence: `outputs/pilot01_backend_visual_review_20260923/pi_results_audit.json`.

- Comparison is complete: frames 0,1,2, PT/NCNN detection counts 1,1,2 each; four matched boxes, no unmatched detections.
  Recomputed matches from raw predictions; maximum corresponding-joint distance 0.0010376257051154846 pixels.
  This is consistency on three frames, not accuracy or proof that the detected people/joints are correct.
- Benchmark is complete: three warmups excluded, 30/30 measured frames (0–29), no errors.
  Recomputed median **0.4827727220 s**, p95 **0.5361398882 s**, predict throughput **2.1587532648/s**.
  Throughput denominator is **13.8969100770 s summed measured predict-call time**; not camera/full-pipeline FPS.
  Total benchmark elapsed **25.1156623610 s**, with preparation/loading/warmup/telemetry included as documented.
- Video, PT and NCNN artifact hashes match local originals; frame indices/PTS/time bases match the M1.1 timeline.
  Runtime reports Python 3.11.2/aarch64, Ultralytics 8.4.152, Torch 2.9.1, NCNN 1.0.20260526.
  CPU, imgsz320, batch1, rect=False, conf=.25/IoU=.7; NCNN threads=4, Torch intraop/interop=3/4 after loading.
  Input is float32; internal NCNN precision is not established by input dtype or FP32 weight export.

Telemetry was inspected from raw `get_throttled` values and decoded again, not inferred from the summary:

- All 32 samples are available. **30/32** have `0x50005`: current undervoltage (bit 0) and throttling (bit 2),
  plus historical undervoltage/throttling (bits 16/18). This includes after warmup and **29/30** measured-frame samples.
- Before warmup and after measured frame 26 have `0x50000`, historical flags only at those two samples.
  After final frame 29: `0x50005`; current undervoltage/throttling were still observed at that sample.
- Temperature: before warmup 61.8°C, after warmup 62.8°C, after last measured frame 63.3°C; sampled range 60.8–63.7°C.
  Sample counts are not fractions of elapsed time; no claim about conditions between readings or the exact cause.
- **No backend speed-advantage conclusion.** Earlier PT runs had different starting temperatures and one run per size;
  current power/throttle flags confound timing. Power supply/cooling remain unverified. M2 remains IN_PROGRESS.

## New Windows visual review, same eight source PNGs

Output: `outputs/pilot01_backend_visual_review_20260923/`.
Sources come only from `outputs/pilot01_preview/samples.csv`: frames **0,59,118,177,237,296,355,414** with exact CSV timestamps.
Columns are original PT640, original PT320, exported NCNN320; identical RGB source pixels, rect=False and existing conf/IoU.
Used the existing `.venv-ncnn`, no downloads/dependency changes. Eight PT640 predictions were reused only after checking
CSV/PNG/model SHA-256, frame/time/dimensions, requested/effective settings, CPU/float32 and relevant package versions.
PT320/NCNN320 required **16 new predict calls total**; no other frames or Pi benchmarks were run.

- `backend_review_sheet.jpg`: three columns × eight rows; overview 1920×3296.
- `comparison_frame_000000.png` through the eight selected frame filenames: 3840×800, each source panel kept at 1280×720 plus caption.
- `poses.json`: all raw boxes/confidences and 17 original x/y/confidence values per detection; zero/multiple detections represented.
- `quality_flags.json`: separate finite/confidence/boundary/gate flags keyed by frame, variant, detection and joint.
- `run_info.json`: source hashes, exact timestamps, package/model/config provenance, reuse proof, runtime and human-review pending.
- `verification.json` and `pi_results_audit.json`: actual validation evidence; prior inputs/outputs remain unchanged.

Drawing reuses `pose_review.annotate_review` and its current gate: finite coordinates/confidence, confidence>=0.5, strictly
inside image bounds. Every finite box is drawn; skeleton edges require both passing endpoints. Pink X marks boundary
knees/ankles separately; display clipping never changes raw values. Other boundary flags remain in JSON. D indices are
frame-local, not tracking IDs. Confidence/gate/agreement do not establish accuracy or visibility; partial-body cases remain.

PT640 has 8 total detections; PT320 and NCNN320 have 11 each, with additional detections at frames 59,118,237.
No automatic false-positive labels were assigned. All 30 detections / 510 joint observations remain present.
Checked CSV indices/timestamps/hashes/pixel hashes, exact preservation of reused raw predictions, every flag and image dimensions.
**44 focused pose tests PASS; pip check PASS in both unchanged environments.** Contact sheet was opened to inspect rendering;
human pose assessment is still pending. Next action: review corresponding knees/ankles and extra boxes in the full-size PNGs.

Generator: `tools/generate_pose_backend_review.py#generate`; focused tests: `tests/test_pose_backend_visual_review.py`.
If explicitly generating another review later, choose a new output directory; the utility refuses existing output directories.
