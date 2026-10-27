# Frozen supervised full-video development demonstrations

## CLI and exact batch commands
`python -m patient_monitoring.supervised_predict --features FEATURES_JSONL --timeline TIMELINE_CSV --model-dir FROZEN_DIR --output-dir NEW_DIR --source SOURCE_MP4`
`--source` is optional; when present it creates a timed review video. Existing output directories are refused. Actual five commands: outputs/supervised_full_video_v001_20261026/commands.ps1 (and commands.json). Do not rerun those commands into existing directories; choose NEW_DIR explicitly.

The batch uses SE08_V002, SE08_V004, SE09_V002, SE10_V001, SE11_V001. The registry supplies only source/features/timeline paths; the prediction CLI does not load annotation CSVs, training labels or baseline predictions. No YOLO, fitting, retraining, thresholds or smoothing. Frozen manifest checks every saved model artifact before and after processing; load_bundle verifies config/schema/model/preprocessing hashes and loads state_dict with weights_only=True.

## Window support and model input
Source-relative decision grid:0,0.25,0.5,... through exact video end (no extra off-grid final decision). Negative support starts before1s are explicit unavailable startup records; thereafter window=[end-1,end). Source-origin PTS and time_base are stored; rational support/decision strings preserve exact grid. Each decision contains seconds, availability/reason, label or null, uncalibrated softmax scores, full aggregated feature values/coverage/missing reasons and observed frame indices/PTS. Never assign an unavailable window a model label.

Reuse supervised_dataset.make_window and aligned_frames unchanged: duration-weighted observed support, backward derivative support entirely inside window, no feature at/after window end, track-break/max0.5s-gap rules identical to dataset generation. Source display durations must align with the supplied timeline. No annotation-boundary resets; only invalid tracks affect availability. Offline frame duration is source timing metadata, not a future pose measurement. One-second window accumulation and grid/frame scheduling cause delay despite no lookahead.

Reuse the frozen28-column vectorizer and transform:14 mean/std,7coverage,7missing flags; training medians/scales only, no fit. Partial/missing features remain represented; no new minimum coverage gate. Video IDs/timestamps are provenance, not predictive inputs. Frozen model still returns a four-class result for available windows, even partial support; scores are not calibrated confidence.

## Review display (separate from predictions)
At each original frame PTS, consume only decisions with decision_end<=frame_relative_PTS. Hold the latest available decision until a new decision. An unavailable decision clears the held label. Current unresolved/invalid target, reseed or invalid temporal continuity clears it immediately; it cannot reappear merely because the subject returns, only after another valid decision. No retroactive label on earlier frames. If a decision occurs between frames, it is first shown at the next frame PTS. A decision exactly at video end remains in JSONL but has no earlier frame on which to display. Initial/cleared state is UNAVAILABLE, not a fifth learned activity.

frame_display.jsonl preserves every source frame index/PTS/time_base, displayed decision time, target status and clear reason. Review renders native-size source pixels with a separate footer, no aspect-ratio stretch, no audio. Encoder fixes source track timescale, PTS and frame durations; decoded output must match every integer PTS/time_base/duration. Errors leave summary incomplete; originals remain unchanged.

## Verification and interpretation
7focused tests plus13supervised-action and12dataset regressions passed before execution: exact training/inference vector parity for identical supports (including a real dataset window), poisoned future features, annotation-independent grid, gaps/reseed rejection, preprocessing without fit, no future display and clearing stale labels. Actual video exports are independently decoded for frame/timing verification.

These are development demonstrations on previously used recordings. Window label counts and displayed-frame counts are diagnostics, NOT accuracy. No accuracy/P/R/F1 is computed on unlabelled regions or inferred from recording intent. Full-video windows differ from annotation-anchored training windows; unlabelled regions remain unlabelled. Same-support/overlap membership, if reported, is provenance only, not an independent test. Retain existing user annotations and frozen baselines unchanged. Do not interpret softmax or a clean-looking demo as patient generalization or deployment readiness.
