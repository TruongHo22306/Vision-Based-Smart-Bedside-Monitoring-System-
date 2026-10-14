"""One v002 correction: image motion does not veto supported sitting/lying posture.

Reuse unchanged v001 time windows, thresholds, masks and barriers. No new state,
debounce or label holding. Image-motion uncertainty remains separately visible.
"""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from . import action_baseline as v1

VERSION = "action_rules_v002"
POLICY = "non_upright_posture_independent_of_image_motion"


class BaselineV2(v1.Baseline):
    def step(self, row):
        result = super().step(row)
        result["baseline_version"] = VERSION
        result["v001_decision"] = {k: result[k] for k in ("predicted_label", "reasons")}
        result["image_motion_cautions"] = []
        posture = result["posture_evidence"]
        if posture in ("sitting", "lying"):
            # v001 only exposes posture after all six channels pass their original coverage gates.
            result["locomotion_evidence"] = dict(status="unknown", reason="image_motion_does_not_establish_locomotion_in_non_upright_pose")
            if result["reasons"] in (["motion_threshold_band"], ["motion_posture_conflict"]):
                result["image_motion_cautions"] = result["reasons"][:]
                result["predicted_label"] = posture
                result["reasons"] = ["supported_posture_independent_of_image_motion"]
        else:
            result["locomotion_evidence"] = dict(
                status="candidate" if result["predicted_label"] == "moving" else "unknown",
                reason="unchanged_v001_upright_motion_heuristic")
        return result


def run_baseline(path, output, config=v1.Config()):
    path, output = Path(path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output exists; use a NEW versioned directory")
    digest = v1.sha256(path)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(thresholds_and_windows=asdict(config), decision_policy=POLICY,
                    additional_temporal_delay_seconds=0, added_temporal_stabilization=False)
    config_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    provenance = dict(baseline_version=VERSION, status="incomplete", input_path=str(path), input_sha256=digest,
                      config=settings, config_sha256=config_hash, code_sha256=v1.sha256(__file__),
                      v001_code_sha256=v1.sha256(v1.__file__), python_version=platform.python_version(),
                      created_at_utc=datetime.now(timezone.utc).isoformat(), inference_performed=False,
                      annotations_used=False, thresholds_validated=False)
    summary = dict(status="incomplete", completed=False, frame_count=0, label_counts={}, unknown_reason_counts={},
                   image_motion_caution_counts={}, changed_from_v001_count=0, transitions=[], errors=[],
                   denominator="All written rows. Cautions may coexist with a posture label. Counts are diagnostics, not accuracy.")
    v1.write_json(output / "provenance.json", provenance)
    v1.write_json(output / "summary.json", summary)
    labels, unknowns, cautions = Counter(), Counter(), Counter()
    previous, engine = None, BaselineV2(config)
    try:
        with path.open(encoding="utf-8") as source, (output / "predictions.jsonl").open("x", encoding="utf-8") as dest:
            for line in source:
                row = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {v}")))
                result = engine.step(row)
                result.update(input_sha256=digest, config_sha256=config_hash)
                dest.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                label = result["predicted_label"]
                labels[label] += 1
                if label == "unknown":
                    unknowns.update(result["reasons"])
                cautions.update(result["image_motion_cautions"])
                summary["changed_from_v001_count"] += int(label != result["v001_decision"]["predicted_label"])
                if previous and previous["video_id"] == result["video_id"] and previous["predicted_label"] != label:
                    summary["transitions"].append(dict(frame_index=result["frame_index"], pts=result["pts"],
                        time_base=result["time_base"], presentation_time_seconds=result["presentation_time_seconds"],
                        from_label=previous["predicted_label"], to_label=label, reasons=result["reasons"],
                        window_reset_reason=result["window"]["reset_reason"]))
                previous = result
                summary["frame_count"] += 1
        if not summary["frame_count"]:
            raise ValueError("Empty feature input")
        if v1.sha256(path) != digest:
            raise ValueError("Input changed during run")
        summary.update(status="complete", completed=True)
    except (Exception, KeyboardInterrupt) as exc:
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    summary.update(label_counts={k: labels[k] for k in ("standing", "sitting", "lying", "moving", "unknown")},
                   unknown_reason_counts=dict(unknowns), image_motion_caution_counts=dict(cautions),
                   transition_count=len(summary["transitions"]))
    provenance.update(status=summary["status"], errors=summary["errors"],
                      output_sha256=v1.sha256(output / "predictions.jsonl") if (output / "predictions.jsonl").exists() else None)
    v1.write_json(output / "summary.json", summary)
    v1.write_json(output / "provenance.json", provenance)
    return summary
