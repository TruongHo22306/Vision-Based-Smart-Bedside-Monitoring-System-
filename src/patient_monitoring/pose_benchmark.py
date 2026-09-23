"""Short CPU predict-call benchmark; no images, pose exports or model downloads."""

from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from time import perf_counter

import av

from .pose_extraction import write_report
from .pose_preview import PREDICT_CONFIG, sha256_file
from .video_inspection import InspectionError, TimelineAudit

FLAGS = ("undervoltage", "arm_frequency_capped", "throttled", "soft_temperature_limit")


def throttle_flags(value):
    return dict(current={name: bool(value & (1 << i)) for i, name in enumerate(FLAGS)},
                historical={name: bool(value & (1 << (i+16))) for i, name in enumerate(FLAGS)},
                unknown_bits_hex=hex(value & ~0xF000F))


def sample_telemetry():
    """Two sequential commands, <=1 second each; failures never mean 'no throttling'."""
    started = perf_counter()
    result = dict(sampled_at_utc=datetime.now(timezone.utc).isoformat(), status="unavailable",
        temperature_celsius=None, throttled_hex=None, flags=None, raw={}, errors=[])
    executable = shutil.which("vcgencmd")
    if executable is None:
        result["errors"].append("vcgencmd unavailable on PATH")
    else:
        for command in ("measure_temp", "get_throttled"):
            try:
                completed = subprocess.run([executable, command], capture_output=True, text=True,
                                           timeout=1.0, check=True)
                raw = completed.stdout.strip()
                result["raw"][command] = raw
                if command == "measure_temp":
                    match = re.fullmatch(r"temp=(-?\d+(?:\.\d+)?)'C", raw)
                    if not match:
                        raise ValueError("unrecognized temperature response")
                    result["temperature_celsius"] = float(match[1])
                else:
                    match = re.fullmatch(r"throttled=(0x[0-9a-fA-F]+)", raw)
                    if not match:
                        raise ValueError("unrecognized throttle response")
                    value = int(match[1], 16)
                    result.update(throttled_hex=hex(value), flags=throttle_flags(value))
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                result["errors"].append(f"{command}: {type(exc).__name__}: {exc}")
        available = sum(result[k] is not None for k in ("temperature_celsius", "throttled_hex"))
        result["status"] = ("unavailable", "partial", "available")[available]
    result["collection_seconds"] = perf_counter() - started
    return result


def percentile(values, fraction):
    """Linear interpolation at rank (n-1)*fraction on ascending samples (type 7)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank-lo)


def timing_statistics(rows):
    measured = [r for r in rows if r["phase"] == "measured" and r["status"] == "complete"]
    seconds = [r["predict_seconds"] for r in measured]
    denominator = sum(seconds)
    components = {}
    for key in ("preprocess", "inference", "postprocess"):
        values = [r["ultralytics_speed_ms"].get(key) for r in measured]
        values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        components[key] = dict(sample_count=len(values), median_ms=percentile(values, .5), p95_ms=percentile(values, .95))
    return dict(measured_completed=len(measured), predict_seconds_sum=denominator,
        predict_median_seconds=percentile(seconds, .5), predict_p95_seconds=percentile(seconds, .95),
        predict_call_throughput_frames_per_second=len(seconds)/denominator if denominator > 0 else None,
        throughput_denominator="Sum of successful measured model.predict wall seconds only; excludes warmup, decode/preparation, model load, telemetry and output I/O; NOT camera or full-pipeline FPS",
        percentile_method="Linear interpolation on sorted samples at rank (n-1)*q; median q=.5, p95 q=.95; no rounding before aggregation",
        ultralytics_reported_components=components)


def prepare_frames(source, count, prepared):
    """First N decoded frames, independent of imgsz/warmup; no seek/FPS resampling."""
    audit = TimelineAudit()
    with av.open(str(source)) as container:
        if not container.streams.video:
            raise InspectionError("No video stream")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            row = audit.observe(frame.pts, frame.time_base)
            if frame.is_corrupt:
                raise InspectionError(f"Corrupt decoded frame {row['frame_index']}")
            row.update(stream_index=stream.index, width=frame.width, height=frame.height)
            prepared.append((row, frame.to_image()))
            if len(prepared) == count:
                break
    if len(prepared) != count:
        raise InspectionError(f"Requested {count} frames, only decoded {len(prepared)}")
    return audit.summary()


def torch_threads():
    import torch
    return dict(intraop=torch.get_num_threads(), interop=torch.get_num_interop_threads())


def model_identity(weights, imgsz=None):
    """Hash local PT or an unambiguous NCNN directory; never resolve remote models."""
    if weights.is_file() and weights.suffix.lower() == '.pt' and weights.stat().st_size:
        return dict(backend='pt', sha256=sha256_file(weights), artifacts=None)
    if not weights.is_dir() or not weights.name.endswith('_ncnn_model'):
        raise InspectionError('Existing nonempty .pt or local *_ncnn_model directory required; no download')
    import yaml
    required = ('model.ncnn.param', 'model.ncnn.bin', 'metadata.yaml')
    if any(not (weights/n).is_file() or not (weights/n).stat().st_size for n in required):
        raise InspectionError('NCNN directory requires model.ncnn.param/bin and metadata.yaml')
    if {p.name for p in weights.glob('*.param')} != {'model.ncnn.param'}:
        raise InspectionError('Ambiguous NCNN parameter files')
    info = yaml.safe_load((weights/'metadata.yaml').read_text(encoding='utf-8'))
    if not isinstance(info, dict) or info.get('task') != 'pose' or info.get('kpt_shape') != [17, 3] or info.get('batch') != 1:
        raise InspectionError('NCNN metadata must describe batch=1, pose, 17x3 keypoints')
    if imgsz is not None and info.get('imgsz') != [imgsz, imgsz]:
        raise InspectionError('imgsz must match the NCNN export shape')
    artifacts = {n: sha256_file(weights/n) for n in required}
    digest = hashlib.sha256(json.dumps(artifacts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(backend='ncnn', sha256=digest, artifacts=artifacts, export_metadata=info,
                hash_definition='SHA256 of UTF-8 sorted compact JSON filename-to-SHA256 map of three runtime files')


def load_local_model(weights, output, config=None):
    # Check twice: never let Ultralytics treat an absent path as a download request.
    identity = model_identity(weights, config.get('imgsz') if config else None)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_CONFIG_DIR"] = str(output / "library_settings")
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_settings")
    import torch
    before = torch_threads()
    from ultralytics import YOLO
    model = YOLO(str(weights), task="pose")
    if model.task != "pose":
        raise InspectionError("Weights must be a pose model")
    if identity['backend'] == 'pt':
        model.to("cpu")
        model.model.float()
    else:
        # Load NCNN Net during model-loading timing, not the first measured predict.
        model.predictor = model._smart_load('predictor')(
            overrides={**(config or PREDICT_CONFIG), 'mode': 'predict', 'task': 'pose'},
            _callbacks=model.callbacks)
        model.predictor.setup_model(model=model.model, verbose=False)
    return model, before


def runtime_snapshot(model, config):
    predictor = getattr(model, "predictor", None)
    args = getattr(predictor, "args", None)
    backend = getattr(predictor, "model", None)
    if backend is not None and getattr(backend, 'format', None) == 'ncnn':
        net = backend.backend.net
        if str(backend.device) != 'cpu' or backend.fp16 or net.opt.use_vulkan_compute:
            raise InspectionError('Expected NCNN CPU with float32 input and Vulkan disabled')
        names = ('num_threads', 'use_vulkan_compute', 'use_fp16_packed', 'use_fp16_storage',
                 'use_fp16_arithmetic', 'use_bf16_storage', 'use_int8_inference', 'use_packing_layout')
        return dict(backend='ncnn', actual_model_device=str(backend.device),
            actual_input_dtype='torch.float32', actual_model_dtype=None,
            precision_note='NCNN internal arithmetic is hardware/layer dependent; options are capabilities, not proof of per-operation precision.',
            ncnn_options={n: getattr(net.opt, n, None) for n in names},
            torch_threads=torch_threads(), torch_threads_note='Host preprocessing only; NOT NCNN thread count',
            effective_predict_config={key: getattr(args, key) for key in config if hasattr(args, key)},
            runtime_unreported_config=[key for key in config if not hasattr(args, key)])
    parameter = next(model.model.parameters())
    device, dtype = str(parameter.device), str(parameter.dtype)
    if device != "cpu" or dtype != "torch.float32":
        raise InspectionError(f"Expected CPU float32, observed {device}/{dtype}")
    if backend is not None and (str(backend.device) != "cpu" or backend.fp16):
        raise InspectionError("Predictor backend is not CPU float32")
    return dict(actual_model_device=device, actual_model_dtype=dtype, torch_threads=torch_threads(),
        effective_predict_config={key: getattr(args, key) for key in config if hasattr(args, key)},
        runtime_unreported_config=[key for key in config if not hasattr(args, key)])


def benchmark_pose(source: Path, weights: Path, output: Path, imgsz=640, warmup=3, frames=30):
    started = perf_counter()
    source, weights, output = (p.expanduser().resolve() for p in (source, weights, output))
    if output.exists():
        raise InspectionError("Output exists; choose a fresh benchmark directory")
    output.mkdir(parents=True, exist_ok=False)
    config = {**PREDICT_CONFIG, "imgsz": imgsz}
    report = dict(schema_version=1, status="incomplete", completed=False,
        started_at_utc=datetime.now(timezone.utc).isoformat(), source_path=str(source), weights_path=str(weights),
        video_sha256=None, model_sha256=None, requested_config=config,
        warmup_requested=warmup, measured_requested=frames, warmup_completed=0,
        python_version=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
        package_versions={d.metadata["Name"]: d.version for d in metadata.distributions()},
        input_hashing_seconds=None, data_preparation_seconds=None, model_loading_seconds=None,
        selected_frames=[], runtime_snapshots={}, errors=[],
        selection_policy="First N decoded frames on first video stream. Warmup cycles this prepared set separately. Selection independent of imgsz and warmup count.",
        telemetry_note="Sequential sampled readings, outside predict timer. Historical flags are not current flags; no conclusion about throttling between samples.",
        thread_policy="No thread count setters or explicit optimizations in benchmark. Library defaults may change threads; snapshots record observations.",
        total_time_definition="perf_counter from benchmark function entry (after CLI/module import) through cleanup and statistics; includes hashes, prep, load, warmup, telemetry and intermediate writes; excludes final report serialization",
    )
    report_path = output / "benchmark_summary.json"
    write_report(report_path, report)
    prepared, rows = [], []
    stage = "validate_inputs"
    try:
        if type(imgsz) is not int or imgsz <= 0 or imgsz % 32:
            raise InspectionError("imgsz must be a positive multiple of 32; no silent rounding")
        if type(warmup) is not int or warmup < 0 or type(frames) is not int or frames <= 0:
            raise InspectionError("warmup must be >=0 and frames must be >0")
        if not source.is_file() or not source.stat().st_size:
            raise InspectionError("Video must be an existing nonempty local file")
        tick = perf_counter()
        identity = model_identity(weights, imgsz)
        stage = "hash_inputs"
        report.update(video_sha256=sha256_file(source), model_sha256=identity['sha256'], model_identity=identity,
                      input_color_contract='PyAV PIL RGB -> Ultralytics BGR loader -> RGB normalized model input')
        report["input_hashing_seconds"] = perf_counter() - tick
        stage = "data_preparation"
        tick = perf_counter()
        try:
            report["timestamp_audit"] = prepare_frames(source, frames, prepared)
        finally:
            report["data_preparation_seconds"] = perf_counter() - tick
            report["selected_frames"] = [row for row, _ in prepared]
        stage = "model_loading"
        tick = perf_counter()
        try:
            model, before = load_local_model(weights, output, config)
            report["torch_threads_before_model_load"] = before
        finally:
            report["model_loading_seconds"] = perf_counter() - tick
        report["runtime_snapshots"]["after_model_load"] = runtime_snapshot(model, config)
        write_report(report_path, report)
        with (output / "frame_timings.jsonl").open("x", encoding="utf-8") as timings, \
                (output / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry:
            def sample(event, index=None):
                sample_row = dict(event=event, source_frame_index=index,
                    benchmark_elapsed_seconds=perf_counter()-started, **sample_telemetry())
                telemetry.write(json.dumps(sample_row, allow_nan=False) + "\n")
                telemetry.flush()

            sample("before_warmup")
            for phase, count in (("warmup", warmup), ("measured", frames)):
                for ordinal in range(count):
                    metadata_row, image = prepared[ordinal % len(prepared)]
                    stage = f"{phase}_predict"
                    record = dict(phase=phase, ordinal=ordinal, **metadata_row, status="incomplete")
                    tick = perf_counter()
                    try:
                        results = model.predict(source=image, **config)
                    except (Exception, KeyboardInterrupt) as exc:
                        record.update(predict_seconds=perf_counter()-tick, error=f"{type(exc).__name__}: {exc}")
                        timings.write(json.dumps(record, allow_nan=False) + "\n")
                        timings.flush()
                        raise
                    elapsed = perf_counter() - tick
                    stage = f"{phase}_result"
                    if len(results) != 1 or tuple(results[0].orig_shape) != (metadata_row["height"], metadata_row["width"]):
                        raise InspectionError("Expected exactly one result with original frame dimensions")
                    prediction = results[0]
                    record.update(status="complete", predict_seconds=elapsed,
                        person_count=len(prediction.boxes),
                        ultralytics_speed_ms={k: float(v) if v is not None and math.isfinite(v) else None
                                               for k, v in prediction.speed.items()})
                    rows.append(record)
                    timings.write(json.dumps(record, allow_nan=False) + "\n")
                    timings.flush()
                    if phase == "warmup":
                        report["warmup_completed"] += 1
                    else:
                        sample("after_measured_frame", metadata_row["frame_index"])
                        if (ordinal+1) % 5 == 0:
                            print(f"Measured {ordinal+1}/{frames} frames", flush=True)
                            report["statistics"] = timing_statistics(rows)
                            write_report(report_path, report)
                if phase == "warmup":
                    sample("after_warmup")
                    report["runtime_snapshots"]["after_warmup"] = runtime_snapshot(model, config)
            report["runtime_snapshots"]["after_measured"] = runtime_snapshot(model, config)
        stage = "verify_inputs_unchanged"
        if sha256_file(source) != report["video_sha256"] or model_identity(weights, imgsz)['sha256'] != report["model_sha256"]:
            raise InspectionError("Video or weights changed during benchmark")
        report.update(status="complete", completed=True)
    except (Exception, KeyboardInterrupt) as exc:
        report["errors"].append(dict(stage=stage, type=type(exc).__name__, message=str(exc)))
    finally:
        for _, image in prepared:
            image.close()
    report["statistics"] = timing_statistics(rows)
    report["statistics"]["partial_run"] = not report["completed"]
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["total_benchmark_elapsed_seconds"] = perf_counter() - started
    write_report(report_path, report)
    return report
