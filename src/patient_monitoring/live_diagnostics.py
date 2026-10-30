"""Bounded live diagnostics; stdlib imports only, no thread setters."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import os
import re
import shutil
import subprocess
import time

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


def image_size(value):
    import argparse
    try:
        n=int(value)
    except (TypeError,ValueError):raise argparse.ArgumentTypeError('imgsz must be a positive integer multiple of 32')
    if str(n)!=str(value) or n<=0 or n%32:raise argparse.ArgumentTypeError('imgsz must be a positive integer multiple of 32')
    return n


def prepare_settings(output):
    paths={}
    for key,folder in [('YOLO_CONFIG_DIR','ultralytics_settings'),('MPLCONFIGDIR','matplotlib_settings')]:
        p=(Path(output)/folder).resolve();p.mkdir(parents=True,exist_ok=True)
        check=p/'.write_check'
        with check.open('x') as f:f.write('writable')
        check.unlink();os.environ[key]=str(p);paths[key]=str(p)
    os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    return paths


def thread_snapshot(stage,torch):
    return dict(stage=stage,monotonic_ns=time.monotonic_ns(),intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads(),torch_version=torch.__version__)


class Performance:
    def __init__(self):
        self.first=None;self.count=0;self.total=0.;self.minimum=None;self.maximum=None
        self.gaps=0;self.gap_total=0.;self.gap_max=None;self.over_limit=0
    def observe(self,seconds,interval):
        if self.first is None:self.first=seconds
        else:
            self.count+=1;self.total+=seconds
            self.minimum=seconds if self.minimum is None else min(seconds,self.minimum)
            self.maximum=seconds if self.maximum is None else max(seconds,self.maximum)
        if interval is not None:
            self.gaps+=1;self.gap_total+=interval;self.gap_max=interval if self.gap_max is None else max(interval,self.gap_max)
            self.over_limit+=int(interval>.5)
    def summary(self):
        return dict(first_predict_seconds=self.first,subsequent_predict_count=self.count,subsequent_predict_sum_seconds=self.total,subsequent_predict_mean_seconds=self.total/self.count if self.count else None,subsequent_predict_min_seconds=self.minimum,subsequent_predict_max_seconds=self.maximum,processed_timestamp_gap_count=self.gaps,processed_timestamp_gap_mean_seconds=self.gap_total/self.gaps if self.gaps else None,processed_timestamp_gap_max_seconds=self.gap_max,processed_timestamp_gaps_over_0_5s=self.over_limit,continuity_limit_seconds=.5,continuity_note='Adjacent gaps >0.5s exceed feature/window limit and necessarily exceed target last-accepted limit if seeded; shorter gaps alone do not guarantee association. No policy changes.',predict_timing='model.predict call only, preprocessing/postprocessing included; excludes PIL conversion, quality serialization, telemetry, classifier and UI. First includes lazy setup/warmup. Not camera FPS.')
