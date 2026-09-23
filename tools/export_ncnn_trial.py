"""Export existing M2 weights in an isolated environment; never download models."""
import argparse
from importlib import metadata
import os
from pathlib import Path
import platform
import shutil
import sys

from patient_monitoring.pose_extraction import write_report
from patient_monitoring.pose_preview import sha256_file

M2_SHA256 = '869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0'


def export_trial(weights, output):
    weights, output = weights.resolve(), output.resolve()
    project = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() == (project / '.venv').resolve():
        raise ValueError('Use a separate trial environment, never the project .venv')
    if not weights.is_file() or sha256_file(weights) != M2_SHA256:
        raise ValueError('Existing verified M2 weights required')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='incomplete', source_sha256=M2_SHA256, source_path=str(weights),
        python=platform.python_version(), platform=platform.platform(), executable=sys.executable,
        versions={d.metadata['Name']: d.version for d in metadata.distributions()},
        export_settings=dict(format='ncnn', imgsz=320, batch=1, device='cpu', quantize=32),
        precision_note='FP32 exported weights (PNNX fp16=False). NCNN runtime precision options are reported separately; not proof every operation uses float32.',
        artifacts={}, errors=[])
    write_report(output/'export_info.json', report)
    staged = output/'export_source.pt'
    try:
        if metadata.version('ultralytics') != '8.4.152' or metadata.version('pnnx') != '20260526':
            raise ValueError('Trial requires audited Ultralytics 8.4.152 and PNNX 20260526')
        os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true',
                          YOLO_CONFIG_DIR=str(output/'settings'), MPLCONFIGDIR=str(output/'mpl_settings'))
        from ultralytics import YOLO
        from ultralytics.utils import downloads
        from ultralytics.utils.export import ncnn as exporter
        def no_download(*args, **kwargs):
            raise RuntimeError('Model/asset downloads disabled for this trial')
        downloads.safe_download = no_download
        report['exporter_source_sha256'] = sha256_file(Path(exporter.__file__))
        shutil.copy2(weights, staged)
        result = Path(YOLO(str(staged), task='pose').export(**report['export_settings'])).resolve()
        if not result.is_relative_to(output):
            raise ValueError('Exporter returned a path outside the new output directory')
        destination = output/'yolo11n_pose_320_ncnn_model'
        result.rename(destination)
        for name in ('model.ncnn.param', 'model.ncnn.bin', 'metadata.yaml'):
            if not (destination/name).is_file() or not (destination/name).stat().st_size:
                raise ValueError(f'Missing exported artifact: {name}')
        report['artifacts'] = {p.relative_to(output).as_posix(): sha256_file(p)
                               for p in sorted(destination.rglob('*')) if p.is_file()}
        if sha256_file(weights) != M2_SHA256:
            raise ValueError('Source weights changed')
        report.update(status='complete', model_directory=str(destination))
    except (Exception, KeyboardInterrupt) as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        staged.unlink(missing_ok=True)
        write_report(output/'export_info.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = export_trial(args.weights, args.output_dir)
    print(result['status'], result['errors'])
    sys.exit(0 if result['status'] == 'complete' else 1)
