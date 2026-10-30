"""Explicit local live pose backends; reuse benchmark validation and NCNN setup."""
from pathlib import Path
import importlib.metadata


def validate_pose(weights,backend,imgsz):
    from .pose_benchmark import model_identity
    from .pose_preview import MODEL_SHA256
    if backend not in ('pt','ncnn'):raise ValueError('Unsupported pose backend')
    identity=model_identity(Path(weights),imgsz)
    if identity['backend']!=backend:raise ValueError('Explicit pose backend does not match local weights; no fallback')
    if backend=='pt' and identity['sha256']!=MODEL_SHA256:raise ValueError('Verified original PT weights required')
    return identity


def load_pose(weights,backend,output,config):
    validate_pose(weights,backend,config['imgsz'])
    if backend=='pt':
        from ultralytics import YOLO
        return YOLO(str(Path(weights).resolve()),task='pose')
    # Existing benchmark path calls predictor.setup_model, never .to/.float on NCNN.
    # Its settings path is created before that loader/import, preserving writability.
    output=Path(output).resolve();(output/'library_settings').mkdir(exist_ok=True)
    from .pose_benchmark import load_local_model
    return load_local_model(Path(weights).resolve(),output,config)[0]


def ncnn_snapshot(model,config):
    from .pose_benchmark import runtime_snapshot
    result=runtime_snapshot(model,config)
    if result.get('backend')!='ncnn':raise ValueError('NCNN requested but runtime is not NCNN')
    return result


def package_versions():
    versions={}
    for name in ('torch','ultralytics','ncnn','numpy','opencv-python','av','Pillow'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return versions
