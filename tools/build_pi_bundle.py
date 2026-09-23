"""Create and verify an allowlisted transfer bundle; standard library only."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

MODEL_SHA256 = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
DOCS = (
    "SETUP.md", "STATUS.md", "PROJECT_PLAN.md", "PI_SETUP.md", "INSPECT_VIDEO.md",
    "PREVIEW_VIDEO.md", "POSE_PREVIEW.md", "EXTRACT_POSES.md", "POSE_REVIEW.md",
    "TARGET_SELECTION.md",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def payload_files(root):
    # Never scan .venv, .git, outputs, model caches or arbitrary raw recordings.
    paths = [root / p for p in (
        "AGENTS.md", "README.md", "pyproject.toml", ".gitignore",
        "data/raw/pilot01.mp4", "models/yolo11n-pose.pt", "tools/build_pi_bundle.py",
    )]
    paths.extend(root / "docs" / name for name in DOCS)
    for folder in (root / "src" / "patient_monitoring", root / "tests"):
        paths.extend(folder.glob("*.py"))
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing/linked/escaped payload file: {path}")
    if not any(p.name == "__main__.py" for p in paths) or not any(p.name.startswith("test_") for p in paths):
        raise ValueError("Source or tests missing")
    if digest(root / "models/yolo11n-pose.pt") != MODEL_SHA256:
        raise ValueError("Weights differ from the verified M2 YOLO11n-pose model")
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def verify_bundle(archive, expected, manifest_bytes):
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(expected) | {"SHA256SUMS"}:
            raise ValueError("Archive member set differs from allowlisted payload")
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            data = handle.extractfile(member).read()
            if member.name == "SHA256SUMS":
                if data != manifest_bytes:
                    raise ValueError("Embedded manifest mismatch")
            elif hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise ValueError(f"Archive hash mismatch: {member.name}")
    return len(names)


def build_bundle(root, output_dir, name):
    root, output_dir = root.resolve(), output_dir.resolve()
    if output_dir == root or output_dir.is_relative_to(root):
        raise ValueError("Bundle and manifest must be outside the source workspace")
    if output_dir.exists():
        raise ValueError("Output directory exists; choose a fresh directory")
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
        raise ValueError("Use letters, digits, hyphens or underscores in bundle name")
    paths = payload_files(root)
    expected = {p.relative_to(root).as_posix(): digest(p) for p in paths}
    manifest_bytes = "".join(f"{h}  {p}\n" for p, h in expected.items()).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=False)
    archive = output_dir / f"{name}.tar.gz"
    with archive.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as handle:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected[relative]:
                raise ValueError(f"Source changed during packaging: {relative}")
            info = tarfile.TarInfo(relative)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            handle.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("SHA256SUMS")
        info.size, info.mode, info.mtime = len(manifest_bytes), 0o644, 0
        handle.addfile(info, io.BytesIO(manifest_bytes))
    member_count = verify_bundle(archive, expected, manifest_bytes)
    if any(digest(root / path) != h for path, h in expected.items()):
        raise ValueError("Payload changed after packaging")
    manifest = output_dir / f"{name}.files.sha256"
    with manifest.open("xb") as handle:
        handle.write(manifest_bytes)
    archive_hash = digest(archive)
    with (output_dir / f"{name}.tar.gz.sha256").open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{archive_hash}  {archive.name}\n")
    report = dict(status="verified_on_windows", archive=str(archive), archive_sha256=archive_hash,
        archive_bytes=archive.stat().st_size, payload_file_count=len(expected), archive_member_count=member_count,
        payload_bytes=sum(p.stat().st_size for p in paths), manifest=str(manifest), model_sha256=MODEL_SHA256,
        verification="Exact member allowlist; every payload SHA-256 matches source and manifest; source rechecked after packaging",
        exclusions=[".venv", ".git", "outputs", "cache", "secrets", "other data/weights"],
        pi_dependency_resolution="not_run", pi_installation="not_run", inference="not_run", benchmark="not_run")
    with (output_dir / "bundle_verification.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="patient-monitoring-pi4-pilot01-20260920")
    args = parser.parse_args()
    try:
        print(json.dumps(build_bundle(args.source, args.output_dir, args.name), indent=2))
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"Bundle creation failed: {exc}\n")
