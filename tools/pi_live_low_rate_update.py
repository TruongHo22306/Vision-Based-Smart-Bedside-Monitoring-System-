"""Build a incremental live performance update or apply its verified payload with backups."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import uuid

FILES = ('src/patient_monitoring/live_demo.py','tests/test_live_low_rate.py','docs/PI_LIVE_LOW_RATE.md','tools/pi_live_low_rate_update.py')
MANIFEST = "UPDATE_SHA256SUMS"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root, name):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Symlink/escaped path refused: {path}")
    return path


def build(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source) or output.exists():
        raise ValueError("Use a fresh output directory outside the source workspace")
    hashes = {name: sha(checked_path(source, name)) for name in FILES}
    manifest = "".join(f"{hashes[name]}  {name}\n" for name in FILES).encode()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "pi-live-low-rate-update-20261029.tar.gz"
    with archive.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as tar:
        for name in (*FILES, MANIFEST):
            data = manifest if name == MANIFEST else (source/name).read_bytes()
            if name != MANIFEST and hashlib.sha256(data).hexdigest() != hashes[name]:
                raise ValueError("Source changed while packaging")
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    with tarfile.open(archive, "r:gz") as tar:
        assert len(tar.getmembers()) == len(FILES)+1
        assert set(tar.getnames()) == set(FILES) | {MANIFEST}
        for member in tar.getmembers():
            assert member.isfile()
            data = tar.extractfile(member).read()
            assert data == (manifest if member.name == MANIFEST else (source/member.name).read_bytes())
    (output/MANIFEST).write_bytes(manifest)
    (output/(archive.name+'.sha256')).write_text(f"{sha(archive)}  {archive.name}\n", encoding="ascii")
    report = dict(status="archive_verified_on_windows", payload_files=list(FILES),
                  archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                  excluded="videos, pose weights, .venv, dependencies, all outputs and classifier artifacts (prerequisite: prior live update)", pi_applied=False)
    (output/'update_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def apply(staging, project):
    staging, project = staging.resolve(), project.resolve()
    if staging == project or staging.is_relative_to(project):
        raise ValueError("Stage update outside the existing project")
    if not (project/'AGENTS.md').is_file() or not (project/'pyproject.toml').is_file():
        raise ValueError("Existing project AGENTS.md/pyproject.toml not found")
    entries = [line.split('  ',1) for line in (staging/MANIFEST).read_text().splitlines()]
    if len(entries) != len(FILES) or {name for _,name in entries} != set(FILES):
        raise ValueError("Update manifest does not match exact live performance file allowlist")
    hashes = {name:h for h,name in entries}
    previous = {}
    for name in FILES:
        incoming, destination = checked_path(staging,name), checked_path(project,name)
        if not incoming.is_file() or sha(incoming) != hashes[name]:
            raise ValueError(f"Incoming checksum failed: {name}")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"Destination is not a regular file: {name}")
        previous[name] = sha(destination) if destination.exists() else None
    backup_root = checked_path(project, 'backups')
    backup = backup_root / ('pi-live-low-rate-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    backup.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="incomplete", backup_path=str(backup), project=str(project),
                   previous_sha256=previous, incoming_sha256=hashes, replaced=[], errors=[])
    receipt_path = backup/'update_receipt.json'
    def save():
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        # Finish every backup before replacing even the first destination.
        for name in FILES:
            if previous[name] is not None:
                saved = backup/name
                saved.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(project/name,saved)
                if sha(saved) != previous[name]:
                    raise ValueError(f"Backup verification failed: {name}")
        for name in FILES:
            destination = checked_path(project,name)
            if (sha(destination) if destination.exists() else None) != previous[name]:
                raise ValueError(f"Destination changed after backup: {name}")
            destination.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as handle:
                temporary = Path(handle.name)
                handle.write((staging/name).read_bytes())
            try:
                if sha(temporary) != hashes[name]:
                    raise ValueError(f"Incoming file changed: {name}")
                temporary.replace(destination)
            finally:
                if temporary.exists(): temporary.unlink()
            receipt['replaced'].append(name)
            save()
        receipt['status']='complete'
    except Exception as exc:
        receipt['errors'].append(f'{type(exc).__name__}: {exc}')
    save()
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    pack=commands.add_parser('build')
    pack.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    pack.add_argument('--output-dir',type=Path,required=True)
    install=commands.add_parser('apply')
    install.add_argument('--staging',type=Path,default=Path(__file__).resolve().parents[1])
    install.add_argument('--project',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=build(args.source,args.output_dir) if args.command=='build' else apply(args.staging,args.project)
        print(json.dumps(result,indent=2))
        if result['status']=='incomplete': parser.exit(1,'Update incomplete; see backup receipt.\n')
    except (OSError,ValueError,AssertionError) as exc:
        parser.exit(1,f'Update failed: {exc}\n')
