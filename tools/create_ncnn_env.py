"""Create an isolated writable trial overlay, reusing project packages read-only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import venv


def create(output):
    project = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (project/'.venv').resolve():
        raise ValueError('Run with the existing project .venv interpreter')
    output = output.resolve()
    if output.exists() or output.parent != project or not output.name.startswith('.venv-'):
        raise ValueError('Use a new .venv-* directly inside this project')
    baseline = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    venv.EnvBuilder(with_pip=True).create(output)
    python = output/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    site = Path(subprocess.check_output([str(python), '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip())
    (site/'project_baseline.pth').write_text(sysconfig.get_paths()['purelib']+'\n'+str(project/'src')+'\n', encoding='utf-8')
    (output/'overlay_info.json').write_text(json.dumps(dict(base_python=sys.executable,
        baseline_site_packages=sysconfig.get_paths()['purelib'], baseline_pip_freeze=baseline,
        policy='Separate writable site-packages. Reuse baseline imports read-only; install extras here with --no-deps. Never upgrade baseline packages.'), indent=2)+'\n', encoding='utf-8')
    print(f'Created {python}; no packages installed. Follow docs/PI_NCNN_TRIAL.md.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    create(parser.parse_args().output_dir)
