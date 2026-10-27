"""Sequential bounded OpenCV camera probe; never imports pose/classifier modules."""
import argparse
import json
import time
from pathlib import Path
from .camera_io import Capture


def attempt(camera,backend):
    started=time.monotonic();capture=Capture(camera,backend,max_frames=1);reason=None
    try:
        while True:
            capture.diagnostics()
            if capture.counters[0]>0:reason='frame_received';break
            if capture.status.value<0:reason='worker_reported_failure';break
            timeout=capture.timeout_reason()
            if timeout:reason=timeout;break
            if not capture.process.is_alive():reason='worker_exited_without_frame';break
            capture.newest(.05)
    finally:cleanup=capture.close()
    if cleanup['capture_process_alive']:raise RuntimeError('Worker still alive; refusing next backend probe')
    return dict(camera=camera,backend=backend,outcome=reason,elapsed_seconds=time.monotonic()-started,**cleanup)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera',type=int,default=0);p.add_argument('--output-dir',required=True);a=p.parse_args()
    out=Path(a.output_dir)
    try:out.mkdir(parents=True,exist_ok=False)
    except FileExistsError:p.exit(2,f'Output exists: {out}. Choose a NEW --output-dir; nothing overwritten.\n')
    results=[]
    for backend in ('dshow','msmf'):
        result=attempt(a.camera,backend);results.append(result)
        (out/(backend+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(attempts=results,sequential=True,no_pose_or_classifier=True),indent=2),encoding='utf-8')


if __name__=='__main__':main()
