"""Pi preparation tests: synthetic inputs only, no hardware/inference."""
import json,logging,queue,tempfile,threading,unittest,shutil,importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from patient_monitoring import live_demo as live
from patient_monitoring.camera_io import capture_loop
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'outputs/supervised_action_development_v001_20261021'
class PiLiveTests(unittest.TestCase):
 def test_v4l2_explicit_api_and_cleanup(self):
  import cv2
  cap=SimpleNamespace(isOpened=lambda:True,set=lambda *a:None,read=lambda:(True,np.zeros((2,3,3),np.uint8)),release=lambda:None)
  with patch.object(cv2,'VideoCapture',return_value=cap) as open_camera:
   capture_loop(0,queue.Queue(1),threading.Event(),[0]*5,SimpleNamespace(value=0),backend='v4l2',max_frames=1)
   open_camera.assert_called_once_with(0,cv2.CAP_V4L2)
 def test_warning_exact_once_other_messages_and_errors_preserved(self):
  f=live.HalfWarningOnce()
  def record(text,level=logging.WARNING):return logging.LogRecord('ultralytics',level,'',0,text,(),None)
  self.assertTrue(f.filter(record(f.message)));self.assertFalse(f.filter(record(f.message)))
  self.assertTrue(f.filter(record(f.message,logging.ERROR)))
  self.assertTrue(f.filter(record("other warning")));self.assertTrue(f.filter(record("'int8' is deprecated")))
  self.assertEqual(f.suppressed,1)
 def test_minimal_bundle_load_and_hash_verification(self):
  live.load_runtime()
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)
   for n in ('model_state.pt','preprocessing.json','feature_schema.json','config.json','model_hashes.json','frozen_manifest.json'):shutil.copyfile(MODEL/n,out/n)
   live.verify_live_manifest(out);live.load_bundle(out)
   (out/'preprocessing.json').write_text('{}')
   with self.assertRaises(ValueError):live.verify_live_manifest(out)
 def test_low_rate_timeout_and_no_silent_reacquisition(self):
  live.load_runtime();model,prep,schema=live.load_bundle(MODEL)
  people=[dict(detection_index=0,box_xyxy=[10,10,90,90],box_confidence=.9,keypoints=[dict(x=30+i,y=30+i,confidence=.9,passes_quality_gate=True) for i in range(17)])]
  for dt,expected in [(500000000,'matched'),(500000001,'gap_timeout'),(1041947089,'gap_timeout')]:
   e=live.LiveEngine(model,prep,schema);e.step(people,100,100,0,seed=0)
   s,ds,_=e.step(people,100,100,dt);self.assertEqual(s['status'],expected)
   if expected=='gap_timeout':
    self.assertIsNone(e.label);s,_,_=e.step(people,100,100,dt+100000000);self.assertEqual(s['status'],'association_ended')
 def test_update_backup_and_tamper_refusal(self):
  spec=importlib.util.spec_from_file_location('updater',ROOT/'tools/pi_live_update.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);stage=root/'stage';project=root/'project';stage.mkdir();project.mkdir()
   (project/'AGENTS.md').write_text('test');(project/'pyproject.toml').write_text('test')
   for n in u.FILES:
    p=stage/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('new')
   first=u.FILES[0];p=project/first;p.parent.mkdir(parents=True);p.write_text('old')
   (stage/u.MANIFEST).write_text(''.join(u.sha(stage/n)+'  '+n+'\n' for n in u.FILES))
   result=u.apply(stage,project);self.assertEqual(result['status'],'complete');self.assertEqual((Path(result['backup_path'])/first).read_text(),'old')
   (stage/first).write_text('tampered')
   with self.assertRaises(ValueError):u.apply(stage,project)
if __name__=='__main__':unittest.main()
