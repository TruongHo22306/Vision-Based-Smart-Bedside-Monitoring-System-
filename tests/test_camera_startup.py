import json,queue,subprocess,sys,threading,unittest
from types import SimpleNamespace
from patient_monitoring.camera_io import capture_loop,startup_timeout
class StartupTests(unittest.TestCase):
 def test_light_imports(self):
  command='import patient_monitoring.live_demo, patient_monitoring.camera_probe,sys,json;print(json.dumps([m for m in ("torch","ultralytics","patient_monitoring.pose_extraction","patient_monitoring.supervised_action") if m in sys.modules]))'
  result=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True,check=True)
  self.assertEqual(json.loads(result.stdout),[])
 def test_stage_timeouts(self):
  for stage,want in [(0,'worker_startup_timeout'),(1,'worker_import_timeout'),(2,'device_open_timeout'),(3,'first_frame_timeout'),(4,'first_frame_timeout'),(5,None)]:
   self.assertEqual(startup_timeout(stage,100,100,11000000100),want)
   self.assertIsNone(startup_timeout(stage,100,100,10000000100))
 def test_timestamped_stages_and_release(self):
  import numpy as np
  cap=SimpleNamespace(isOpened=lambda:True,set=lambda *a:None,read=lambda:(True,np.zeros((2,3,3),dtype=np.uint8)),released=False)
  cap.release=lambda:setattr(cap,'released',True)
  events=queue.Queue();c=[0]*5
  capture_loop(0,queue.Queue(1),threading.Event(),c,SimpleNamespace(value=0),factory=lambda _:cap,backend='dshow',events=events,max_frames=1)
  records=[]
  while not events.empty():records.append(events.get())
  names=[r['stage'] for r in records]
  self.assertEqual(names,['worker_started','opencv_imported','before_VideoCapture','open_returned','before_first_read','first_read_returned','before_release','device_released','worker_finished'])
  self.assertTrue(all(a['monotonic_ns']<=b['monotonic_ns'] for a,b in zip(records,records[1:])))
  self.assertTrue(cap.released);self.assertEqual(c[0],1)
 def test_existing_output_actionable_no_traceback(self):
  import tempfile
  with tempfile.TemporaryDirectory() as td:
   r=subprocess.run([sys.executable,'-m','patient_monitoring.live_demo','--camera','0','--weights','absent','--model-dir','absent','--output-dir',td],capture_output=True,text=True)
   self.assertEqual(r.returncode,2);self.assertIn('NEW --output-dir',r.stderr);self.assertNotIn('Traceback',r.stderr)
if __name__=='__main__':unittest.main()
