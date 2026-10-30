import argparse,contextlib,io,json,logging,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import numpy as np
from patient_monitoring import live_demo as live
from patient_monitoring.live_diagnostics import Performance,image_size,prepare_settings,sample_telemetry,throttle_flags
ROOT=Path(__file__).resolve().parents[1]
class PerformanceTests(unittest.TestCase):
 def test_size_and_no_global_config_mutation(self):
  self.assertEqual(image_size(640),640);self.assertEqual(image_size('320'),320)
  for v in (0,-32,33,'320.5',True):
   with self.assertRaises(argparse.ArgumentTypeError):image_size(v)
 def test_first_subsequent_gap_equality(self):
  p=Performance();self.assertIsNone(p.summary()['subsequent_predict_mean_seconds'])
  p.observe(8,None);p.observe(2,.5);p.observe(4,.50001)
  s=p.summary();self.assertEqual(s['first_predict_seconds'],8);self.assertEqual(s['subsequent_predict_mean_seconds'],3);self.assertEqual(s['processed_timestamp_gaps_over_0_5s'],1)
 def test_settings_existing_absolute_writable(self):
  with tempfile.TemporaryDirectory() as td,patch.dict(os.environ):
   paths=prepare_settings(td)
   for k,p in paths.items():self.assertTrue(Path(p).is_absolute());self.assertTrue(Path(p).is_dir());self.assertEqual(os.environ[k],p)
 def test_telemetry_unavailable_and_historical(self):
  with patch('patient_monitoring.live_diagnostics.shutil.which',return_value=None):self.assertEqual(sample_telemetry()['status'],'unavailable')
  f=throttle_flags(0x50000);self.assertFalse(any(f['current'].values()));self.assertTrue(f['historical']['undervoltage']);self.assertTrue(f['historical']['throttled'])
 def test_telemetry_timeout_bound(self):
  import subprocess
  with patch('patient_monitoring.live_diagnostics.shutil.which',return_value='vcgencmd'),patch('patient_monitoring.live_diagnostics.subprocess.run',side_effect=subprocess.TimeoutExpired('vcgencmd',1)) as run:
   s=sample_telemetry();self.assertEqual(s['status'],'unavailable');self.assertEqual(run.call_count,2)
   self.assertTrue(all(c.kwargs['timeout']==1 for c in run.call_args_list))
 def test_live_stage_order_config_and_no_thread_setters(self):
  live.load_runtime();calls=[]
  class Pose:
   def __init__(self,*a,**kw):
    self.check_dirs();calls.append('yolo_init')
   def check_dirs(self):
    for k in ('YOLO_CONFIG_DIR','MPLCONFIGDIR'):self_path=Path(os.environ[k]);assert self_path.is_absolute() and self_path.is_dir()
   def predict(self,**kw):
    self.check_dirs();calls.append(('predict',kw['imgsz']));assert kw['half'] is False and kw['rect'] is False
    return [SimpleNamespace(speed={'inference':1.})]
  class Capture:
   def __init__(self,*a):self.status=SimpleNamespace(value=1);self.counters=[2,0,0];self.n=0;self.drained=0
   def diagnostics(self):return []
   def timeout_reason(self):return None
   def newest(self):
    if self.n==2:raise KeyboardInterrupt
    self.n+=1;return self.n,self.n*600000000,np.zeros((10,10,3),np.uint8)
   def close(self):return dict(captured=2,capture_first_host_ns=600000000,capture_last_host_ns=1200000000,overwritten_pending=0,dropped_incoming=0,drained_older=0)
  ultra=ModuleType('ultralytics');ultra.YOLO=Pose;utils=ModuleType('ultralytics.utils');utils.LOGGER=logging.getLogger('test-live')
  with tempfile.TemporaryDirectory() as td,patch.dict(os.environ),patch.dict('sys.modules',{'ultralytics':ultra,'ultralytics.utils':utils}),patch.object(live,'Capture',Capture),patch.object(live,'quality_people',return_value=[]),patch.object(live,'sample_telemetry',return_value={'status':'unavailable'}),patch.object(live.torch,'set_num_threads',side_effect=AssertionError('setter reached')),patch.object(live.torch,'set_num_interop_threads',side_effect=AssertionError('setter reached')),contextlib.redirect_stdout(io.StringIO()):
   args=SimpleNamespace(camera=0,backend='v4l2',weights=ROOT/'models/yolo11n-pose.pt',model_dir=ROOT/'outputs/supervised_action_development_v001_20261021',output_dir=Path(td)/'new',duration=60,no_display=True,imgsz=320)
   result=live.run(args)
  self.assertEqual(result['processed'],2)
  self.assertEqual([x['stage'] for x in result['thread_snapshots']],['before_classifier_load','after_classifier_load','after_yolo_initialization','after_first_predict'])
  self.assertEqual(result['pose_config']['imgsz'],320);self.assertEqual(live.PREDICT_CONFIG['imgsz'],640)
  self.assertEqual(result['processed_timestamp_gaps_over_0_5s'],1);self.assertEqual(result['subsequent_predict_count'],1)
  self.assertEqual(calls,['yolo_init',('predict',320),('predict',320)])
if __name__=='__main__':unittest.main()
