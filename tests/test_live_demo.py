import copy,json,queue,tempfile,threading,unittest
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import torch
from patient_monitoring.live_demo import LiveEngine,aggregate_live,offer_latest,capture_loop,Log,BASE
from patient_monitoring.supervised_dataset import make_window
from patient_monitoring.supervised_action import load_bundle,vectorize,transform
ROOT=Path(__file__).resolve().parents[1]
class LiveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.model,cls.prep,cls.schema=load_bundle(ROOT/'outputs/supervised_action_development_v001_20261021')
  p=ROOT/'outputs/walking_20261020_v001/P001_SE10_V001/poses_pt640/poses.jsonl'
  with p.open() as f:cls.people=json.loads(next(f))['people']
 def engine(self):return LiveEngine(self.model,self.prep,self.schema)
 def test_timestamp_and_explicit_seed(self):
  e=self.engine();self.assertIsNone(e.step(self.people,1280,720,1000)[0]);self.assertEqual(e.reason,'explicit_seed_required')
  selected,_,_=e.step(self.people,1280,720,1000,seed=0);self.assertEqual(selected['status'],'seeded')
  with self.assertRaises(ValueError):e.step(self.people,1280,720,1000)
 def test_same_inputs_live_offline_parity_and_no_future(self):
  e=self.engine()
  for i in range(13):e.step(self.people,1280,720,i*100000000,seed=0 if i==0 else None)
  hist=list(e.history);end=Fraction(1);live,reason=aggregate_live(hist,end)
  self.assertIsNone(reason)
  past=[r for r in hist if r['pts']*BASE<end];bounds=[r['pts']*BASE for r in past]+[end]
  offline=make_window(past,bounds,0,end)[0]
  np.testing.assert_array_equal(vectorize([live],self.schema),vectorize([offline],self.schema))
  with torch.no_grad():
   a=self.model(torch.from_numpy(transform(vectorize([live],self.schema),self.prep)));b=self.model(torch.from_numpy(transform(vectorize([offline],self.schema),self.prep)))
  torch.testing.assert_close(a,b,rtol=0,atol=0)
  hist[-1]['features']={'poison':True};self.assertEqual(aggregate_live(hist,end)[0],live)
 def test_irregular_intervals_causal(self):
  e=self.engine()
  for n in [0,110000000,240000000,400000000,590000000,790000000,1010000000,1170000000]:
   _,ds,_=e.step(self.people,1280,720,n,seed=0 if n==0 else None)
   self.assertTrue(all(d['decision_host_ns']<=n for d in ds if 'decision_host_ns' in d))
  self.assertAlmostEqual(e.history[-1]['temporal_pair']['delta_seconds'],.16)
 def test_gap_end_and_explicit_reset(self):
  e=self.engine();e.step(self.people,1280,720,0,seed=0)
  s,_,_=e.step(self.people,1280,720,600000000);self.assertEqual(s['status'],'gap_timeout');self.assertIsNone(e.label)
  s,_,_=e.step(self.people,1280,720,700000000);self.assertEqual(s['status'],'association_ended')
  e.reset();self.assertFalse(e.history);self.assertIsNone(e.association);self.assertIsNone(e.label)
  s,_,_=e.step(self.people,1280,720,800000000,seed=0);self.assertEqual(s['status'],'seeded');self.assertEqual(len(e.history),1)
 def test_bounded_history(self):
  e=self.engine()
  for i in range(600):e.step(self.people,1280,720,i*1000000,seed=0 if i==0 else None)
  self.assertLessEqual(len(e.history),256)
 def test_latest_buffer(self):
  q=queue.Queue(maxsize=1);c=[0]*5
  for i in range(100):offer_latest(q,i,c);self.assertEqual(q.qsize(),1)
  self.assertEqual(q.get(),99);self.assertEqual(c[1],99);self.assertEqual(c[2],0)
 def test_camera_failure_releases(self):
  for opened in [False,True]:
   cap=SimpleNamespace(isOpened=lambda:opened,set=lambda *a:None,read=lambda:(False,None),released=False)
   cap.release=lambda:setattr(cap,'released',True)
   state=SimpleNamespace(value=0);capture_loop(0,queue.Queue(1),threading.Event(),[0]*5,state,factory=lambda _:cap)
   self.assertEqual(state.value,-1);self.assertTrue(cap.released)
 def test_capture_stop_releases(self):
  stop=threading.Event();cap=SimpleNamespace(isOpened=lambda:True,set=lambda *a:None,released=False)
  def read():stop.set();return True,np.zeros((2,2,3),np.uint8)
  cap.read=read;cap.release=lambda:setattr(cap,'released',True);c=[0]*5
  capture_loop(0,queue.Queue(1),stop,c,SimpleNamespace(value=0),factory=lambda _:cap)
  self.assertTrue(cap.released);self.assertEqual(c[0],1);self.assertGreater(c[3],0)
 def test_bounded_log(self):
  with tempfile.TemporaryDirectory() as td:
   log=Log(Path(td)/'x.jsonl',max_bytes=100)
   for i in range(100):log.add({'i':i})
   log.close();self.assertLessEqual(log.size,100);self.assertGreater(log.omitted,0)
if __name__=='__main__':unittest.main()
