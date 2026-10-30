import copy,json,unittest,argparse
from pathlib import Path
from patient_monitoring.live_demo import LiveEngine,association_gap
from patient_monitoring.supervised_action import load_bundle
ROOT=Path(__file__).resolve().parents[1]
class LowRateTests(unittest.TestCase):
 @classmethod
 def setUpClass(c):
  c.bundle=load_bundle(ROOT/'outputs/supervised_action_development_v001_20261021')
  with (ROOT/'outputs/walking_20261020_v001/P001_SE10_V001/poses_pt640/poses.jsonl').open() as f:c.people=json.loads(next(f))['people'][:1]
 def engine(self,gap=1.5):return LiveEngine(*self.bundle,experimental_association_gap=gap)
 def test_slow_match_no_feature_bridge(self):
  e=self.engine()
  for i in range(8):
   s,ds,_=e.step(self.people,1280,720,i*675000000,seed=0 if i==0 else None)
   self.assertEqual(s['status'],'seeded' if i==0 else 'matched')
   self.assertFalse(any(d.get('available') for d in ds));self.assertIsNone(e.label)
   if i:self.assertEqual(e.latest_step['feature_temporal_pair']['reason'],'maximum_time_gap_exceeded')
  self.assertGreater(e.unavailable_reasons['timestamp_gap'],0)
 def test_strict_default(self):
  e=self.engine(None);e.step(self.people,1280,720,0,seed=0)
  self.assertEqual(e.step(self.people,1280,720,675000000)[0]['status'],'gap_timeout')
 def test_ambiguity_does_not_advance_reference(self):
  e=self.engine();e.step(self.people,1280,720,0,seed=0)
  people=copy.deepcopy(self.people)*2;people[1]=copy.deepcopy(people[0]);people[1]['detection_index']=1
  s,_,_=e.step(people,1280,720,675000000);self.assertEqual(s['status'],'ambiguous')
  s,_,_=e.step(self.people,1280,720,1600000000);self.assertEqual(s['status'],'gap_timeout')
 def test_disappearance_expiration_and_reset(self):
  e=self.engine();e.step(self.people,1280,720,0,seed=0)
  self.assertEqual(e.step([],1280,720,700000000)[0]['status'],'no_detections')
  self.assertEqual(e.step(self.people,1280,720,1600000000)[0]['status'],'gap_timeout')
  self.assertEqual(e.step(self.people,1280,720,1700000000)[0]['status'],'association_ended')
  e.reset();self.assertIsNone(e.association);self.assertFalse(e.history)
  self.assertEqual(e.step(self.people,1280,720,1800000000,seed=0)[0]['status'],'seeded')
 def test_option_bounds(self):
  for n in (0,.5,2.01,float('inf'),float('nan')):
   with self.assertRaises(argparse.ArgumentTypeError):association_gap(n)
  self.assertEqual(association_gap(1.5),1.5)
if __name__=='__main__':unittest.main()
