import copy,json,tempfile,unittest,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import torch
from test_pose_features import row
from patient_monitoring.live_low_rate import load_low_rate,LowRateEngine,verify_bundle,validate_schema,PINS
from patient_monitoring.low_rate_experiment import recompute,window,apply
ROOT=Path(__file__).resolve().parents[1];MODEL=ROOT/'outputs/low_rate_development_v001_20261030'
class LowRateLiveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.bundle=load_low_rate(MODEL)
 def engine(self):return LowRateEngine(*self.bundle,experimental_association_gap=1.5)
 def test_regular_and_irregular_offline_parity(self):
  for times in [[i*1000000000 for i in range(12)],[0,500000000,1250000000,2250000000,3500000000,4000000000,4750000000,5750000000,7000000000,7500000000,8250000000]]:
   e=self.engine();saved=[];available=0
   for i,t in enumerate(times):
    people=[row(shift=i*.1)['selected_person']];selected,decisions,_=e.step(people,200,100,t,seed=0 if i==0 else None);saved.append(selected)
    feats=recompute(saved,list(range(len(saved))),'live',e.config['max_gap_seconds']);bounds=[F(r['pts'],1000000000) for r in saved]
    for d in decisions:
     if 'available' not in d:continue
     expected,reason=window(feats,bounds,F(d['support_end']),e.config);self.assertEqual(d['reason'],reason);self.assertEqual(d['available'],expected is not None);self.assertLessEqual(d['decision_host_ns'],t)
     if expected:
      available+=1;np.testing.assert_allclose(np.array(d['vector'],float),np.array(expected['vector'],float),equal_nan=True,rtol=0,atol=1e-12)
      with torch.no_grad():logits=e.model(torch.from_numpy(apply(np.asarray([expected['vector']],float),e.prep)))[0]
      np.testing.assert_allclose(d['logits'],logits.numpy(),rtol=0,atol=1e-12);self.assertEqual(d['label'],e.prep['classes'][int(logits.argmax())])
   self.assertGreater(available,0)
 def test_missing_reset_and_expiry(self):
  e=self.engine();person=[row()['selected_person']]
  for i in range(5):e.step(person,200,100,i*750000000,seed=0 if i==0 else None)
  self.assertIsNotNone(e.label);e.expire_display(6000000000);self.assertIsNone(e.label);self.assertEqual(e.reason,'display_support_expired')
  s,ds,_=e.step([],200,100,3750000000);self.assertIsNone(e.label)
  s,ds,_=e.step(person,200,100,4500000000);self.assertFalse(any(d.get('available') for d in ds));self.assertIsNone(e.label)
  e.reset();self.assertFalse(e.history);self.assertIsNone(e.association);e.step(person,200,100,5000000000,seed=0);self.assertIsNone(e.label)
 def test_nonincreasing_timestamp(self):
  e=self.engine();person=[row()['selected_person']];e.step(person,200,100,1,seed=0)
  with self.assertRaises(ValueError):e.step(person,200,100,1)
 def test_schema_and_hash_mismatch(self):
  _,prep,schema,_=self.bundle;s=copy.deepcopy(schema);s['order'].reverse()
  with self.assertRaises(ValueError):validate_schema(s,prep)
  q=copy.deepcopy(prep);q['classes'].reverse()
  with self.assertRaises(ValueError):validate_schema(schema,q)
  with tempfile.TemporaryDirectory() as td:
   for n in PINS:shutil.copyfile(MODEL/n,Path(td)/n)
   verify_bundle(td);(Path(td)/'model_state.pt').write_bytes(b'bad')
   with self.assertRaises(ValueError):load_low_rate(td)
 def test_future_features_not_used_and_history_bounded(self):
  e=self.engine();person=[row()['selected_person']]
  for i in range(80):e.step(person,200,100,i*250000000,seed=0 if i==0 else None)
  self.assertLessEqual(len(e.selected_history),256);self.assertLess(len(e.selected_history),35)
  f=copy.deepcopy(e.history);t=[F(r['pts'],1000000000) for r in f];end=t[-2];v=window(f,t,end,e.config);f[-1]['features']={'poison':True};self.assertEqual(v,window(f,t,end,e.config))
if __name__=='__main__':unittest.main()
