import copy,unittest,tempfile,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import torch
from test_pose_features import row
from test_live_async_ui import result
from patient_monitoring.live_acquisition import AcquisitionEngine,load_acquisition,verify_bundle,PINS
from patient_monitoring.acquisition_features import AcquisitionFeatures,aggregate,freshness
from patient_monitoring.acquisition_action import vectorize,load_bundle
from patient_monitoring.low_rate_experiment import apply
from patient_monitoring.live_async_ui import UIState
MODEL=Path(__file__).resolve().parents[1]/'outputs/acquisition_action_development_20261103_v001'
class AcquisitionLiveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.bundle=load_acquisition(MODEL)
 def engine(self):return AcquisitionEngine(*self.bundle,experimental_association_gap=1.5)
 def test_offline_live_exact_vector_logits_acquisition_timing(self):
  for times in [[i*500000000 for i in range(12)],[i*1000000000 for i in range(7)],[0,500000000,1250000000,2250000000,3500000000,4000000000,4750000000,5750000000]]:
   e=self.engine();offline=AcquisitionFeatures('live');count=0
   for i,t in enumerate(times):
    selected,ds,_=e.step([row(shift=i*.1)['selected_person']],200,100,t,seed=0 if i==0 else None)
    _,r=offline.step(selected)
    self.assertEqual(bool(ds),r is not None)
    if r is None:continue
    d=ds[0];self.assertEqual(d['decision_host_ns'],t);self.assertEqual(d['available'],r['available']);self.assertEqual(d['reason'],r['reason'])
    if r['available']:
     count+=1;self.assertEqual(d['vector'],r['vector']);self.assertEqual(d['channels'],r['channels'])
     with torch.no_grad():z=e.model(torch.from_numpy(apply(vectorize([r],e.schema),e.prep)))[0]
     np.testing.assert_array_equal(d['logits'],z.numpy());self.assertEqual(d['label'],e.prep['classes'][int(z.argmax())])
     self.assertEqual(d['support_deadline_host_ns'],int(F(r['expires_at'])*1000000000))
   self.assertGreater(count,0)
 def test_missing_reset_gap_and_no_stale_label(self):
  e=self.engine();person=[row()['selected_person']]
  for i in range(5):e.step(person,200,100,i*750000000,seed=0 if i==0 else None)
  self.assertIsNotNone(e.label);e.expire_display(4250000001);self.assertIsNone(e.label)
  _,ds,_=e.step([],200,100,3750000000);self.assertFalse(ds[-1]['available']);self.assertIsNone(e.display_support_host_ns)
  _,ds,_=e.step(person,200,100,4500000000);self.assertFalse(ds[-1]['available'])
  e.reset();self.assertFalse(e.features.history);self.assertIsNone(e.current_record);self.assertIsNone(e.label)
  e.step(person,200,100,6000000000,seed=0)
  _,ds,_=e.step(person,200,100,8000000000);self.assertFalse(ds[-1]['available']);self.assertIsNone(e.label)
 def test_missing_channel_does_not_refresh_and_ui_deadline(self):
  e=self.engine();person=row()['selected_person']
  for i in range(6):e.step([person],200,100,i*500000000,seed=0 if i==0 else None)
  p=copy.deepcopy(person)
  for j in p['keypoints']:j['passes_quality_gate']=False
  _,ds,_=e.step([p],200,100,3000000000)
  d=ds[-1];used=[c for c in d['channel_support'].values() if c['used']]
  self.assertTrue(any(c['support_host_ns']==2500000000 for c in used))
  self.assertEqual(d['support_deadline_host_ns'],3750000000)
  s=UIState(1.25);r=result(stamp=3000000000,support=d['latest_support_host_ns']);r.update(support_deadline_host_ns=d['support_deadline_host_ns'],channel_support=d['channel_support']);s.accept(r)
  self.assertIsNotNone(s.tick(3750000000)['label']);self.assertIsNone(s.tick(3750000001)['label'])
  s.reset(1,4000000000);self.assertFalse(s.accept(r));self.assertIsNone(s.tick(4100000000)['label'])
 def test_model_version_hash_refusal(self):
  with self.assertRaises(ValueError):load_bundle(MODEL,expected_version='low_rate_v001')
  with tempfile.TemporaryDirectory() as td:
   for n in [*PINS,'frozen_manifest.json']:shutil.copyfile(MODEL/n,Path(td)/n)
   verify_bundle(td);(Path(td)/'feature_schema.json').write_text('{}')
   with self.assertRaises(ValueError):load_acquisition(td)
 def test_substride_no_backdated_decision(self):
  e=self.engine();person=[row()['selected_person']]
  e.step(person,200,100,1000000000,seed=0)
  self.assertFalse(e.step(person,200,100,1100000000)[1])
  d=e.step(person,200,100,1800000000)[1];self.assertEqual(len(d),1);self.assertEqual(d[0]['decision_host_ns'],1800000000)
if __name__=='__main__':unittest.main()
