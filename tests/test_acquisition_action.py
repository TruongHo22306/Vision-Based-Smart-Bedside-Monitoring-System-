import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from patient_monitoring.acquisition_action import vectorize,validate_schema,metrics,load_bundle
from patient_monitoring.acquisition_features import VERSION,ORDER
from patient_monitoring.low_rate_experiment import fit,apply
from patient_monitoring.supervised_dataset import write,sha,CLASSES
class ActionTests(unittest.TestCase):
 def schema(self):return dict(version=VERSION,dimensions=35,order=ORDER)
 def row(self):return dict(version=VERSION,available=True,vector=[1.]*14+[1.]*7+[0.]*14)
 def test_vector_order_and_metadata_excluded(self):
  r=self.row();a=vectorize([r],self.schema());r.update(video_id='test',label='lying',occasion='x',timestamp=999,baseline='moving');np.testing.assert_array_equal(a,vectorize([r],self.schema()))
  self.assertEqual(a.shape,(1,35))
 def test_wrong_version_order_flags(self):
  s=self.schema();s['version']='low_rate_3s_v001'
  with self.assertRaises(ValueError):vectorize([self.row()],s)
  s=self.schema();s['order']=list(reversed(ORDER))
  with self.assertRaises(ValueError):vectorize([self.row()],s)
  r=self.row();r['vector'][0]=None
  with self.assertRaises(ValueError):vectorize([r],self.schema())
  r['vector'][21]=1.;self.assertTrue(np.isnan(vectorize([r],self.schema())[0,0]))
 def test_training_only_imputation_constant_missing_flags(self):
  a=vectorize([self.row(),self.row()],self.schema());a[:,0]=np.nan;a[:,21]=1.;p=fit(a)
  self.assertTrue(p['fallback'][0]);self.assertEqual(p['medians'][0],0);self.assertEqual(p['scales'][1],1)
  before=copy.deepcopy(p);other=a.copy();other[:,1]=1000
  b=apply(other,p);self.assertEqual(p,before);np.testing.assert_array_equal(b[:,14:],other[:,14:])
 def test_unknown_metrics_denominators(self):
  r=metrics([dict(truth='standing',predicted_label=None),dict(truth='standing',predicted_label='moving')],'truth')
  self.assertEqual(r['accuracy_including_unavailable'],0);self.assertEqual(r['unavailable'],1);self.assertEqual(r['per_class']['standing']['actual'],2);self.assertIsNone(r['per_class']['moving']['recall'])
 def test_persistence_version_and_tamper_rejection(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);schema=self.schema();a=vectorize([self.row()],schema);prep=fit(a);prep.update(feature_version=VERSION,classes=list(CLASSES))
   model=torch.nn.Linear(35,4,dtype=torch.float64);torch.save(model.state_dict(),p/'model_state.pt')
   write(p/'feature_schema.json',schema);write(p/'preprocessing.json',prep);write(p/'experiment_config.json',dict(classes=list(CLASSES)))
   write(p/'frozen_manifest.json',dict(feature_version=VERSION,artifact_sha256={n:sha(p/n) for n in ['model_state.pt','feature_schema.json','preprocessing.json','experiment_config.json']}))
   other,_,_=load_bundle(p);torch.testing.assert_close(model(torch.from_numpy(a)),other(torch.from_numpy(a)),rtol=0,atol=0)
   with self.assertRaises(ValueError):load_bundle(p,expected_version='wrong')
   (p/'model_state.pt').write_bytes(b'corrupt')
   with self.assertRaises(ValueError):load_bundle(p)
if __name__=='__main__':unittest.main()
