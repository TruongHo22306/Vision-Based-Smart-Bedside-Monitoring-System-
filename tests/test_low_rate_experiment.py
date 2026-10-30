import json,unittest
from pathlib import Path
from fractions import Fraction as F
from unittest.mock import patch
import numpy as np
from test_pose_features import row
from patient_monitoring.low_rate_experiment import window,recompute,select,fit,apply,CHANNELS
C=json.loads(Path('outputs/low_rate_experiment_preparation_v001_20261030/config.json').read_text())
class LowRateExperimentTests(unittest.TestCase):
 def test_real_dt_features_and_no_bridge(self):
  rows=[row(i,i*1000,shift=i) for i in range(5)];f=recompute(rows,list(range(5)),'test',1.5)
  self.assertEqual(f[1]['temporal_pair']['delta_seconds'],1);self.assertTrue(f[1]['temporal_pair']['valid'])
  rows[2]['selected_person']=None;g=recompute(rows,[0,1,3,4],'test',1.5);self.assertEqual(g[1]['target_status'],'unresolved_source_interval')
 def test_std_requires_three_valid_differences(self):
  rows=[row(i,i*1000,shift=i) for i in range(5)];f=recompute(rows,list(range(5)),'test',1.5)
  w,reason=window(f,[F(i) for i in range(5)],F(3),C);self.assertIsNone(reason)
  v=w['channels']['hip_vx_per_s'];self.assertTrue(v['mean_valid']);self.assertIsNone(v['std']);self.assertFalse(v['std_valid']);self.assertEqual(v['observations'],2)
 def test_startup_gap_and_sparse_observations(self):
  rows=[row(i,i*1600) for i in range(4)];f=recompute(rows,list(range(4)),'test',1.5)
  self.assertEqual(window(f,[F(8*i,5) for i in range(4)],F(2),C)[1],'startup')
  self.assertIsNotNone(window(f,[F(8*i,5) for i in range(4)],F(3),C)[1])
 def test_training_only_missing_and_constant(self):
  x=np.zeros((2,35));x[:,0]=np.nan;x[:,1]=[2,4];p=fit(x);self.assertTrue(p['fallback'][0]);self.assertEqual(p['medians'][1],3)
  held=x.copy();held[:,1]=100;before=json.dumps(p);y=apply(held,p);self.assertTrue(np.isfinite(y).all());self.assertEqual(json.dumps(p),before)
 def test_irregular_selection_unique_no_future(self):
  times=[F(i,10) for i in range(60)];ids=select(times,[.5,.75,1,1.25]);self.assertEqual(len(ids),len(set(ids)));self.assertEqual(ids[:4],[0,5,13,23])
if __name__=='__main__':unittest.main()
