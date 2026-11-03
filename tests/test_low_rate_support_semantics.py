import copy,json,unittest
from fractions import Fraction as F
from pathlib import Path
from test_pose_features import row
from patient_monitoring.low_rate_experiment import recompute,window
ROOT=Path(__file__).resolve().parents[1]
class SupportSemanticsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.config=json.loads((ROOT/'outputs/low_rate_development_v001_20261030/config.json').read_text())
 def data(self):
  rows=[row(i,i*1000,shift=i*i) for i in range(4)]
  return recompute(rows,list(range(4)),'synthetic',1.5),list(map(F,range(4)))
 def test_current_geometry_does_not_change_frozen_window(self):
  f,t=self.data();a=window(f,t,F(3),self.config)[0];self.assertIsNotNone(a)
  f[-1]['features']['box_size_image']['value']=[.99,.99]
  self.assertEqual(a,window(f,t,F(3),self.config)[0])
  f[-2]['features']['box_size_image']['value']=[.98,.98]
  self.assertNotEqual(a['vector'],window(f,t,F(3),self.config)[0]['vector'])
 def test_current_backward_velocity_valid_but_excluded(self):
  f,t=self.data();self.assertTrue(f[-1]['features']['hip_image_velocity_per_second']['valid'])
  a=window(f,t,F(3),self.config)[0]
  self.assertEqual(a['channels']['hip_vx_per_s']['observations'],2)
  self.assertFalse(a['channels']['hip_vx_per_s']['std_valid'])
  f[-1]['features']['hip_image_velocity_per_second']['value']=[999.,999.]
  self.assertEqual(a,window(f,t,F(3),self.config)[0])
  self.assertEqual(a['latest_age'],1.)
 def test_irregular_latest_point_after_grid_is_not_backdated(self):
  pts=[0,700,1600,2500,3200];rows=[row(i,p,shift=i) for i,p in enumerate(pts)]
  f=recompute(rows,list(range(5)),'synthetic',1.5);t=[F(p,1000) for p in pts]
  a=window(f,t,F(3),self.config)[0];self.assertIsNotNone(a);self.assertEqual(a['latest_age'],.5)
  self.assertNotIn(4,a['source_indices'])
if __name__=='__main__':unittest.main()
