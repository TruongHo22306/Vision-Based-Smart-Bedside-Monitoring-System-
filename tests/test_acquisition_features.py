import copy,json,unittest
from fractions import Fraction as F
from pathlib import Path
from test_pose_features import row
from patient_monitoring.acquisition_features import AcquisitionFeatures,aggregate,freshness,CONFIG
class AcquisitionTests(unittest.TestCase):
 def ready(self,times=(0,1000,2000,3000)):
  e=AcquisitionFeatures('test');rs=[]
  for i,t in enumerate(times):
   f,r=e.step(row(i,t,shift=i*i));rs.append(r)
  return e,rs[-1]
 def test_current_geometry_and_backward_interval(self):
  e,r=self.ready();self.assertTrue(r['available']);self.assertEqual(r['observation_count'],3)
  self.assertEqual(r['channels']['hip_vx_per_s']['observations'],3);self.assertTrue(r['channels']['hip_vx_per_s']['std_valid'])
  self.assertEqual(r['channels']['hip_vx_per_s']['coverage'],1)
  f=copy.deepcopy(e.history);f[-1]['features']['box_size_image']['value']=[.99,.99]
  self.assertNotEqual(r['vector'],aggregate(f,F(3))['vector']);self.assertEqual(r['channels']['box_width_fraction']['support_age_seconds'],0)
 def test_left_open_geometry_right_closed_velocity(self):
  e,r=self.ready();f=copy.deepcopy(e.history);f[0]['features']['box_size_image']['value']=[999,999]
  self.assertEqual(r['vector'],aggregate(f,F(3))['vector']);self.assertEqual(r['channels']['box_width_fraction']['observations'],3)
 def test_no_future_and_parity(self):
  e=AcquisitionFeatures('test');prefix=[]
  for i,t in enumerate([0,500,1250,2250,3500,4000,4750,5750]):
   f,r=e.step(row(i,t,shift=i));prefix.append(f)
   if r and r['temporal_available']:
    a=aggregate(prefix,F(t,1000));self.assertEqual(a['vector'],r['vector']);self.assertEqual(a['channels'],r['channels'])
  with self.assertRaises(ValueError):aggregate(prefix,F(4))
 def test_missing_current_does_not_refresh_channel(self):
  e,r=self.ready();f=copy.deepcopy(e.history);f[-1]['features']['box_size_image']['valid']=False
  r=aggregate(f,F(3));c=r['channels']['box_width_fraction'];self.assertEqual(c['support_age_seconds'],1);self.assertEqual(c['coverage'],2/3)
  self.assertTrue(c['mean_valid']);self.assertFalse(c['std_valid']);self.assertEqual(r['expires_at'],'13/4')
  self.assertTrue(freshness(r,F(13,4))[0]);self.assertFalse(freshness(r,F(326,100))[0])
 def test_stale_channel_is_missing_not_zero(self):
  e,r=self.ready();f=copy.deepcopy(e.history)
  for x in f[-2:]:x['features']['box_size_image']['valid']=False
  r=aggregate(f,F(3));self.assertIsNone(r['channels']['box_width_fraction']['mean']);self.assertEqual(r['channels']['box_width_fraction']['reason'],'stale_support')
 def test_missing_gap_reset_and_hidden_loss(self):
  for kind in ['missing','gap','reset','hidden']:
   e,r=self.ready()
   if kind=='reset':e.reset()
   q=row(4,5000 if kind=='gap' else 4000)
   if kind=='missing':q['selected_person']=None;q['status']='no_detections'
   _,r=e.step(q,hidden_barrier=kind=='hidden');self.assertFalse(r['available'])
   _,r=e.step(row(5,6000 if kind=='gap' else 5000));self.assertFalse(r['available'])
 def test_weights_units_and_no_future_duration(self):
  e,r=self.ready((0,500,1500,3000));f=copy.deepcopy(e.history)
  for i,(geo,vel) in enumerate(zip([.1,.4,.7],[1.,2.,4.]),1):
   f[i]['features']['box_size_image']['value']=[geo,geo]
   f[i]['features']['hip_image_velocity_per_second']['value']=[vel,vel]
  r=aggregate(f,F(3));g=r['channels']['box_width_fraction'];v=r['channels']['hip_vx_per_s']
  self.assertAlmostEqual(g['mean'],.4);self.assertAlmostEqual(g['std'],(.06)**.5)
  self.assertEqual(g['weight_sum'],3);self.assertEqual(g['weight_unit'],'observations')
  self.assertAlmostEqual(v['mean'],(1*.5+2*1+4*1.5)/3);self.assertEqual(v['weight_sum'],3)
  self.assertEqual(v['weight_unit'],'seconds');self.assertEqual(v['latest_valid_timestamp'],'3')
 def test_invalid_timestamp(self):
  e,r=self.ready()
  with self.assertRaises(ValueError):e.step(row(4,3000))
 def test_no_one_difference_variability(self):
  e,r=self.ready();f=copy.deepcopy(e.history)
  for x in f[1:3]:x['features']['hip_image_velocity_per_second']['valid']=False
  r=aggregate(f,F(3));self.assertIsNone(r['channels']['hip_vx_per_s']['std']);self.assertFalse(r['channels']['hip_vx_per_s']['std_valid'])
if __name__=='__main__':unittest.main()
