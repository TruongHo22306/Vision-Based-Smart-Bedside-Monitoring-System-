import unittest
from fractions import Fraction as F
from test_pose_features import row
from patient_monitoring.sparse_replay import select_indices,sparse_features,eligible_label
from patient_monitoring.supervised_dataset import make_window
class SparseTests(unittest.TestCase):
 def test_grid_no_duplicates_and_jitter(self):
  times=[F(0),F(26,100),F(49,100),F(51,100),F(76,100),F(101,100)]
  ids=select_indices(times,F(1,2));self.assertEqual(ids,[0,3,5]);self.assertGreater(times[ids[1]]-times[ids[0]],F(1,2))
 def test_recomputes_velocity_over_selected_dt(self):
  rows=[row(i,i*100,shift=i) for i in range(11)]
  f=sparse_features(rows,[0,5,10],'x');self.assertEqual(f[1]['temporal_pair']['delta_seconds'],.5)
  self.assertTrue(f[1]['temporal_pair']['valid']);self.assertEqual(f[1]['source_frame_index'],5)
 def test_hidden_missing_barrier(self):
  rows=[row(i,i*100) for i in range(6)];rows[2]['selected_person']=None
  f=sparse_features(rows,[0,5],'x');self.assertFalse(f[1]['temporal_pair']['valid']);self.assertTrue(f[1]['skipped_source_track_barrier']);self.assertEqual(f[0]['target_status'],'unresolved_source_interval')
 def test_large_gap_rejects_window(self):
  rows=[row(i,i*750) for i in range(4)];f=sparse_features(rows,[0,1,2,3],'x')
  self.assertEqual(make_window(f[:-1],[F(0),F(3,4),F(3,2),F(9,4)],F(0),F(1))[1],'timestamp_gap')
 def test_labels_only_wholly_eligible(self):
  s=[dict(direct=True,lo=1,hi=3,annotation={'activity_label':'moving'})];b=[F(i) for i in range(5)]
  self.assertEqual(eligible_label(F(1),F(3),s,b),'moving');self.assertIsNone(eligible_label(F(0),F(2),s,b));s[0]['direct']=False;self.assertIsNone(eligible_label(F(1),F(2),s,b))
if __name__=='__main__':unittest.main()
