import unittest
from unittest.mock import patch
import test_live_low_rate_classifier as fixtures
from test_pose_features import row
from patient_monitoring.live_demo import DisplaySummary
from patient_monitoring.live_low_rate import LowRateEngine

class DisplayTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.LowRateLiveTests.setUpClass();cls.bundle=fixtures.LowRateLiveTests.bundle
 def ready(self):
  e=LowRateEngine(*self.bundle,experimental_association_gap=1.5)
  for i in range(5):
   with patch('patient_monitoring.live_low_rate.time.monotonic_ns',side_effect=iter(range(10000000000,10000000100))):
    _,ds,_=e.step([row()['selected_person']],200,100,i*750000000,seed=0 if i==0 else None)
  return e,ds
 def test_latest_grid_and_true_completion(self):
  e,ds=self.ready();last=ds[-1]
  self.assertTrue(last['available']);self.assertEqual(e.display_decision_host_ns,last['decision_host_ns']);self.assertEqual(e.display_support_host_ns,2250000000)
  for d in ds:self.assertGreater(d['computed_host_ns'],d['compute_started_host_ns'])
 def test_expiry_boundary_does_not_revive_history(self):
  e,ds=self.ready();stats=DisplaySummary()
  self.assertIsNotNone(stats.observe(e,3500000000,True)['label'])
  r=stats.observe(e,3500000001,True);self.assertIsNone(r['label']);self.assertEqual(r['reason'],'display_support_expired')
  self.assertTrue(any(d['available'] for d in ds));self.assertEqual(stats.summary()['frames_with_label_submitted_to_ui'],1)
  self.assertIsNone(stats.observe(e,3600000000,True)['label'])
 def test_post_telemetry_expiry_and_headless(self):
  e,_=self.ready();stats=DisplaySummary()
  self.assertIsNotNone(e.label)
  self.assertEqual(stats.observe(e,3400000000,False)['non_display_reason'],'ui_disabled')
  self.assertEqual(stats.observe(e,3600000000,True)['non_display_reason'],'display_support_expired')
  self.assertEqual(stats.summary()['frames_with_label_submitted_to_ui'],0)
 def test_reset_clears_decision_provenance(self):
  e,_=self.ready();e.reset();r=DisplaySummary().observe(e,4000000000,True)
  self.assertIsNone(r['decision_host_ns']);self.assertIsNone(r['computed_host_ns']);self.assertIsNone(r['label'])
if __name__=='__main__':unittest.main()
