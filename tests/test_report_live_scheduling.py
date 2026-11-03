import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).resolve().parents[1]/'tools/report_live_scheduling.py'
s=importlib.util.spec_from_file_location('report_live_scheduling',p);report=importlib.util.module_from_spec(s);s.loader.exec_module(report)
class ReportTests(unittest.TestCase):
    def test_timing_and_deduplicated_received_results(self):
        f=dict(type='frame',target_epoch=1,capture_sequence=5,host_acquired_ns=0,process_started_host_ns=100000000,predict_seconds=.4,classifier_started_host_ns=500000000,classifier_completed_host_ns=600000000,completed_host_ns=600000000)
        u=dict(type='ui_update',target_epoch=1,frame_id=5,completed_host_ns=600000000,published_host_ns=700000000,ui_received_host_ns=900000000,assessed_host_ns=950000000)
        r=report.analyze(dict(processed=1,log_omitted_records=0),[f,u,{**u,'assessed_host_ns':1000000000}])
        self.assertEqual(r['received_unique_pose_results'],1)
        for name,value in [('frame_age_at_worker_start',.1),('predict',.4),('classifier',.1),('completion_to_publication',.1),('publication_to_ui_receipt',.2),('receipt_to_ui_assessment',.05)]:
            self.assertAlmostEqual(r['timings'][name]['mean_seconds'],value)
    def test_missing_milestones_not_invented(self):
        r=report.analyze(dict(processed=2,log_omitted_records=3),[dict(type='frame',capture_sequence=0,predict_seconds=.5)])
        self.assertIsNone(r['timings']['frame_age_at_worker_start']['mean_seconds']);self.assertEqual(r['timings']['classifier']['missing'],1);self.assertEqual(r['log_coverage'],'partial')
if __name__=='__main__':unittest.main()
