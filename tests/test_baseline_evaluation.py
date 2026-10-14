import copy
import unittest
from patient_monitoring.baseline_evaluation import evaluate_video


def fixture():
    timeline=[dict(frame_index=i,pts=t,time_base='1/10',presentation_time_seconds=t/10,source_duration=d)
              for i,(t,d) in enumerate(((0,1),(1,2),(3,1),(4,2)))]
    predictions=[dict(r,video_id='V',predicted_label=p) for r,p in zip(timeline,('unknown','standing','moving','standing'))]
    annotation=dict(segment_id='V_A001',video_id='V',start_sec=0,end_sec=.6,start_frame_index=0,
        end_frame_index_exclusive=4,annotation_status='reviewed',label_certainty='clear',segment_kind='activity',
        activity_label='standing',reviewer_id='human',reviewed_at_utc='2026-10-15T00:00:00Z',
        annotator_id='human',annotated_at_utc='2026-10-15T00:00:00Z',uncertainty_reason='')
    return timeline,predictions,annotation


class EvaluationTests(unittest.TestCase):
    def test_unknown_startup_in_denominator_and_duration_weights(self):
        t,p,a=fixture();r=evaluate_video('V',t,p,[a])
        self.assertEqual(r['frame_weighted']['overall_accuracy']['value'],.5)
        self.assertAlmostEqual(r['duration_weighted']['overall_accuracy']['value'],2/3)
        self.assertEqual(r['frame_weighted']['prediction_coverage']['value'],.75)
        self.assertEqual(r['frame_weighted']['confusion_matrix']['standing']['unknown'],1)

    def test_half_open_adjacent_intervals(self):
        t,p,a=fixture();a.update(end_sec=.3,end_frame_index_exclusive=2)
        b=dict(a,segment_id='V_A002',start_sec=.3,end_sec=.6,start_frame_index=2,end_frame_index_exclusive=4,activity_label='moving')
        r=evaluate_video('V',t,p,[a,b])
        self.assertEqual(r['frame_weighted']['confusion_matrix']['standing']['moving'],0)
        self.assertEqual(r['frame_weighted']['confusion_matrix']['moving']['moving'],1)

    def test_unconfirmed_no_accuracy(self):
        t,p,a=fixture()
        for status in ('draft','needs_review','unannotated'):
            a['annotation_status']=status
            r=evaluate_video('V',t,p,[a])
            self.assertIsNone(r['frame_weighted']['overall_accuracy'])
            self.assertIsNone(r['duration_weighted']['confusion_matrix'])

    def test_alignment_rejected(self):
        t,p,a=fixture();p[1]['pts']=2;p[1]['presentation_time_seconds']=.2
        with self.assertRaises(ValueError):evaluate_video('V',t,p,[a])

    def test_rounded_boundary_rejected(self):
        t,p,a=fixture();a['end_sec']=.5
        with self.assertRaises(ValueError):evaluate_video('V',t,p,[a])

    def test_overlaps_bounds_and_provenance_rejected(self):
        t,p,a=fixture()
        with self.assertRaises(ValueError):evaluate_video('V',t,p,[a,dict(a,segment_id='V_A002')])
        for changes in ({'end_frame_index_exclusive':5},{'reviewer_id':''}):
            with self.assertRaises(ValueError):evaluate_video('V',t,p,[dict(a,**changes)])

    def test_excluded_unknown_truth_and_missing_annotation(self):
        t,p,a=fixture();a.update(activity_label='unknown',segment_kind='uncertain',end_sec=.3,end_frame_index_exclusive=2,uncertainty_reason='occlusion')
        r=evaluate_video('V',t,p,[a])
        self.assertEqual(r['annotation_coverage']['confirmed_excluded']['frames'],2)
        self.assertEqual(r['annotation_coverage']['unannotated']['frames'],2)
        self.assertIsNone(r['frame_weighted']['overall_accuracy'])

    def test_undefined_metrics_and_timestamp_inconsistency(self):
        t,p,a=fixture();r=evaluate_video('V',t,p,[a])
        self.assertIsNone(r['frame_weighted']['per_class']['lying']['precision']['value'])
        self.assertEqual(r['frame_weighted']['per_class']['standing']['recall']['denominator'],4)
        t[0]['presentation_time_seconds']=.1
        with self.assertRaises(ValueError):evaluate_video('V',t,p,[a])


if __name__=='__main__':unittest.main()
