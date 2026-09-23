"""Synthetic association checks; not pose/identity ground truth."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from patient_monitoring.target_selection import TargetAssociator, select_target, valid_box
from patient_monitoring.pose_preview import KEYPOINT_NAMES
from patient_monitoring.video_inspection import InspectionError, TimelineAudit


def person(index, box=(10., 10., 30., 30.)):
    return dict(detection_index=index, box_xyxy=list(box), box_confidence=.1,
        keypoints=[dict(index=i, name=KEYPOINT_NAMES[i], x=0., y=100., confidence=.95,
            finite_values=True, finite_components=dict(x=True,y=True,confidence=True), confidence_pass=True,
            at_or_outside_boundary=True, passes_quality_gate=False) for i in range(17)])


def records(times, people):
    audit = TimelineAudit()
    return [dict(**audit.observe(t, Fraction(1,100)), source_path='synthetic.mp4', stream_index=0,
                 width=100, height=100, inference_status='complete', person_count=len(p), people=p)
            for t,p in zip(times,people)]


class AssociationTests(unittest.TestCase):
    def run_rows(self, rows, **kwargs):
        engine = TargetAssociator(0, 0, **kwargs)
        return [engine.step(r) for r in rows]

    def test_reordered_indices_select_geometry_not_index_confidence_or_quality(self):
        other = person(0, (60.,60.,90.,90.))
        other['box_confidence'] = .99
        rows = records([0,10,20], [[person(0)], [other,person(1)], [person(7),person(2,(60.,60.,90.,90.))]])
        before = deepcopy(rows)
        result = self.run_rows(rows)
        self.assertEqual([r['selected_source_detection_index'] for r in result], [0,1,7])
        self.assertEqual(result[1]['selected_person'], rows[1]['people'][1])
        self.assertTrue(all(not p['passes_quality_gate'] for p in result[1]['selected_person']['keypoints']))
        self.assertEqual(rows,before)

    def test_missing_does_not_update_reference_and_resumes_within_gap(self):
        result = self.run_rows(records([0,10,40], [[person(0)],[],[person(9)]]))
        self.assertEqual([r['status'] for r in result], ['seeded','no_detections','matched'])
        self.assertIsNone(result[1]['selected_person'])
        self.assertEqual(result[2]['reference_frame_index'],0)
        self.assertEqual(result[2]['elapsed_since_reference_seconds'],.4)

    def test_ambiguity_and_low_iou_do_not_move_reference(self):
        rows = records([0,10,20,30], [[person(0)], [person(5),person(6)],
            [person(1,(60.,60.,90.,90.))],[person(2)]])
        result=self.run_rows(rows)
        self.assertEqual([r['status'] for r in result], ['seeded','ambiguous','low_iou','matched'])
        self.assertEqual(result[-1]['reference_frame_index'],0)
        self.assertEqual(result[1]['iou_separation'],0.)
        self.assertEqual(self.run_rows(rows,ambiguity_margin=0)[1]['status'],'ambiguous')

    def test_timeout_is_elapsed_time_not_frame_count_and_never_reacquires(self):
        result=self.run_rows(records([0,51,52], [[person(0)],[person(8)],[person(9)]]))
        self.assertEqual([r['status'] for r in result], ['seeded','gap_timeout','association_ended'])
        self.assertTrue(all(r['selected_person'] is None for r in result[1:]))
        equal=self.run_rows(records([0,50], [[person(0)],[person(3)]]))
        self.assertEqual(equal[1]['status'],'matched')

    def test_margin_threshold_and_single_candidate(self):
        rows=records([0,10], [[person(0)], [person(4),person(8,(11.,10.,31.,30.))]])
        self.assertEqual(self.run_rows(rows,ambiguity_margin=.1)[1]['status'],'ambiguous')
        self.assertEqual(self.run_rows(rows,ambiguity_margin=.09)[1]['status'],'matched')
        single=self.run_rows(records([0,10], [[person(0)],[person(8)]]),ambiguity_margin=1)[1]
        self.assertEqual(single['status'],'matched')
        self.assertIsNone(single['runner_up_iou'])

    def test_invalid_boxes_rejected_and_seed_not_automatically_replaced(self):
        for box in ([0,0,0,10],[20,0,10,20],[-1,0,20,20],[0,0,101,20],[None,0,20,20],[float('nan'),0,20,20]):
            with self.subTest(box=box):
                self.assertIsNotNone(valid_box(box,100,100))
                result=self.run_rows(records([0,10,20], [[person(0)],[person(1,box)],[person(2)]]))
                self.assertEqual(result[1]['status'],'invalid_boxes')
                self.assertEqual(result[2]['reference_frame_index'],0)
        result=self.run_rows(records([0,10], [[],[person(0)]]))
        self.assertEqual([r['status'] for r in result],['seed_rejected','association_ended'])

    def test_bad_timestamps_reject_and_do_not_extend_gap(self):
        for times in ([0,None,60],[0,0,60],[0,-1,60]):
            with self.subTest(times=times):
                result=self.run_rows(records(times,[[person(0)],[person(0)],[person(0)]]))
                self.assertEqual(result[1]['status'],'invalid_timestamp')
                self.assertEqual(result[2]['status'],'gap_timeout')
        rows=records([0,10],[[person(0)],[person(0)]])
        rows[1]['presentation_time_seconds']=.11
        self.assertEqual(self.run_rows(rows)[1]['status'],'invalid_timestamp')

    def test_explicit_later_seed_and_duplicate_detection_ids(self):
        engine=TargetAssociator(1,4)
        rows=records([0,10,20],[[person(0)],[person(4)],[person(4),person(4)]])
        result=[engine.step(r) for r in rows]
        self.assertEqual([r['status'] for r in result],['awaiting_seed','seeded','invalid_detections'])

    def test_parameter_validation(self):
        for kwargs in (dict(min_iou=-.1),dict(ambiguity_margin=2),dict(max_gap=0),dict(max_gap=float('inf'))):
            with self.assertRaises(InspectionError):
                TargetAssociator(0,0,**kwargs)


class SelectionFileTests(unittest.TestCase):
    def test_rows_timestamps_raw_joints_gap_ranges_and_preservation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            poses=root/'poses.jsonl'
            rows=records([0,10,30,90,100],[[person(0)],[],[person(4)],[person(1)],[person(2)]])
            original=''.join(json.dumps(r)+'\n' for r in rows)
            poses.write_text(original,encoding='utf-8')
            with patch.dict('sys.modules',{'ultralytics':None}):
                summary=select_target(poses,root/'out',0,0)
            actual=[json.loads(line) for line in (root/'out/target_poses.jsonl').read_text().splitlines()]
            self.assertTrue(summary['completed'])
            self.assertEqual(summary['written_frame_count'],5)
            self.assertEqual(summary['selected_frame_count'],2)
            self.assertEqual(len(summary['gap_ranges']),2)
            self.assertEqual(summary['gap_ranges'][0]['resumed_at_frame_index'],2)
            self.assertIsNone(summary['gap_ranges'][1]['resumed_at_frame_index'])
            for source,out in zip(rows,actual):
                for key in ('frame_index','pts','time_base','presentation_time_seconds','relative_time_seconds','delta_seconds'):
                    self.assertEqual(source[key],out[key])
                if out['selected_person'] is not None:
                    self.assertEqual(out['selected_person'],source['people'][0])
            self.assertEqual(poses.read_text(encoding='utf-8'),original)
            with self.assertRaises(InspectionError):
                select_target(poses,root/'out',0,0)

    def test_nonexistent_seed_reports_incomplete_but_keeps_rows(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            poses=root/'poses.jsonl'
            poses.write_text(json.dumps(records([0],[[person(0)]])[0])+'\n')
            summary=select_target(poses,root/'out',8,0)
            self.assertFalse(summary['completed'])
            self.assertEqual(summary['written_frame_count'],1)


if __name__ == '__main__':
    unittest.main()
