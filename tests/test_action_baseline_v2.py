import copy
import json
from pathlib import Path
import tempfile
import unittest

from patient_monitoring.action_baseline import Baseline, FEATURES, Config
from patient_monitoring.action_baseline_v2 import BaselineV2, run_baseline
from test_action_baseline import frame, sequence


class CorrectionTests(unittest.TestCase):
    def test_non_upright_posture_survives_motion_with_visible_cautions(self):
        for geometry,label in [({'knee':100},'sitting'),({'torso':90,'aspect':2},'lying')]:
            for speed,reason in [(.02,'motion_threshold_band'),(.06,'motion_posture_conflict')]:
                a,b=Baseline(),BaselineV2()
                for row in sequence(**geometry,hip=speed):
                    old,new=a.step(row),b.step(row)
                self.assertEqual(old['predicted_label'],'unknown')
                self.assertEqual(new['predicted_label'],label)
                self.assertEqual(new['image_motion_cautions'],[reason])
                self.assertEqual(new['locomotion_evidence']['status'],'unknown')
                self.assertEqual(new['v001_decision']['predicted_label'],'unknown')
                self.assertEqual(new['motion_evidence'],old['motion_evidence'])

    def test_other_branches_and_all_support_unchanged(self):
        for options in ({},{'hip':.06},{'hip':.02},{'torso':45,'hip':.06},{'knee':145},{'knee':100}):
            a,b=Baseline(),BaselineV2()
            for row in sequence(**options):
                x,y=a.step(row),b.step(row)
                self.assertEqual(x['predicted_label'],y['predicted_label'])
                self.assertEqual(x['support'],y['support'])
                self.assertEqual(x['window'],y['window'])

    def test_missing_motion_or_geometry_remains_unknown(self):
        for name in FEATURES:
            rows=sequence(torso=90,aspect=2,hip=.06)
            rows[-1]['features'][name]={'value':None,'valid':False,'reason':'missing'}
            b=BaselineV2()
            out=[b.step(r) for r in rows][-1]
            self.assertEqual(out['predicted_label'],'unknown')
            self.assertIn('insufficient:'+name,out['reasons'])
            self.assertEqual(out['image_motion_cautions'],[])

    def test_unresolved_track_break_gap_reset_no_label_hold(self):
        for mutation in ({'target_status':'unresolved','selected_source_detection_index':None},
                         {'track_break':True},{'target_status':'seeded'},
                         {'pts':35000,'presentation_time_seconds':3.5}):
            b=BaselineV2()
            for r in sequence(torso=90,aspect=2,hip=.06):last=b.step(r)
            self.assertEqual(last['predicted_label'],'lying')
            row=frame(21,2.1,torso=90,aspect=2,hip=.06);row.update(mutation)
            out=b.step(row)
            self.assertEqual(out['predicted_label'],'unknown')
            self.assertIsNotNone(out['window']['reset_reason'])

    def test_causal_prefix_no_additional_delay(self):
        rows=sequence(torso=90,aspect=2,hip=.06)
        a=BaselineV2();full=[a.step(r) for r in rows]
        b=BaselineV2();prefix=[b.step(r) for r in rows[:12]]
        self.assertEqual(full[:12],prefix)
        self.assertTrue(all(r['predicted_label']=='unknown' for r in full[:8]))
        self.assertEqual(full[8]['predicted_label'],'lying')
        self.assertEqual(full[8]['presentation_time_seconds'],.8)

    def test_coverage_at_15_and_30_hz_and_irregular_sampling(self):
        for denominator,ticks in [(15,list(range(31))),(30,list(range(61))),
                                  (30,[0,1,3,5,8,10,13,16,20,24,27,30,34,39,44,49,54,60])]:
            b=BaselineV2(); results=[]
            for i,tick in enumerate(ticks):
                r=frame(i,tick/denominator,torso=90,aspect=2,hip=.06)
                r.update(pts=tick,time_base=f'1/{denominator}')
                results.append(b.step(r))
            first=next(x for x in results if x['predicted_label']!='unknown')
            self.assertAlmostEqual(first['presentation_time_seconds'],.8)
            for f in FEATURES:
                self.assertAlmostEqual(first['support'][f]['covered_seconds'],.8)
                self.assertAlmostEqual(results[-1]['support'][f]['covered_seconds'],1)
            self.assertAlmostEqual(results[-1]['support'][FEATURES[4]]['value'],.06)

    def test_no_filename_or_video_identity_rules(self):
        a,b=BaselineV2(),BaselineV2()
        for r in sequence(torso=90,aspect=2,hip=.06):
            changed=copy.deepcopy(r);changed.update(video_id='standing',source_path='moving.mp4')
            self.assertEqual(a.step(r)['predicted_label'],b.step(changed)['predicted_label'])

    def test_new_output_version_and_no_overwrite_and_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            src,out=Path(tmp)/'in.jsonl',Path(tmp)/'run'
            src.write_text(''.join(json.dumps(r)+'\n' for r in sequence(torso=90,aspect=2,hip=.06)))
            summary=run_baseline(src,out)
            self.assertTrue(summary['completed'])
            self.assertEqual(summary['label_counts']['unknown'],8)
            self.assertEqual(summary['image_motion_caution_counts']['motion_posture_conflict'],13)
            p=json.loads((out/'provenance.json').read_text())
            self.assertEqual(p['config']['additional_temporal_delay_seconds'],0)
            self.assertEqual(p['config']['thresholds_and_windows'],Config().__dict__)
            with self.assertRaises(ValueError):run_baseline(src,out)
            src.write_text(json.dumps(frame(0,0))+'\nBAD\n')
            bad=run_baseline(src,Path(tmp)/'partial')
            self.assertFalse(bad['completed']);self.assertEqual(bad['frame_count'],1)


if __name__=='__main__':unittest.main()
