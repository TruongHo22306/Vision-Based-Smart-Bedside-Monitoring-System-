import copy
import math
import unittest
from dataclasses import replace
from patient_monitoring.action_baseline_v3 import BaselineV3,LocomotionConfig
from patient_monitoring.action_baseline_v2 import BaselineV2
from test_action_baseline import frame


def sample(i,t,transport=0,depth=0,gait=0,**kwargs):
    r=frame(i,t,**kwargs)
    scale=100*math.exp(depth*t)
    def f(value):return dict(value=value,valid=True,reason=None)
    r['features'].update(hip_midpoint_px=f([200+transport*100*t,300]),
                         shoulder_midpoint_px=f([200+transport*100*t,300-scale]),torso_scale_px=f(scale))
    r['joints']=[dict(index=idx,body_xy=f([(-1 if idx==15 else 1)*gait*math.sin(t*6),2]),
                      source_quality={'passes_quality_gate':True}) for idx in (15,16)]
    return r


class LocomotionTests(unittest.TestCase):
    def final(self,**kw):
        e=BaselineV3()
        return [e.step(sample(i,i/10,**kw)) for i in range(21)][-1]

    def test_box_only_not_walking(self):
        self.assertEqual(self.final(box=.2)['predicted_label'],'standing')

    def test_transport_without_leg_change_is_not_walking(self):
        self.assertEqual(self.final(transport=.3,hip=.1)['predicted_label'],'unknown')

    def test_legs_without_transport_not_walking(self):
        self.assertEqual(self.final(gait=.2,box=.2)['predicted_label'],'unknown')

    def test_lateral_and_depth_candidates(self):
        for settings in ({'transport':.3},{'depth':.1},{'depth':-.1}):
            self.assertEqual(self.final(gait=.2,**settings)['predicted_label'],'moving')

    def test_missing_joint_resets_and_no_bridging(self):
        e=BaselineV3()
        for i in range(11):r=e.step(sample(i,i/10,transport=.3,gait=.2))
        row=sample(11,1.1,transport=.3,gait=.2);row['joints'][0]['source_quality']['passes_quality_gate']=False
        self.assertEqual(e.step(row)['predicted_label'],'unknown')
        for i in range(12,20):self.assertFalse(e.step(sample(i,i/10,transport=.3,gait=.2))['landmark_motion']['valid'])
        self.assertTrue(e.step(sample(20,2,transport=.3,gait=.2))['landmark_motion']['valid'])

    def test_breaks_gaps_and_unresolved(self):
        for mutation in ({'track_break':True},{'target_status':'seeded'}, {'target_status':'unresolved','selected_source_detection_index':None},
                         {'pts':30000,'presentation_time_seconds':3.0}, {'frame_index':50}, {'track_id':2}):
            e=BaselineV3()
            for i in range(11):e.step(sample(i,i/10,transport=.3,gait=.2))
            r=sample(11,1.1,transport=.3,gait=.2);r.update(mutation)
            result=e.step(r);self.assertFalse(result['landmark_motion']['valid']);self.assertEqual(result['predicted_label'],'unknown')

    def test_causality_irregular_time_and_identity_not_input(self):
        times=[0,.02,.1,.18,.3,.43,.55,.7,.8,.93,1.03]
        a,b=BaselineV3(),BaselineV3()
        for i,t in enumerate(times):
            r=sample(i,t,transport=.3,gait=.2);r2=copy.deepcopy(r);r2.update(video_id='lying',source_path='lying.mp4')
            x,y=a.step(r),b.step(r2)
            self.assertEqual(x['landmark_motion'],y['landmark_motion']);self.assertEqual(x['predicted_label'],y['predicted_label'])
            prefix=BaselineV3()
            for j in range(i+1):expected=prefix.step(sample(j,times[j],transport=.3,gait=.2))
            self.assertEqual(x,expected)
            self.assertEqual(x['landmark_motion']['valid'],t>=.8)

    def test_non_upright_exact_v002_behavior(self):
        for kw in ({'knee':100},{'torso':90,'aspect':2}):
            a,b=BaselineV2(),BaselineV3()
            for i in range(21):
                r=sample(i,i/10,box=.2,**kw);r['joints']=[]
                x,y=a.step(r),b.step(r)
                self.assertEqual(x['predicted_label'],y['predicted_label']);self.assertEqual(x['reasons'],y['reasons'])

    def test_scale_normalization(self):
        a,b=BaselineV3(),BaselineV3()
        for i in range(21):
            r=sample(i,i/10,transport=.3,gait=.2);s=copy.deepcopy(r)
            for key in ('hip_midpoint_px','shoulder_midpoint_px'):
                s['features'][key]['value']=[2*v for v in s['features'][key]['value']]
            s['features']['torso_scale_px']['value']*=2
            self.assertEqual(a.step(r)['landmark_motion'],b.step(s)['landmark_motion'])

    def test_invalid_config(self):
        for kw in ({'body_speed':0},{'minimum_span_seconds':2},{'ankle_range_quiet':.3}):
            with self.assertRaises(ValueError):LocomotionConfig(**kw)


if __name__=='__main__':unittest.main()
