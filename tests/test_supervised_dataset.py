import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
from patient_monitoring.supervised_dataset import aligned_frames, make_window, segments, build, CHANNELS


def fixture(times=(0, .2, .45, .7, .9, 1.1)):
    rows=[]
    for i,t in enumerate(times):
        features={f:dict(valid=True,reason=None,value=90.0 if c is None else [1.,2.])
                  for f,c in CHANNELS.values()}
        rows.append(dict(frame_index=i,pts=round(t*1000),time_base='1/1000',
            presentation_time_seconds=t,relative_time_seconds=t,timestamp_valid=True,
            target_status='matched' if i else 'seeded',
            temporal_pair=dict(valid=i>0,previous_frame_index=i-1),features=features))
    return rows,[F(str(t)) for t in times]+[F('1.3')]


def annotation(lo,hi,bounds,**extra):
    return dict(start_frame_index=str(lo),end_frame_index_exclusive=str(hi),
        start_sec=str(float(bounds[lo])),end_sec=str(float(bounds[hi])),
        annotation_status='reviewed',label_certainty='clear',segment_kind='activity',
        activity_label='standing',annotator_id='human',reviewer_id='human',
        annotated_at_utc='date',reviewed_at_utc='date',**extra)


class SupervisedDatasetTests(unittest.TestCase):
    def test_irregular_duration_weighting(self):
        rows,b=fixture()
        value,reason=make_window(rows,b,F(0),F(1))
        self.assertIsNone(reason)
        self.assertEqual(value['frame_indices'],[0,1,2,3,4])
        self.assertAlmostEqual(value['model_inputs']['left_knee_deg']['coverage'],1)
        self.assertAlmostEqual(value['model_inputs']['hip_vx_per_s']['coverage'],.9)

    def test_half_open_no_future_frame(self):
        r,b=fixture((0,.25,.5,.75,1,1.1))
        v,_=make_window(r,b,F(0),F(1))
        self.assertNotIn(4,v['frame_indices'])

    def test_derivative_not_cross_start(self):
        r,b=fixture()
        v,_=make_window(r,b,F('.3'),F('1.3'))
        self.assertAlmostEqual(v['model_inputs']['hip_vx_per_s']['coverage'],.65)

    def test_missing_preserved_without_discarding_other_channels(self):
        r,b=fixture()
        for row in r:
            row['features']['left_knee_angle_deg']=dict(valid=False,value=None,reason='missing_joint')
        v,_=make_window(r,b,F(0),F(1))
        self.assertIsNone(v['model_inputs']['left_knee_deg']['mean'])
        self.assertEqual(v['model_inputs']['left_knee_deg']['coverage'],0)
        self.assertAlmostEqual(v['model_inputs']['right_knee_deg']['mean'],90)

    def test_unresolved_break(self):
        r,b=fixture();r[2]['target_status']='missing'
        self.assertEqual(make_window(r,b,F(0),F(1))[1],'unresolved_or_invalid_timestamp')

    def test_reseed_break(self):
        r,b=fixture();r[2]['target_status']='seeded'
        self.assertEqual(make_window(r,b,F(0),F(1))[1],'track_break')

    def test_temporal_gap(self):
        r,b=fixture((0,.2,.8,.9,1,1.1))
        self.assertEqual(make_window(r,b,F(0),F(1))[1],'timestamp_gap')

    def test_excluded_segment_not_joined(self):
        _,b=fixture()
        a=annotation(0,2,b,_direct_confirmation_source=True)
        c=annotation(3,6,b,_direct_confirmation_source=True)
        ss=segments([a,c],b)
        self.assertEqual(len(ss),2)
        self.assertTrue(all(s['duration']<1 for s in ss))

    def test_overlap_and_boundary_rejected(self):
        _,b=fixture();a=annotation(0,3,b)
        with self.assertRaisesRegex(ValueError,'overlapping'):
            segments([a,a],b)
        a['end_sec']='.6'
        with self.assertRaisesRegex(ValueError,'boundary'):
            segments([a],b)

    def test_preliminary_not_promoted(self):
        _,b=fixture();a=annotation(0,6,b)
        self.assertFalse(segments([a],b)[0]['direct'])
        a['_direct_confirmation_source']=True;a['annotation_status']='draft'
        self.assertFalse(segments([a],b)[0]['direct'])

    def test_alignment(self):
        r,b=fixture();t=[dict(x,source_duration=200) for x in r]
        self.assertEqual(aligned_frames(r,t),b)
        t[2]['pts']+=1
        with self.assertRaisesRegex(ValueError,'mismatch'):aligned_frames(r,t)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):build('unused',d)


if __name__=='__main__':unittest.main()
