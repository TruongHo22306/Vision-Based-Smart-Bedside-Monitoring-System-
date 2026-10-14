from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import av
from PIL import Image

from patient_monitoring.action_baseline import sha256, write_json
from patient_monitoring.action_review import display_lines, export_review
from patient_monitoring.video_inspection import TimelineAudit


def fixture(root):
    source=root/'source.mp4';pred=root/'predictions.jsonl';target=root/'target_poses.jsonl'
    base=Fraction(1,30000); pts=[0,2000,4001,8000]; durations=[2000,2001,3999,1000]
    duration_map=dict(zip(pts,durations))
    with av.open(str(source),'w') as c:
        s=c.add_stream('libx264',rate=15);s.width=640;s.height=360;s.pix_fmt='yuv420p'
        s.time_base=s.codec_context.time_base=base;s.codec_context.max_b_frames=0
        for index,stamp in enumerate(pts):
            frame=av.VideoFrame.from_image(Image.new('RGB',(640,360),(index*30,50,70)))
            frame.pts=stamp;frame.time_base=base;frame.duration=duration_map[stamp]
            for p in s.encode(frame):p.duration=duration_map[p.pts];c.mux(p)
        for p in s.encode():p.duration=duration_map[p.pts];c.mux(p)
    predictions=[];targets=[];audit=TimelineAudit()
    with av.open(str(source)) as c:
        for f in c.decode(video=0):
            shared=dict(**audit.observe(f.pts,f.time_base),source_path=str(source.resolve()),width=f.width,height=f.height,
                        stream_index=0,selected_source_detection_index=None)
            targets.append(dict(**shared,status='unresolved',selected_person=None))
            predictions.append(dict(**shared,video_id='test',baseline_version='action_rules_v002',target_status='unresolved',
                predicted_label='unknown',reasons=['unresolved_target'],posture_evidence=None,motion_evidence=None,
                image_motion_cautions=[],locomotion_evidence={'status':'unknown'},config_sha256='test'))
    for p,rows in ((pred,predictions),(target,targets)):
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    write_json(root/'provenance.json',dict(status='complete',baseline_version='action_rules_v002',output_sha256=sha256(pred),
                                        config={},config_sha256='test',code_sha256='fixture'))
    return source,pred,target,predictions,targets


class ReviewTests(unittest.TestCase):
    def test_real_vfr_encoding_preserves_every_pts_duration_and_unknown_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video,pred,target,_,_=fixture(root)
            result=export_review(video,pred,target,root/'review')
            self.assertEqual(result['status'],'complete',result['errors'])
            self.assertEqual(result['aligned_frames'],4)
            self.assertEqual(result['verified_output_frames'],4)
            self.assertTrue(result['all_pts_and_durations_exact'])
            with self.assertRaises(ValueError):export_review(video,pred,target,root/'review')

    def test_prediction_target_misalignment_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video,pred,target,_,targets=fixture(root)
            targets[1]['pts']+=1
            target.write_text(''.join(json.dumps(r)+'\n' for r in targets))
            result=export_review(video,pred,target,root/'review')
            self.assertEqual(result['status'],'incomplete')
            self.assertIn('alignment mismatch',result['errors'][0])

    def test_count_mismatch_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video,pred,target,_,targets=fixture(root)
            target.write_text(''.join(json.dumps(r)+'\n' for r in targets[:-1]))
            result=export_review(video,pred,target,root/'review')
            self.assertEqual(result['status'],'incomplete')
            self.assertIn('frame counts differ',result['errors'][0])

    def test_motion_warning_is_separate_from_retained_posture(self):
        p=dict(video_id='test',frame_index=2,presentation_time_seconds=.2,pts=6000,time_base='1/30000',
               predicted_label='lying',posture_evidence='lying',reasons=['supported_posture_independent_of_image_motion'],
               image_motion_cautions=['motion_posture_conflict'],motion_evidence='high',locomotion_evidence={'status':'unknown'})
        lines=display_lines(p,{'selected_source_detection_index':0})
        self.assertIn('NOT GROUND TRUTH',lines[0]);self.assertIn('Predicted: lying',lines[2])
        self.assertIn('Warning: image motion / posture conflict',lines[4])
        self.assertIn('Locomotion: unknown',lines[5])


if __name__=='__main__':unittest.main()
