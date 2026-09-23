"""Focused cache identity and separate raw/quality drawing contracts; no inference."""
from copy import deepcopy
import importlib.util
from importlib import metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from patient_monitoring.pose_preview import KEYPOINT_NAMES, PREDICT_CONFIG
from patient_monitoring.pose_review import gate_edges
from patient_monitoring.video_inspection import InspectionError

spec=importlib.util.spec_from_file_location('backend_review',Path(__file__).resolve().parents[1]/'tools/generate_pose_backend_review.py')
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class BackendVisualReviewTests(unittest.TestCase):
    def fixture(self):
        return dict(detection_index=0,class_id=0,box_xyxy=[1.,2.,90.,99.],box_confidence=.9,
            keypoints=[dict(index=i,name=n,x=30.,y=40.,confidence=.9) for i,n in enumerate(KEYPOINT_NAMES)])

    def test_raw_boundary_and_low_confidence_preserved_separately(self):
        person=self.fixture()
        person['keypoints'][13].update(x=100.,confidence=.99)
        person['keypoints'][14].update(confidence=.1)
        before=deepcopy(person)
        raw,flags,drawing=review.split_quality([person],100,100)
        self.assertEqual(raw,[before])
        self.assertEqual(person,before)
        boundary=flags[0]['keypoints'][13]
        self.assertTrue(boundary['confidence_pass'])
        self.assertTrue(boundary['at_or_outside_boundary'])
        self.assertFalse(boundary['passes_quality_gate'])
        self.assertFalse(flags[0]['keypoints'][14]['confidence_pass'])
        self.assertTrue(all(13 not in edge and 14 not in edge for edge in gate_edges(drawing[0]['keypoints'])))
        self.assertNotIn('confidence_pass',raw[0]['keypoints'][0])
        self.assertNotIn('x',flags[0]['keypoints'][0])

    def test_zero_multiple_and_nonfinite_values_remain_represented(self):
        self.assertEqual(review.split_quality([],100,100),([],[],[]))
        a,b=self.fixture(),self.fixture()
        b['detection_index']=1
        b['keypoints'][0]['x']=float('nan')
        raw,flags,drawing=review.split_quality([a,b],100,100)
        self.assertEqual(len(raw),2)
        self.assertEqual([len(p['keypoints']) for p in raw],[17,17])
        self.assertIsNone(raw[1]['keypoints'][0]['x'])
        self.assertFalse(flags[1]['keypoints'][0]['finite_values'])
        self.assertFalse(drawing[1]['keypoints'][0]['passes_quality_gate'])
        json.dumps(raw,allow_nan=False)

    def test_reuse_rejects_changed_hash_timestamp_and_effective_settings(self):
        source=dict(frame_index=0,actual_time_seconds=0.,source_sha256='png',width=100,height=100)
        row={**source,'person_count':1,'people':[self.fixture()]}
        run=dict(status='complete',processed_image_count=1,input_csv_sha256='csv',model={'sha256':'pt'},
            config=dict(PREDICT_CONFIG),effective_predict_config={k:v for k,v in PREDICT_CONFIG.items() if k!='half'},
            actual_model_device='cpu',actual_model_dtype='torch.float32',
            package_versions={n:metadata.version(n) for n in ('ultralytics','torch','torchvision','numpy','opencv-python','pillow')})
        with TemporaryDirectory() as temp:
            folder=Path(temp)
            def save(r,record):
                (folder/'run_info.json').write_text(json.dumps(r))
                (folder/'poses.json').write_text(json.dumps({'images':[record]}))
            save(run,row)
            cached,_=review.verified_reuse(folder,[source],'csv','pt')
            self.assertEqual(cached[0]['people'],row['people'])
            for key,value in (('source_sha256','different'),('actual_time_seconds',1.)):
                changed={**row,key:value};save(run,changed)
                with self.assertRaises(InspectionError):review.verified_reuse(folder,[source],'csv','pt')
            changed=deepcopy(run);changed['effective_predict_config']['rect']=True
            save(changed,row)
            with self.assertRaises(InspectionError):review.verified_reuse(folder,[source],'csv','pt')
            save(run,row)
            with self.assertRaises(InspectionError):review.verified_reuse(folder,[source],'csv','other_pt')


if __name__=='__main__': unittest.main()
