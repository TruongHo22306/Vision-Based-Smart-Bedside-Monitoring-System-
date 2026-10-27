import copy
import json
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from patient_monitoring.supervised_dataset import aligned_frames,csv_rows,make_window
from patient_monitoring.supervised_action import read,vectorize,transform,load_bundle
from patient_monitoring.supervised_predict import windows,display_rows
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'outputs/supervised_action_development_v001_20261021'

class PredictionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  inputs=read(ROOT/'outputs/SE11_V002_annotations_dataset_v005_20261021/inputs.json')
  v=next(v for v in inputs['videos'] if v['video_id']=='P001_SE08_V002')
  cls.rows=[json.loads(x) for x in Path(v['features']).read_text().splitlines()]
  cls.bounds=aligned_frames(cls.rows,csv_rows(v['timeline']))
  cls.schema=read(MODEL/'feature_schema.json')
 def test_training_parity_identical_support(self):
  train=make_window(self.rows,self.bounds,Fraction(0),Fraction(1))[0]
  pred=next(d['support'] for d in windows(self.rows,self.bounds) if d['decision_time']=='1')
  self.assertEqual(train,pred)
  np.testing.assert_array_equal(vectorize([train],self.schema),vectorize([pred],self.schema))
  actual=[json.loads(x) for x in (ROOT/'outputs/SE11_V002_annotations_dataset_v005_20261021/dataset/windows.jsonl').read_text().splitlines()]
  a=next(r for r in actual if r['video_id']=='P001_SE08_V002' and r['start']==0 and r['end']==1)
  np.testing.assert_array_equal(vectorize([a],self.schema),vectorize([pred],self.schema))
 def test_no_future_feature_access(self):
  changed=copy.deepcopy(self.rows)
  for i,r in enumerate(changed):
   if self.bounds[i]>=1:r['features']={'poison':None}
  a=make_window(self.rows,self.bounds,Fraction(0),Fraction(1))
  b=make_window(changed,self.bounds,Fraction(0),Fraction(1));self.assertEqual(a,b)
 def test_annotation_independent_schedule(self):
  changed=copy.deepcopy(self.rows)
  for r in changed:r.update(activity_label='arbitrary',annotation_boundary=True,baseline_prediction='lying')
  a=list(windows(self.rows,self.bounds));b=list(windows(changed,self.bounds));self.assertEqual(a,b)
  self.assertEqual([d['decision_time'] for d in a[:5]],['0','1/4','1/2','3/4','1'])
 def test_gap_and_reseed_not_bridged(self):
  for status in ['no_detections','seeded']:
   rows=copy.deepcopy(self.rows);rows[5]['target_status']=status
   d=next(d for d in windows(rows,self.bounds) if d['decision_time']=='1')
   self.assertFalse(d['available'])
 def test_frozen_preprocessing_no_fit(self):
  with patch('patient_monitoring.supervised_action.fit_preprocessing',side_effect=AssertionError('must not fit')):
   model,prep,schema=load_bundle(MODEL);old=copy.deepcopy(prep)
   value=make_window(self.rows,self.bounds,Fraction(0),Fraction(1))[0]
   with torch.no_grad():logits=model(torch.from_numpy(transform(vectorize([value],schema),prep)))
   self.assertTrue(torch.isfinite(logits).all());self.assertEqual(prep,old)
 def test_display_timing_and_clear(self):
  rows=copy.deepcopy(self.rows)
  decisions=[dict(decision_time='1',available=True,label='standing',availability_reason=None)]
  display=list(display_rows(rows,decisions))
  for r,d in zip(rows,display):
   if Fraction(r['pts'])*Fraction(r['time_base'])<1:self.assertIsNone(d['displayed_label'])
  idx=next(i for i,t in enumerate(self.bounds[:-1]) if t>=Fraction(3,2))
  rows[idx]['target_status']='no_detections'
  display=list(display_rows(rows,decisions))
  self.assertEqual(display[idx-1]['displayed_label'],'standing');self.assertTrue(all(d['displayed_label'] is None for d in display[idx:]))
 def test_unavailable_decision_clears(self):
  decisions=[dict(decision_time='1',available=True,label='standing',availability_reason=None),dict(decision_time='5/4',available=False,label=None,availability_reason='track_break')]
  display=list(display_rows(self.rows,decisions))
  self.assertTrue(all(d['displayed_label'] is None for d in display if Fraction(d['pts'])*Fraction(d['time_base'])>=Fraction(5,4)))

if __name__=='__main__':unittest.main()
