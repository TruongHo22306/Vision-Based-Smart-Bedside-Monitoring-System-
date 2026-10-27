import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from patient_monitoring import supervised_action as s

PREP = Path(__file__).resolve().parents[1]/'outputs/supervised_training_preparation_v001_20261021'


def row(value=2.0):
    return {'model_inputs':{c:dict(mean=value+i,std=0.5+i,valid=True,coverage=1.0) for i,c in enumerate(s.CHANNELS)}}


class SupervisedActionTests(unittest.TestCase):
    def setUp(self):
        self.schema = s.read(PREP/'feature_schema.json')

    def test_exact_order(self):
        x=s.vectorize([row()],self.schema)
        np.testing.assert_array_equal(x[0,:14],np.array([[2+i,.5+i] for i in range(7)]).ravel())
        np.testing.assert_array_equal(x[0,14:21],np.ones(7));np.testing.assert_array_equal(x[0,21:],np.zeros(7))
        self.assertEqual(x.dtype,np.float64)

    def test_metadata_never_enters_inputs(self):
        a=row();b=copy.deepcopy(a);b.update(label='moving',video_id='OTHER',start=-999,recording_occasion_group_id='SECRET',baseline_predictions=['lying'],source_path='x')
        np.testing.assert_array_equal(s.vectorize([a],self.schema),s.vectorize([b],self.schema))

    def test_invalid_values(self):
        for patch in [dict(mean=float('nan')),dict(std=-1),dict(valid=1),dict(coverage=2),dict(mean=None),dict(valid=False),dict(mean=True)]:
            with self.subTest(patch=patch):
                a=row();a['model_inputs'][s.CHANNELS[0]].update(patch)
                with self.assertRaisesRegex(ValueError,'Row 0'):s.vectorize([a],self.schema)

    def test_channel_and_schema_mismatch(self):
        a=row();a['model_inputs']['label']={}
        with self.assertRaises(ValueError):s.vectorize([a],self.schema)
        schema=copy.deepcopy(self.schema);schema['column_order']=list(reversed(schema['column_order']))
        with self.assertRaises(ValueError):s.vectorize([row()],schema)

    def test_training_only_fit(self):
        train=s.vectorize([row(1),row(3)],self.schema);prep=s.fit_preprocessing(train,self.schema)
        self.assertEqual(prep['medians'][0],2);self.assertEqual(prep['centers'][0],2);self.assertEqual(prep['scales'][0],1)
        snapshot=copy.deepcopy(prep);held=s.vectorize([row(1000)],self.schema);x=s.transform(held,prep)
        self.assertEqual(x[0,0],998);self.assertEqual(prep,snapshot)

    def test_missing_median_and_flag(self):
        a=row();a['model_inputs'][s.CHANNELS[0]]=dict(mean=None,std=None,valid=False,coverage=0)
        x=s.vectorize([row(1),row(3),a],self.schema);prep=s.fit_preprocessing(x,self.schema);y=s.transform(x,prep)
        self.assertEqual(prep['medians'][0],2);self.assertEqual(y.shape,(3,28));self.assertEqual(y[2,0],0);self.assertEqual(y[2,21],1)
        self.assertTrue(np.isnan(x[2,0]))

    def test_all_missing_columns_constant_and_unseen(self):
        a=row()
        for c in s.CHANNELS:a['model_inputs'][c]=dict(mean=None,std=None,valid=False,coverage=0)
        x=s.vectorize([a,a],self.schema);prep=s.fit_preprocessing(x,self.schema)
        self.assertTrue(all(prep['all_missing_fallback']));self.assertEqual(prep['scales'],[1]*14)
        y=s.transform(x,prep);self.assertEqual(y.shape,(2,28));self.assertTrue(np.isfinite(y).all());self.assertEqual(prep['all_channels_missing_rows'],2)
        q=s.fit_preprocessing(s.vectorize([row(),row()],self.schema),self.schema)
        self.assertTrue(all(q['constant_columns']));self.assertFalse(any(q['all_missing_fallback']))
        self.assertTrue(np.isfinite(s.transform(x,q)).all())

    def test_coverage_roundoff_not_rewritten(self):
        a=row();a['model_inputs'][s.CHANNELS[0]]['coverage']=1+5e-16
        x=s.vectorize([a],self.schema);self.assertGreater(x[0,14],1)
        prep=s.fit_preprocessing(x,self.schema);self.assertEqual(s.transform(x,prep)[0,14],x[0,14])

    def test_persistence_without_training(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);x=s.vectorize([row(1),row(3)],self.schema);prep=s.fit_preprocessing(x,self.schema)
            model=torch.nn.Linear(28,4,dtype=torch.float64)
            with torch.no_grad():model.weight.fill_(.1);model.bias.copy_(torch.arange(4,dtype=torch.float64))
            s.save_bundle(p,model,prep,PREP/'feature_schema.json',PREP/'experiment_config.json')
            restored,pp,ss=s.load_bundle(p)
            with torch.no_grad():
                a=model(torch.from_numpy(s.transform(x,prep)));b=restored(torch.from_numpy(s.transform(s.vectorize([row(1),row(3)],ss),pp)))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            with self.assertRaises(ValueError):s.load_bundle(p,expected_schema_hash='wrong')
            (p/'preprocessing.json').write_text('{}')
            with self.assertRaises(ValueError):s.load_bundle(p)

    def test_preprocessing_schema_and_invalid_scale(self):
        x=s.vectorize([row()],self.schema);p=s.fit_preprocessing(x,self.schema);p['scales'][0]=0
        with self.assertRaises(ValueError):s.transform(x,p)
        p=s.fit_preprocessing(x,self.schema);p['class_order'].reverse()
        with self.assertRaises(ValueError):s.transform(x,p)

    def test_refuse_overwrite_before_read_config(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'sentinel').write_text('keep')
            with self.assertRaises(FileExistsError):s.train('not-a-config',p)
            self.assertEqual((p/'sentinel').read_text(),'keep');self.assertEqual(len(list(p.iterdir())),1)

    def test_bad_config_leaves_incomplete_no_steps(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'bad.json').write_text('{}');s.write(p/'config_hash.json',{'config_sha256':'bad'})
            with self.assertRaises(ValueError):s.train(p/'bad.json',p/'new')
            info=s.read(p/'new/run_info.json');self.assertEqual(info['status'],'incomplete');self.assertEqual(info['training_steps_completed'],0)

    def test_metric_denominators(self):
        m=s.metrics([0,0,1],[0,3,1]);self.assertEqual(m['correct'],2)
        self.assertEqual(m['per_class']['standing']['recall'],.5)
        self.assertIsNone(m['per_class']['lying']['f1']);self.assertIsNone(m['per_class']['moving']['recall'])
        self.assertEqual(m['per_class']['moving']['precision'],0)


if __name__=='__main__':unittest.main()
