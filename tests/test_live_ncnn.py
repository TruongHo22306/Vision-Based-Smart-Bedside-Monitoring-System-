import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from patient_monitoring.video_inspection import InspectionError
from patient_monitoring.live_pose import validate_pose,load_pose
ROOT=Path(__file__).resolve().parents[1]
NCNN=ROOT/'models/ncnn_trial_320/yolo11n_pose_320_ncnn_model'
class LiveNCNNTests(unittest.TestCase):
 def test_explicit_identity_and_size(self):
  v=validate_pose(NCNN,'ncnn',320);self.assertEqual(len(v['artifacts']),3)
  for p,b,s in [(NCNN,'pt',320),(NCNN,'ncnn',640),(ROOT/'missing','ncnn',320),(ROOT/'models/yolo11n-pose.pt','ncnn',320)]:
   with self.assertRaises((ValueError,InspectionError)):validate_pose(p,b,s)
 def test_missing_and_wrong_metadata(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x_ncnn_model';p.mkdir()
   with self.assertRaises((ValueError,InspectionError)):validate_pose(p,'ncnn',320)
   for n in ('model.ncnn.param','model.ncnn.bin'): (p/n).write_text('test')
   (p/'metadata.yaml').write_text('task: detect\nbatch: 1\nimgsz: [320,320]\nkpt_shape: [17,3]')
   with self.assertRaises((ValueError,InspectionError)):validate_pose(p,'ncnn',320)
 def test_reuses_ncnn_loader_without_pt_methods(self):
  sentinel=object()
  with tempfile.TemporaryDirectory() as td,patch('patient_monitoring.pose_benchmark.load_local_model',return_value=(sentinel,{})) as loader:
   self.assertIs(load_pose(NCNN,'ncnn',Path(td),{'imgsz':320}),sentinel)
   self.assertTrue((Path(td)/'library_settings').is_dir());loader.assert_called_once()
if __name__=='__main__':unittest.main()
