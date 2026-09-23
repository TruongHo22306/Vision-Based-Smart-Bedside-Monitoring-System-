"""Review-only integration with synthetic frames, never inference."""

from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from patient_monitoring.target_selection import select_target
from patient_monitoring.target_review import target_review
from test_target_selection import person, records


class TargetReviewTests(unittest.TestCase):
    def fixture(self, root, mismatch=False):
        source, poses, output = root/'source.bin', root/'poses.jsonl', root/'out'
        source.write_bytes(b'synthetic source only')
        times = [0,10,70,80,90,100,110,120,130,140]
        rows=records(times, [[person(0)]] + [[]] + [[person(8)]]*8)
        for row in rows:
            row['source_path']=str(source.resolve())
        poses.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        select_target(poses,output,0,0)
        frames=[SimpleNamespace(pts=t+(1 if mismatch and i==4 else 0),time_base=Fraction(1,100),
            width=100,height=100,is_corrupt=False,to_image=lambda:Image.new('RGB',(100,100))) for i,t in enumerate(times)]
        container=MagicMock()
        container.__enter__.return_value=container
        stream=SimpleNamespace(index=0)
        container.streams.video=[stream]
        container.decode.return_value=iter(frames)
        original=(output/'target_poses.jsonl').read_bytes()
        with patch('patient_monitoring.target_review.av.open',return_value=container), \
                patch.dict('sys.modules',{'ultralytics':None}):
            result=target_review(source,poses,output,[0])
        self.assertEqual((output/'target_poses.jsonl').read_bytes(),original)
        container.decode.assert_called_once_with(stream)
        return result,output

    def test_review_selects_up_to_six_unresolved_and_keeps_nulls(self):
        with TemporaryDirectory() as tmp:
            result,output=self.fixture(Path(tmp))
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['unresolved_frame_count'],9)
            self.assertEqual(len(set(result['unresolved_review_frames'])),6)
            self.assertEqual(result['aligned_frame_count'],10)
            self.assertEqual(len(result['selected_frames']),7)
            for row in result['selected_frames']:
                if row['frame_index']:
                    self.assertIsNone(row['selection']['selected_person'])
            self.assertTrue((output/'review/unresolved_selection_sheet.jpg').is_file())

    def test_review_timestamp_mismatch_is_incomplete(self):
        with TemporaryDirectory() as tmp:
            result,_=self.fixture(Path(tmp),True)
            self.assertEqual(result['status'],'incomplete')
            self.assertIn('alignment failed',result['errors'][0])


if __name__ == '__main__':
    unittest.main()
