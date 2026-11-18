import unittest,tempfile,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class UpdateTests(unittest.TestCase):
 def test_backup_verify_tamper_and_payload(self):
  spec=importlib.util.spec_from_file_location('update',ROOT/'tools/pi_acquisition_update.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);stage=root/'stage';project=root/'project';stage.mkdir();project.mkdir()
   (project/'AGENTS.md').write_text('test');(project/'pyproject.toml').write_text('test')
   for n in u.FILES:
    p=stage/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('new')
   first=u.FILES[0];p=project/first;p.parent.mkdir(parents=True);p.write_text('old')
   (stage/u.MANIFEST).write_text(''.join(u.sha(stage/n)+'  '+n+'\n' for n in u.FILES))
   receipt=u.apply(stage,project);self.assertEqual(receipt['status'],'complete');self.assertEqual((Path(receipt['backup_path'])/first).read_text(),'old')
   self.assertEqual(len(receipt['replaced']),len(u.FILES))
   (stage/first).write_text('tamper')
   with self.assertRaises(ValueError):u.apply(stage,project)
  self.assertFalse(any('.venv' in n or n.endswith('.mp4') or n.startswith('models/') for n in u.FILES))
if __name__=='__main__':unittest.main()
