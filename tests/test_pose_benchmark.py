"""Synthetic timer/telemetry checks; these are not Raspberry Pi benchmarks."""

from fractions import Fraction
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from patient_monitoring.pose_benchmark import benchmark_pose, prepare_frames, sample_telemetry, throttle_flags
from patient_monitoring.video_inspection import InspectionError, TimelineAudit


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video, self.weights = self.root/'input.bin', self.root/'weights.pt'
        self.video.write_bytes(b'synthetic video placeholder')
        self.weights.write_bytes(b'synthetic weights placeholder, never loaded')
        self.now = 0.

    def fixture(self, imgsz=640, warmup=2, frames=3, fail=False):
        durations = iter([99.]*warmup + [float(i+1) for i in range(frames)])
        calls = []
        def predict(**kwargs):
            calls.append(kwargs)
            self.now += next(durations)
            if fail and len(calls) == warmup+2:
                raise RuntimeError('synthetic predict failure')
            return [SimpleNamespace(orig_shape=(16,32), boxes=[],
                speed=dict(preprocess=2., inference=10., postprocess=3.))]
        def prepare(source, count, prepared):
            self.now += 5
            audit=TimelineAudit()
            for i in range(count):
                row=audit.observe(100+i*7,Fraction(1,100))
                row.update(stream_index=0,width=32,height=16)
                prepared.append((row,Image.new('RGB',(32,16))))
            return audit.summary()
        def load(*args):
            self.now += 20
            return SimpleNamespace(predict=predict),dict(intraop=4,interop=4)
        def telemetry():
            self.now += 10
            return dict(status='unavailable',collection_seconds=10)
        out=self.root/f'run_{imgsz}_{warmup}_{frames}_{fail}'
        with patch('patient_monitoring.pose_benchmark.perf_counter',side_effect=lambda:self.now), \
             patch('patient_monitoring.pose_benchmark.prepare_frames',side_effect=prepare), \
             patch('patient_monitoring.pose_benchmark.load_local_model',side_effect=load), \
             patch('patient_monitoring.pose_benchmark.sample_telemetry',side_effect=telemetry), \
             patch('patient_monitoring.pose_benchmark.runtime_snapshot',return_value={'actual_model_device':'cpu','actual_model_dtype':'torch.float32'}):
            report=benchmark_pose(self.video,self.weights,out,imgsz,warmup,frames)
        return report,out,calls

    def test_warmup_and_telemetry_excluded_from_predict_statistics(self):
        report,out,calls=self.fixture()
        self.assertTrue(report['completed'])
        stats=report['statistics']
        self.assertEqual(stats['measured_completed'],3)
        self.assertEqual(stats['predict_seconds_sum'],6)
        self.assertEqual(stats['predict_median_seconds'],2)
        self.assertAlmostEqual(stats['predict_p95_seconds'],2.9)
        self.assertEqual(stats['predict_call_throughput_frames_per_second'],.5)
        self.assertEqual(report['data_preparation_seconds'],5)
        self.assertEqual(report['model_loading_seconds'],20)
        self.assertEqual(report['total_benchmark_elapsed_seconds'],279)
        rows=[json.loads(s) for s in (out/'frame_timings.jsonl').read_text().splitlines()]
        self.assertEqual([r['phase'] for r in rows],['warmup']*2+['measured']*3)
        telemetry=[json.loads(s) for s in (out/'telemetry.jsonl').read_text().splitlines()]
        self.assertEqual([r['event'] for r in telemetry],['before_warmup','after_warmup']+['after_measured_frame']*3)
        self.assertEqual([r['source_frame_index'] for r in telemetry[2:]],[0,1,2])
        for call in calls:
            self.assertEqual((call['device'],call['batch'],call['half'],call['rect']),('cpu',1,False,False))
            self.assertEqual((call['conf'],call['iou']),(.25,.7))

    def test_imgsz_and_warmup_do_not_change_measured_frame_selection(self):
        a,_,_=self.fixture(320,0)
        b,_,_=self.fixture(640,5)
        self.assertEqual(a['selected_frames'],b['selected_frames'])
        self.assertEqual(a['statistics']['measured_completed'],3)
        self.assertEqual(b['warmup_completed'],5)

    def test_predict_failure_keeps_incomplete_report_and_partial_rows(self):
        report,out,_=self.fixture(warmup=0,fail=True)
        self.assertFalse(report['completed'])
        self.assertTrue(report['statistics']['partial_run'])
        self.assertEqual(report['statistics']['measured_completed'],1)
        self.assertEqual(report['errors'][0]['stage'],'measured_predict')
        rows=[json.loads(s) for s in (out/'frame_timings.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['status'],'incomplete')

    def test_invalid_inputs_create_incomplete_without_loading_model(self):
        cases=[dict(imgsz=0),dict(imgsz=321),dict(warmup=-1),dict(frames=0),
               dict(weights=self.root/'missing.pt'),dict(source=self.root/'missing.mp4')]
        for i,override in enumerate(cases):
            arguments=dict(source=self.video,weights=self.weights,output=self.root/f'invalid{i}')
            arguments.update(override)
            with patch('patient_monitoring.pose_benchmark.load_local_model') as loader:
                report=benchmark_pose(**arguments)
                loader.assert_not_called()
            self.assertFalse(report['completed'])
            self.assertEqual(report['errors'][0]['stage'],'validate_inputs')

    def test_existing_results_are_never_replaced(self):
        out=self.root/'existing'
        out.mkdir()
        sentinel=out/'benchmark_summary.json'
        sentinel.write_text('keep')
        with self.assertRaises(InspectionError):
            benchmark_pose(self.video,self.weights,out)
        self.assertEqual(sentinel.read_text(),'keep')

    def test_sequential_selection_preserves_pts_and_short_video_fails(self):
        container=MagicMock()
        container.__enter__.return_value=container
        stream=SimpleNamespace(index=2)
        container.streams.video=[stream]
        frames=[SimpleNamespace(pts=p,time_base=Fraction(1,100),width=32,height=16,is_corrupt=False,
            to_image=lambda:Image.new('RGB',(32,16))) for p in (500,507,523)]
        container.decode.side_effect=lambda _:iter(frames)
        with patch('patient_monitoring.pose_benchmark.av.open',return_value=container):
            prepared=[]
            prepare_frames(self.video,2,prepared)
            self.assertEqual([r['frame_index'] for r,_ in prepared],[0,1])
            self.assertEqual([r['pts'] for r,_ in prepared],[500,507])
            self.assertEqual([r['presentation_time_seconds'] for r,_ in prepared],[5.,5.07])
            for _,im in prepared: im.close()
            prepared=[]
            with self.assertRaises(InspectionError): prepare_frames(self.video,4,prepared)
            for _,im in prepared: im.close()


class TelemetryTests(unittest.TestCase):
    def test_unavailable_is_not_zero_flags(self):
        with patch('patient_monitoring.pose_benchmark.shutil.which',return_value=None), \
             patch('patient_monitoring.pose_benchmark.subprocess.run') as command:
            result=sample_telemetry()
        command.assert_not_called()
        self.assertEqual(result['status'],'unavailable')
        self.assertIsNone(result['flags'])
        self.assertIsNone(result['temperature_celsius'])

    def test_current_and_history_are_distinct_with_short_timeouts(self):
        with patch('patient_monitoring.pose_benchmark.shutil.which',return_value='/usr/bin/vcgencmd'), \
             patch('patient_monitoring.pose_benchmark.subprocess.run',side_effect=[
                 SimpleNamespace(stdout="temp=74.0'C"),SimpleNamespace(stdout='throttled=0xd0000')]) as command:
            result=sample_telemetry()
        self.assertEqual(result['temperature_celsius'],74.)
        self.assertFalse(any(result['flags']['current'].values()))
        self.assertTrue(result['flags']['historical']['undervoltage'])
        self.assertTrue(result['flags']['historical']['throttled'])
        self.assertTrue(result['flags']['historical']['soft_temperature_limit'])
        self.assertTrue(throttle_flags(4)['current']['throttled'])
        self.assertTrue(all(c.kwargs['timeout']==1. for c in command.call_args_list))

    def test_timeout_and_bad_response_are_unavailable(self):
        with patch('patient_monitoring.pose_benchmark.shutil.which',return_value='vcgencmd'), \
             patch('patient_monitoring.pose_benchmark.subprocess.run',side_effect=[
                 subprocess.TimeoutExpired('measure_temp',1.),SimpleNamespace(stdout='invalid')]):
            result=sample_telemetry()
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(len(result['errors']),2)


if __name__ == '__main__':
    unittest.main()
