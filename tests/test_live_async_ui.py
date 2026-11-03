import threading,time,unittest
from types import SimpleNamespace
from patient_monitoring.live_async_ui import LatestWorker,UIState,run_loop


def result(epoch=0,stamp=0,frame=1,label='standing',reason=None,support=0):
    return dict(target_epoch=epoch,source_timestamp_ns=stamp,frame_id=frame,label=label,reason=reason,
                latest_support_host_ns=support,completed_host_ns=2000000000)

class AsyncTests(unittest.TestCase):
    def test_single_worker_latest_queue_and_clean_shutdown(self):
        started=threading.Event();release=threading.Event();seen=[]
        def process(epoch,payload):
            seen.append(payload);started.set();release.wait(2);return result(epoch,frame=payload)
        w=LatestWorker(process,lambda e,c:None)
        try:
            w.submit(0);self.assertTrue(started.wait(1))
            for i in range(1,101):w.submit(i)
            self.assertEqual(w.pending[1],100);self.assertEqual(w.counts['overwritten_pending_frames'],99)
            release.set()
            deadline=time.monotonic()+2
            while len(seen)<2 and time.monotonic()<deadline:time.sleep(.002)
            self.assertEqual(seen,[0,100])
        finally:release.set();self.assertTrue(w.close());self.assertFalse(w.thread.is_alive())

    def test_reset_inflight_discards_old_epoch_and_exact_seed(self):
        started=threading.Event();release=threading.Event();applied=threading.Event();controls=[]
        def process(epoch,payload):started.set();release.wait(2);return result(epoch,frame=payload)
        def control(epoch,seed):controls.append((epoch,seed));applied.set();return None
        w=LatestWorker(process,control)
        try:
            w.submit(10);self.assertTrue(started.wait(1));seed=(result(frame=7),3)
            self.assertEqual(w.reset(seed),1);release.set();self.assertTrue(applied.wait(1))
            self.assertEqual(controls,[(1,seed)]);self.assertIsNone(w.take());self.assertEqual(w.counts['discarded_old_epoch_results'],1)
        finally:release.set();self.assertTrue(w.close())

    def test_old_and_out_of_order_results(self):
        s=UIState(1.25);s.reset(2,0)
        self.assertFalse(s.accept(result(1)));self.assertTrue(s.accept(result(2,100)))
        self.assertFalse(s.accept(result(2,99)));self.assertEqual(s.result['source_timestamp_ns'],100)

    def test_slow_worker_expiry_uses_source_not_completion_or_receive(self):
        s=UIState(1.25);s.accept(result(support=1000000000))
        self.assertEqual(s.tick(2000000000)['label'],'standing')
        self.assertIsNone(s.tick(2250000001)['label'])
        self.assertEqual(s.tick(3000000000)['reason'],'display_support_expired')
        self.assertAlmostEqual(s.summary(4000000000)['fresh_label_submitted_duration_seconds'],.25)
        self.assertEqual(s.counts['support_expired_results'],1)

    def test_newest_unavailable_never_falls_back(self):
        s=UIState(1.25);s.accept(result());s.tick(100000000)
        s.accept(result(stamp=200000000,frame=2,label=None,reason='insufficient_observed_span',support=None))
        self.assertIsNone(s.tick(200000000)['label']);self.assertEqual(s.tick(300000000)['reason'],'insufficient_observed_span')

    def test_reset_clears_label_and_duration(self):
        s=UIState(1.25);s.accept(result());s.tick(100000000);s.reset(1,200000000)
        self.assertIsNone(s.tick(300000000)['label']);self.assertEqual(s.summary(400000000)['fresh_label_submitted_duration_seconds'],.1)

    def test_worker_failure_propagates(self):
        def fail(e,x):raise ValueError('bad camera image')
        w=LatestWorker(fail,lambda e,c:None);w.submit(1);w.thread.join(1)
        with self.assertRaisesRegex(RuntimeError,'bad camera image'):w.take()
        self.assertTrue(w.close())

    def test_camera_failure_and_cleanup(self):
        class Capture:
            status=SimpleNamespace(value=-1)
        destroyed=[];live=SimpleNamespace(DisplaySummary=lambda:SimpleNamespace(summary=lambda:{}),cv2=SimpleNamespace(destroyAllWindows=lambda:destroyed.append(True)))
        args=SimpleNamespace(duration=.1,no_display=False);engine=SimpleNamespace(config={'max_latest_age_seconds':1.25});info={'processed':0}
        with self.assertRaisesRegex(RuntimeError,'Camera open/read failure'):
            run_loop(live,args,Capture(),None,engine,{},info,None,None,None,None,time.monotonic())
        self.assertTrue(info['async_worker_stopped']);self.assertEqual(destroyed,[True])

    def test_ui_main_thread_required(self):
        errors=[]
        def invoke():
            try:run_loop(None,None,None,None,None,None,None,None,None,None,None,0)
            except RuntimeError as e:errors.append(str(e))
        t=threading.Thread(target=invoke);t.start();t.join();self.assertEqual(errors,['UI must run on main thread'])

    def test_full_loop_fake_camera_ui_frame_alignment_and_seed(self):
        import numpy as np
        from patient_monitoring.live_demo import DisplaySummary
        from patient_monitoring.live_diagnostics import Performance
        from unittest.mock import Mock
        main_id=threading.get_ident();ui_threads=[];predict_threads=[];painted=[];logs=[];seeded=[]
        class Capture:
            status=SimpleNamespace(value=0)
            n=0
            def timeout_reason(self):return None
            def diagnostics(self):return []
            def newest(self):
                self.n+=1
                return self.n,time.monotonic_ns(),np.full((2,2,3),self.n,dtype=np.uint8)
        class CV:
            COLOR_BGR2RGB=1;FONT_HERSHEY_SIMPLEX=0
            last=None;seed_sent=False
            def cvtColor(self,a,b):return a
            def putText(self,*a):pass
            def imshow(self,*a):ui_threads.append(threading.get_ident())
            def waitKey(self,n):
                if painted and not self.seed_sent:self.seed_sent=True;return ord('0')
                return 255
            def getWindowProperty(self,*a):return 1
            WND_PROP_VISIBLE=0
            def destroyAllWindows(self):ui_threads.append(threading.get_ident())
        class Pose:
            def predict(self,source,**kwargs):
                predict_threads.append(threading.get_ident());time.sleep(.035)
                return [int(source[0,0,0])]
        class Engine:
            config={'max_latest_age_seconds':1.25};epoch=0;latest_step=None
            label=None;reason='startup';display_support_host_ns=None;display_decision_host_ns=None;display_computed_host_ns=None
            def reset(self):self.epoch+=1
            def step(self,people,w,h,stamp,seed=None):
                if seed is not None:seeded.append(people[0]['marker']);self.reset()
                return dict(status='matched',selected_person=None),[],0.
        def paint(frame,people,selected,engine,rate,age):
            if people:
                self.assertEqual(int(frame[0,0,0]),people[0]['marker']);painted.append(people[0]['marker'])
            return frame.copy()
        cv=CV();live=SimpleNamespace(np=np,DisplaySummary=DisplaySummary,cv2=cv,Image=SimpleNamespace(fromarray=lambda x:x),quality_people=lambda p,w,h:[dict(detection_index=0,marker=p)],paint=paint)
        args=SimpleNamespace(duration=.3,no_display=False,pose_backend='pt');info={'processed':0}
        stats=run_loop(live,args,Capture(),Pose(),Engine(),{},info,SimpleNamespace(add=logs.append),Performance(),lambda *a:None,lambda *a:None,time.monotonic())
        self.assertGreater(info['processed'],1);self.assertTrue(info['async_worker_stopped']);self.assertTrue(seeded)
        self.assertEqual(set(ui_threads),{main_id});self.assertEqual(len(set(predict_threads)),1);self.assertNotIn(main_id,predict_threads)
        requests=[r for r in logs if r['type']=='explicit_seed_requested']
        self.assertEqual(seeded[0],requests[0]['frame_id']);self.assertLess(stats['first_stamp'],stats['last_stamp'])

    def test_source_fetched_only_when_worker_ready(self):
        ready=threading.Event();release=threading.Event();done=threading.Event();seen=[];source_calls=[];latest=[0]
        def source():
            source_calls.append(threading.get_ident())
            if len(source_calls)>2:return None
            return latest[0]
        def process(epoch,frame):
            seen.append(frame)
            if len(seen)==1:ready.set();release.wait(2)
            else:done.set()
            return result(epoch,frame=frame)
        w=LatestWorker(process,lambda e,c:None,source=source)
        try:
            self.assertTrue(ready.wait(1));latest[0]=106
            self.assertEqual(len(source_calls),1);release.set();self.assertTrue(done.wait(1))
            self.assertEqual(seen[:2],[0,106]);self.assertNotIn(threading.get_ident(),source_calls)
        finally:release.set();self.assertTrue(w.close())

    def test_accounting_excludes_replacements_not_results(self):
        from patient_monitoring.live_async_ui import async_capture_accounting
        info=dict(captured=676,processed=147,overwritten_pending=334,dropped_incoming=66,drained_older=1,
                  async_queue_counts=dict(overwritten_pending_frames=125,frames_cleared_on_reset=1,discarded_old_epoch_results=3,overwritten_results=2))
        a=async_capture_accounting(info);self.assertEqual(a['legacy_unprocessed_residual'],128);self.assertEqual(a['unprocessed_on_stop'],2)
        info['async_queue_counts']['frames_cleared_on_stop']=1
        self.assertEqual(async_capture_accounting(info)['unprocessed_on_stop'],1)

    def test_headless_not_counted_as_ui(self):
        s=UIState(1.25);s.accept(result());r=s.tick(100000000,False)
        self.assertFalse(r['label_submitted_to_ui']);self.assertEqual(s.summary(200000000)['fresh_label_submitted_duration_seconds'],0.)

if __name__=='__main__':unittest.main()
