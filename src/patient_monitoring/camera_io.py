"""Lightweight Windows spawn/camera layer. No pose/classifier or numerical imports."""
import multiprocessing as mp
import queue
import sys
import time

BACKENDS=('auto','dshow','msmf','v4l2')


def offer_latest(q,item,counters):
    try:q.put_nowait(item);return
    except queue.Full:pass
    try:q.get(timeout=.02);counters[1]+=1
    except queue.Empty:pass
    try:q.put(item,timeout=.02)
    except queue.Full:counters[2]+=1


def startup_timeout(stage,stage_ns,launched_ns,now_ns,limit_seconds=10):
    elapsed=(now_ns-(stage_ns or launched_ns))/1e9
    if elapsed<=limit_seconds:return None
    if stage in (0,1):return 'worker_startup_timeout' if stage==0 else 'worker_import_timeout'
    if stage==2:return 'device_open_timeout'
    if stage in (3,4):return 'first_frame_timeout'
    return None


def capture_loop(camera,q,stop,counters,status,factory=None,backend='auto',events=None,stage=None,stage_ns=None,max_frames=None):
    cap=None
    def emit(name,code=None,**extra):
        stamp=time.monotonic_ns()
        if code is not None:
            if stage_ns is not None:stage_ns.value=stamp
            if stage is not None:stage.value=code
        record=dict(stage=name,monotonic_ns=stamp,wall_time_ns=time.time_ns(),camera=camera,backend=backend,**extra)
        if events is not None:
            try:events.put_nowait(record)
            except queue.Full:pass
    try:
        emit('worker_started',1,heavy_modules_loaded=[m for m in ('torch','ultralytics','patient_monitoring.supervised_action','patient_monitoring.pose_extraction') if m in sys.modules])
        import cv2
        emit('opencv_imported',opencv_version=cv2.__version__)
        api={'auto':cv2.CAP_ANY,'dshow':cv2.CAP_DSHOW,'msmf':cv2.CAP_MSMF,'v4l2':cv2.CAP_V4L2}[backend]
        emit('before_VideoCapture',2,api_preference=api)
        cap=(factory(camera) if factory else cv2.VideoCapture(camera,api))
        opened=bool(cap.isOpened());emit('open_returned',3,isOpened=opened)
        if not opened:raise RuntimeError('camera_open_failed')
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        status.value=1
        while not stop.is_set():
            if not counters[0]:emit('before_first_read',4)
            ok,frame=cap.read();stamp=time.monotonic_ns()
            if not counters[0]:emit('first_read_returned',5,ok=bool(ok),shape=None if frame is None else list(frame.shape))
            if not ok or frame is None:raise RuntimeError('camera_read_failed')
            seq=counters[0]
            if not seq:counters[3]=stamp
            counters[4]=stamp;counters[0]+=1
            offer_latest(q,(seq,stamp,frame),counters)
            if max_frames and counters[0]>=max_frames:break
    except BaseException as exc:
        status.value=-1;emit('worker_error',error=repr(exc))
    finally:
        if cap is not None:
            emit('before_release')
            cap.release();emit('device_released')
        emit('worker_finished')
        if hasattr(q,'cancel_join_thread'):q.cancel_join_thread()


class Capture:
    def __init__(self,camera,backend='auto',max_frames=None):
        if backend not in BACKENDS:raise ValueError('Unknown backend')
        ctx=mp.get_context('spawn');self.q=ctx.Queue(maxsize=1);self.events=ctx.Queue(maxsize=32);self.stop=ctx.Event();self.counters=ctx.Array('q',5);self.status=ctx.Value('i',0)
        self.stage=ctx.Value('i',0);self.stage_ns=ctx.Value('q',0);self.launched_ns=time.monotonic_ns()
        self.process=ctx.Process(target=capture_loop,args=(camera,self.q,self.stop,self.counters,self.status),kwargs=dict(backend=backend,events=self.events,stage=self.stage,stage_ns=self.stage_ns,max_frames=max_frames),daemon=True)
        self.process.start();self.drained=0;self.stages=[]
    def diagnostics(self):
        new=[]
        while True:
            try:r=self.events.get_nowait();r['since_launch_seconds']=(r['monotonic_ns']-self.launched_ns)/1e9;new.append(r)
            except queue.Empty:break
        self.stages.extend(new);return new
    def timeout_reason(self,now_ns=None):
        return startup_timeout(self.stage.value,self.stage_ns.value,self.launched_ns,now_ns or time.monotonic_ns())
    def newest(self,timeout=.05):
        try:item=self.q.get(timeout=timeout)
        except queue.Empty:return None
        while True:
            try:item=self.q.get_nowait();self.drained+=1
            except queue.Empty:return item
    def close(self):
        self.stop.set();self.process.join(2);forced=self.process.is_alive()
        if forced:self.process.terminate();self.process.join(2)
        self.diagnostics();alive=self.process.is_alive()
        self.q.close();self.q.cancel_join_thread();self.events.close();self.events.cancel_join_thread()
        return dict(captured=self.counters[0],capture_first_host_ns=self.counters[3],capture_last_host_ns=self.counters[4],overwritten_pending=self.counters[1],dropped_incoming=self.counters[2],drained_older=self.drained,forced_capture_termination=forced,capture_process_alive=alive,capture_exitcode=self.process.exitcode,startup_stages=self.stages,last_stage_code=self.stage.value)
