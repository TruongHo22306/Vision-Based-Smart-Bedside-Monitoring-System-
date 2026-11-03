"""Opt-in low-rate main-thread UI; one inference owner and latest-only slots."""

import copy

import threading

import time

from collections import Counter

from types import SimpleNamespace





class LatestWorker:

    """One running call, one pending frame, one control, one result; epoch fenced."""

    def __init__(self, process, control, source=None, autostart=True):

        self.process, self.control, self.source = process, control, source

        self.cv = threading.Condition()

        self.epoch = 0

        self.pending = self.command = self.result = None

        self.stopping = False

        self.error = None

        self.counts = Counter()

        self.thread = threading.Thread(target=self._run, name='live-inference', daemon=True)

        if autostart:self.thread.start()



    def submit(self, frame):

        with self.cv:

            if self.stopping:return

            if self.pending is not None:self.counts['overwritten_pending_frames'] += 1

            self.pending = (self.epoch, frame)

            self.cv.notify()



    def reset(self, seed=None):

        with self.cv:

            self.epoch += 1

            if self.pending is not None:self.counts['frames_cleared_on_reset'] += 1

            if self.result is not None:self.counts['results_cleared_on_reset'] += 1

            self.pending = self.result = None

            self.command = (self.epoch, seed)

            self.cv.notify()

            return self.epoch



    def take(self):

        with self.cv:

            if self.error:raise RuntimeError('Inference worker failed: '+self.error)

            result, self.result = self.result, None

            return result



    def _run(self):

        try:

            while True:

                with self.cv:

                    self.cv.wait_for(lambda:self.stopping or self.command is not None or self.pending is not None, timeout=.02 if self.source else None)

                    if self.stopping:return

                    command = self.command is not None

                    job = self.command if command else self.pending

                    if command:self.command = None

                    else:self.pending = None

                if job is None:
                    # Capture has ONE consumer: the inference worker, not the UI.
                    with self.cv:epoch=self.epoch
                    fetch_started=time.monotonic_ns()
                    payload=self.source()
                    fetched=time.monotonic_ns()
                    if payload is None:continue
                    with self.cv:
                        self.counts['source_frames_received']+=1
                        if self.stopping:
                            self.counts['frames_cleared_on_stop']+=1;return
                        if epoch!=self.epoch or self.command is not None:
                            self.counts['frames_discarded_epoch_during_fetch']+=1;continue
                    self.fetch_timing=(fetch_started,fetched)
                else:
                    epoch,payload=job
                    self.fetch_timing=None

                result = self.control(epoch, payload) if command else self.process(epoch, payload)

                with self.cv:

                    if self.stopping or epoch != self.epoch:

                        self.counts['discarded_old_epoch_results'] += int(result is not None)

                    elif result is not None:

                        if self.result is not None:self.counts['overwritten_results'] += 1

                        result['published_host_ns']=time.monotonic_ns()
                        self.result = result

        except BaseException as exc:

            with self.cv:self.error = repr(exc);self.cv.notify_all()



    def close(self, timeout=10):

        with self.cv:

            if self.pending is not None:self.counts['frames_cleared_on_stop']+=1
            self.stopping = True;self.pending = self.command = self.result = None;self.cv.notify_all()

        self.thread.join(timeout)

        return not self.thread.is_alive()





class UIState:

    """Expiry uses original support, never arrival time. No history fallback."""

    def __init__(self, ttl):

        self.ttl_ns = round(ttl*1e9)

        self.epoch = 0;self.result = None;self.previous = None

        self.counts = Counter();self.reasons = Counter()

        self.active_ns = 0;self.max_age = 0.;self.age_sum = 0.;self.max_tick_gap = 0.

        self.expired_keys = set()  # only current result key is retained below



    def _account(self, now):

        if self.previous:

            t, label, deadline = self.previous

            self.max_tick_gap = max(self.max_tick_gap,(now-t)/1e9)

            if label:self.active_ns += max(0,min(now,deadline)-t)

        self.previous = None



    def reset(self, epoch, now):

        self._account(now);self.epoch=epoch;self.result=None;self.expired_keys.clear()



    def accept(self, result):

        if result['target_epoch'] != self.epoch:

            self.counts['stale_results_rejected'] += 1;return False

        if self.result and result['source_timestamp_ns'] < self.result['source_timestamp_ns']:

            self.counts['out_of_order_results_rejected'] += 1;return False

        self.result=result;self.expired_keys.clear();return True



    def tick(self, now, ui_enabled=True):

        self._account(now);r=self.result

        label=r['label'] if r else None

        reason=r['reason'] if r else 'explicit_seed_required_or_waiting'

        support=r.get('latest_support_host_ns') if r else None

        deadline=None if support is None else support+self.ttl_ns

        if label is not None and (deadline is None or now>deadline):

            label=None;reason='display_support_expired'

            key=(r['target_epoch'],r['frame_id'],r['completed_host_ns'])

            if key not in self.expired_keys:self.counts['support_expired_results']+=1;self.expired_keys.add(key)

        image_age=None if r is None else (now-r['source_timestamp_ns'])/1e9

        submitted=bool(ui_enabled and label is not None)

        self.counts['ui_updates']+=1;self.counts['label_submission_updates']+=int(submitted)

        absent=None if submitted else ('ui_disabled' if label is not None else reason or 'unavailable')

        if absent:self.reasons[absent]+=1

        if image_age is not None:

            self.counts['image_updates']+=1;self.age_sum+=image_age;self.max_age=max(self.max_age,image_age)

        self.previous=(now,label if submitted else None,deadline)

        return dict(type='ui_update',target_epoch=self.epoch,frame_id=None if r is None else r['frame_id'],

                    source_timestamp_ns=None if r is None else r['source_timestamp_ns'],

                    completed_host_ns=None if r is None else r['completed_host_ns'],assessed_host_ns=now,
                    published_host_ns=None if r is None else r.get('published_host_ns'),ui_received_host_ns=None if r is None else r.get('ui_received_host_ns'),

                    decision_host_ns=None if r is None else r.get('decision_host_ns'),

                    classifier_computed_host_ns=None if r is None else r.get('classifier_computed_host_ns'),

                    latest_support_host_ns=support,support_age_seconds=None if support is None else (now-support)/1e9,

                    label=label,reason=reason,label_submitted_to_ui=submitted,non_display_reason=absent,image_age_seconds=image_age)



    def summary(self, now):

        self._account(now)

        return dict(**self.counts,non_display_reasons=dict(self.reasons),fresh_label_submitted_duration_seconds=self.active_ns/1e9,

                    mean_processed_image_age_seconds=self.age_sum/self.counts['image_updates'] if self.counts['image_updates'] else None,

                    max_processed_image_age_seconds=self.max_age,max_ui_tick_gap_seconds=self.max_tick_gap,

                    duration_policy='Monotonic interval after labeled UI updates, clipped at original support deadline/reset/next update; not observed screen dwell or human perception.')





def async_capture_accounting(info):
    """Results dropped are not raw frames dropped; do not subtract them twice."""
    counts=info.get('async_queue_counts',{})
    legacy=info['captured']-info['processed']-info['overwritten_pending']-info['dropped_incoming']-info['drained_older']
    dropped=sum(counts.get(k,0) for k in ('overwritten_pending_frames','frames_cleared_on_reset','frames_cleared_on_stop','frames_discarded_epoch_during_fetch'))
    residual=legacy-dropped
    return dict(legacy_unprocessed_residual=legacy,async_frames_discarded_before_processing=dropped,
                unprocessed_on_stop=max(0,residual),capture_accounting_residual=residual,
                capture_accounting_note='Residual after all counted raw-frame drops; not a growing backlog. Negative residual is an accounting anomaly. Result overwrites/epoch discards already processed are not subtracted.')


def run_loop(live,args,capture,pose,engine,predict_config,info,log,performance,telemetry,threads,opened):

    if threading.current_thread() is not threading.main_thread():raise RuntimeError('UI must run on main thread')

    state=UIState(engine.config['max_latest_age_seconds']);display=live.DisplaySummary()

    stats=dict(first_stamp=None,last_stamp=None);statuses=Counter();last_telemetry=0.



    def snapshot(epoch,item,people,selected,decisions,seconds,pose_seconds,predict_seconds):

        seq,stamp,frame=item

        return dict(target_epoch=epoch,frame_id=seq,source_timestamp_ns=stamp,completed_host_ns=time.monotonic_ns(),

                    frame=frame,people=people,selected=copy.deepcopy(selected),label=engine.label,reason=engine.reason,

                    latest_support_host_ns=engine.display_support_host_ns,decision_host_ns=engine.display_decision_host_ns,

                    classifier_computed_host_ns=engine.display_computed_host_ns,classifier_aggregation_seconds=seconds,

                    pose_seconds=pose_seconds,predict_seconds=predict_seconds)



    def process(epoch,item):

        nonlocal last_telemetry

        seq,stamp,frame=item;before=time.monotonic_ns();start=time.perf_counter()

        previous=stats['last_stamp'];interval=None if previous is None else (stamp-previous)/1e9

        if previous is not None and stamp<=previous:raise ValueError('Capture timestamp did not increase')

        stats['last_stamp']=stamp

        if stats['first_stamp'] is None:stats['first_stamp']=stamp

        source=live.Image.fromarray(live.cv2.cvtColor(frame,live.cv2.COLOR_BGR2RGB));begin=time.perf_counter()

        predict_started_host_ns=time.monotonic_ns()
        prediction=pose.predict(source=source,**predict_config)[0];predict_seconds=time.perf_counter()-begin
        predict_completed_host_ns=time.monotonic_ns()

        people=live.quality_people(prediction,frame.shape[1],frame.shape[0]);pose_seconds=time.perf_counter()-start

        performance.observe(predict_seconds,interval)

        if info['processed']==0:

            if args.pose_backend=='ncnn':

                from .live_pose import ncnn_snapshot

                info['pose_runtime_after_first_predict']=ncnn_snapshot(pose,predict_config)

            threads('after_first_predict')

        classifier_started_host_ns=time.monotonic_ns()
        selected,decisions,seconds=engine.step(people,frame.shape[1],frame.shape[0],stamp)
        classifier_completed_host_ns=time.monotonic_ns()

        info['processed']+=1;status=selected['status'] if selected else 'awaiting_explicit_seed';statuses[status]+=1

        result=snapshot(epoch,item,people,selected,decisions,seconds,pose_seconds,predict_seconds)

        log.add(dict(type='frame',mode='low-rate',target_epoch=epoch,epoch=engine.epoch,capture_sequence=seq,

                     host_acquired_ns=stamp,completed_host_ns=result['completed_host_ns'],processed_index=info['processed']-1,
                     process_started_host_ns=before,capture_fetch_started_host_ns=worker.fetch_timing[0] if worker.fetch_timing else None,capture_fetch_completed_host_ns=worker.fetch_timing[1] if worker.fetch_timing else None,
                     predict_started_host_ns=predict_started_host_ns,predict_completed_host_ns=predict_completed_host_ns,classifier_started_host_ns=classifier_started_host_ns,classifier_completed_host_ns=classifier_completed_host_ns,

                     processed_interval_seconds=interval,tracking_diagnostics=engine.latest_step,target_status=status,

                     label=engine.label,reason=engine.reason,label_scope='worker result, not UI eligibility',

                     predict_seconds=predict_seconds,pose_seconds=pose_seconds,classifier_aggregation_seconds=seconds,

                     age_before_pose_seconds=(before-stamp)/1e9,host_age_at_completion_seconds=(result['completed_host_ns']-stamp)/1e9))

        for d in decisions:log.add(dict(type='decision',target_epoch=epoch,frame_id=seq,**d))

        if info['processed']==1 or time.monotonic()-last_telemetry>=5:

            telemetry('async_after_predict');last_telemetry=time.monotonic()

        if info['processed']%10==0:print('Processed',info['processed'],flush=True)

        return result



    def control(epoch,seed):

        if seed is None:engine.reset();return None

        r,index=seed;frame=r['frame'];stamp=r['source_timestamp_ns']

        selected,ds,seconds=engine.step(r['people'],frame.shape[1],frame.shape[0],stamp,seed=index)

        log.add(dict(type='explicit_seed_applied',target_epoch=epoch,epoch=engine.epoch,frame_id=r['frame_id'],host_acquired_ns=stamp,detection=index,status=selected['status']))

        for d in ds:log.add(dict(type='decision',target_epoch=epoch,frame_id=r['frame_id'],**d))

        return snapshot(epoch,(r['frame_id'],stamp,frame),r['people'],selected,ds,seconds,0.,0.)



    worker=LatestWorker(process,control,source=lambda:capture.newest(),autostart=False);window_created=False
    worker.thread.start()

    try:

        while not args.duration or time.monotonic()-opened<args.duration:

            if capture.status.value<0:raise RuntimeError('Camera open/read failure')

            timeout=capture.timeout_reason()

            if timeout:raise TimeoutError(timeout)

            for stage in capture.diagnostics():log.add(dict(type='capture_stage',**stage))

            loop_started_host_ns=time.monotonic_ns()
            result=worker.take()
            if result is not None:result['ui_received_host_ns']=time.monotonic_ns()
            received=result is not None and state.accept(result)
            now=time.monotonic_ns();update=state.tick(now,not args.no_display)
            assessment=None
            if received:
                proxy=SimpleNamespace(label=update['label'],reason=update['reason'],display_support_host_ns=result['latest_support_host_ns'],display_decision_host_ns=result['decision_host_ns'],display_computed_host_ns=result['classifier_computed_host_ns'],expire_display=lambda now:None)
                assessment=display.observe(proxy,now,not args.no_display)
                assessment.update(frame_id=result['frame_id'],target_epoch=result['target_epoch'],scope='first UI assessment of received result; repeated UI ticks logged separately')
            key=-1
            render_started_host_ns=time.monotonic_ns()

            if not args.no_display:

                r=state.result

                if r is not None:

                    proxy=SimpleNamespace(classifier_mode='low-rate',label=update['label'],reason=update['reason'])

                    image=live.paint(r['frame'],r['people'],r['selected'],proxy,info['processed']/max(.001,time.monotonic()-opened),update['image_age_seconds'])

                    live.cv2.putText(image,f"PROCESSED FRAME {r['frame_id']} | image age {update['image_age_seconds']:.3f}s",(5,18),live.cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)

                    live.cv2.imshow('Patient monitoring development',image)

                else:
                    # Also clear the old picture/label immediately after seed/reset.
                    image=live.np.zeros((240,640,3),dtype=live.np.uint8)
                    live.cv2.putText(image,'UNAVAILABLE: waiting for processed frame',(8,30),live.cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    live.cv2.imshow('Patient monitoring development',image)
                window_created=True
                render_completed_host_ns=time.monotonic_ns()
                key=live.cv2.waitKey(1)&255

            update.update(loop_started_host_ns=loop_started_host_ns,render_started_host_ns=render_started_host_ns,render_completed_host_ns=render_completed_host_ns if not args.no_display else None,ui_events_completed_host_ns=time.monotonic_ns())
            log_started=time.monotonic_ns()
            log.add(update)
            log.add(dict(type='ui_log_timing',assessed_host_ns=now,logging_seconds=(time.monotonic_ns()-log_started)/1e9))
            if assessment is not None:log.add(assessment)

            if key in (27,ord('q')):break

            if not args.no_display and window_created and live.cv2.getWindowProperty('Patient monitoring development',live.cv2.WND_PROP_VISIBLE)<1:break

            if key==ord('r'):

                epoch=worker.reset();state.reset(epoch,time.monotonic_ns());log.add(dict(type='explicit_reset',target_epoch=epoch,host_ns=time.monotonic_ns()))

            elif ord('0')<=key<=ord('9') and state.result is not None:

                index=key-ord('0');r=state.result

                if any(p['detection_index']==index for p in r['people']):

                    epoch=worker.reset((r,index));state.reset(epoch,time.monotonic_ns());log.add(dict(type='explicit_seed_requested',target_epoch=epoch,frame_id=r['frame_id'],host_acquired_ns=r['source_timestamp_ns'],detection=index))

                else:log.add(dict(type='seed_rejected',reason='detection_not_on_displayed_frame',frame_id=r['frame_id'],detection=index))

            time.sleep(.02)

    finally:

        ui_closed_at=time.monotonic_ns()

        if not args.no_display:live.cv2.destroyAllWindows()

        clean=worker.close()

        info.update(async_worker_stopped=clean,async_queue_counts=dict(worker.counts),async_ui=state.summary(ui_closed_at),display_summary=display.summary(),target_status_counts=dict(statuses))

        if not clean:raise RuntimeError('Inference worker did not stop within 10s; run incomplete; external process timeout required')

        if worker.error:raise RuntimeError('Inference worker failed: '+worker.error)

    return stats

