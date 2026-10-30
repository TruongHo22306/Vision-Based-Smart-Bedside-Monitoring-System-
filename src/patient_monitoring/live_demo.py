"""Live webcam development demo; frozen classifier, explicit target, bounded capture."""
import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import time
from collections import deque,Counter
from fractions import Fraction
from pathlib import Path

from .camera_io import Capture,capture_loop,offer_latest,BACKENDS
from .live_diagnostics import image_size,prepare_settings,thread_snapshot,Performance,sample_telemetry
BASE=Fraction(1,1000000000)
_RUNTIME_LOADED=False


def load_runtime():
    # Windows spawn reimports __main__; keep model dependencies out of camera child.
    global cv2,np,torch,Image,PREDICT_CONFIG,MODEL_SHA256,EDGES,quality_people
    global TargetAssociator,FeatureExtractor,TimelineAudit,make_window,sha,write
    global read,load_bundle,vectorize,transform,CLASSES,_RUNTIME_LOADED
    if _RUNTIME_LOADED:return
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from .pose_preview import PREDICT_CONFIG,MODEL_SHA256,EDGES
    from .pose_extraction import quality_people
    from .target_selection import TargetAssociator
    from .pose_features import FeatureExtractor
    from .video_inspection import TimelineAudit
    from .supervised_dataset import make_window,sha,write
    from .supervised_action import read,load_bundle,vectorize,transform,CLASSES
    _RUNTIME_LOADED=True


class HalfWarningOnce(logging.Filter):
    """Deduplicate only this exact Ultralytics warning, retaining its first occurrence."""
    message = "'half' is deprecated and will be removed in the future. Use 'quantize' instead."
    def __init__(self):
        super().__init__(); self.seen = False; self.suppressed = 0
    def filter(self, record):
        if record.levelno != logging.WARNING or record.getMessage() != self.message:return True
        if self.seen:self.suppressed += 1;return False
        self.seen = True;return True


def verify_live_manifest(modeldir):
    # Keep original training manifest unchanged; deployment needs inference artifacts only.
    frozen=read(modeldir/'frozen_manifest.json')
    if frozen['status']!='frozen_complete':raise ValueError('Incomplete frozen classifier')
    required=('model_state.pt','preprocessing.json','feature_schema.json','config.json','model_hashes.json')
    for name in required:
        if sha(modeldir/name)!=frozen['artifact_sha256'][name]:raise ValueError('Frozen model hash mismatch: '+name)
    return frozen


class Log:
    def __init__(self,path,max_bytes=10000000):
        self.file=Path(path).open('x',encoding='utf-8');self.size=0;self.limit=max_bytes;self.omitted=0
    def add(self,record):
        text=json.dumps(record,allow_nan=False)+'\n';size=len(text.encode('utf-8'))
        if self.size+size>self.limit:self.omitted+=1;return
        self.file.write(text);self.file.flush();self.size+=size
    def close(self):self.file.close()


def aggregate_live(history,end):
    load_runtime()
    # Last row supplies the known next acquisition time only, never future features.
    past=[r for r in history if r['pts']*BASE<end]
    if not past or not history or history[-1]['pts']*BASE<end:return None,'incomplete_temporal_support'
    start=end-1
    first=past[0]['frame_index'];local=[]
    for r in past:
        copy=dict(r);pair=dict(r['temporal_pair']);prev=pair['previous_frame_index']
        pair['previous_frame_index']=None if prev is None else prev-first;copy['temporal_pair']=pair;local.append(copy)
    times=[r['pts']*BASE for r in past]
    following=next((r['pts']*BASE for r in history if r['pts']*BASE>=end),None)
    value,reason=make_window(local,times+[following],start,end)
    if value:
        value['processed_frame_indices']=[past[i]['frame_index'] for i in value['frame_indices']]
    return value,reason


def association_gap(value):
    try:n=float(value)
    except (ValueError,TypeError):raise argparse.ArgumentTypeError('Experimental association gap must be >0.5 and <=2 seconds')
    if not math.isfinite(n) or not .5<n<=2:raise argparse.ArgumentTypeError('Experimental association gap must be >0.5 and <=2 seconds')
    return n


class LiveEngine:
    def __init__(self,model,prep,schema,experimental_association_gap=None):
        load_runtime()
        self.association_gap=.5 if experimental_association_gap is None else association_gap(experimental_association_gap)
        self.decision_counts=Counter();self.unavailable_reasons=Counter();self.seed_events=0;self.reset_events=0;self.skipped_decisions=0
        self.model,self.prep,self.schema=model,prep,schema;self.epoch=0;self.reset()
    def reset(self):
        if self.epoch:self.reset_events+=1
        self.latest_step=None
        self.epoch+=1;self.association=None;self.extractor=FeatureExtractor('live');self.audit=TimelineAudit();self.history=deque(maxlen=256)
        self.origin=None;self.last_host=None;self.next_end=Fraction(0);self.label=None;self.reason='explicit_seed_required'
    def step(self,people,width,height,stamp,seed=None):
        if seed is not None:
            self.reset();self.seed_events+=1;self.association=TargetAssociator(0,seed,max_gap=self.association_gap)
        if self.last_host is not None and stamp<=self.last_host:raise ValueError('Nonincreasing host acquisition timestamp')
        self.last_host=stamp
        if self.association is None:return None,[],0.0
        if self.origin is None:self.origin=stamp
        r=self.audit.observe(stamp-self.origin,BASE);r.update(source_path='live-webcam',stream_index=0,width=width,height=height,people=people,person_count=len(people),inference_status='complete')
        selected=self.association.step(r);feat=self.extractor.step(selected);self.history.append(feat);now=Fraction(stamp-self.origin,1000000000)
        self.latest_step=dict(association={k:selected.get(k) for k in ('status','reason','elapsed_since_reference_seconds','reference_frame_index','best_iou','runner_up_iou','iou_separation','candidates','association_ended')},feature_temporal_pair=feat['temporal_pair'])
        while len(self.history)>2 and self.history[1]['pts']*BASE<now-2:self.history.popleft()
        decisions=[];begin=time.perf_counter()
        # If processing stalled, account for skipped grid slots without unbounded log/list growth.
        missed=0
        while self.next_end<now-2:self.next_end+=Fraction(1,4);missed+=1
        self.skipped_decisions+=missed
        if missed:decisions.append(dict(skipped_grid_decisions=missed,reason='processing_stall_no_retained_support',epoch=self.epoch))
        while self.next_end<=now:
            end=self.next_end;self.next_end+=Fraction(1,4)
            value,reason=(None,'startup_less_than_one_second') if end<1 else aggregate_live(self.history,end)
            d=dict(epoch=self.epoch,support_start=str(end-1),support_end=str(end),decision_host_ns=self.origin+int(end*1000000000),computed_host_ns=time.monotonic_ns(),available=value is not None,reason=reason,label=None)
            if value:
                x=transform(vectorize([value],self.schema),self.prep)
                with torch.no_grad():scores=self.model(torch.from_numpy(x)).softmax(1)[0]
                if not torch.isfinite(scores).all():raise ValueError('Nonfinite model output')
                d.update(label=CLASSES[int(scores.argmax())],scores_uncalibrated=scores.tolist(),coverage={k:v['coverage'] for k,v in value['model_inputs'].items()},missing=[k for k,v in value['model_inputs'].items() if not v['valid']])
            self.decision_counts['available' if value else 'unavailable']+=1
            if not value:self.unavailable_reasons[reason]+=1
            self.label=d['label'];self.reason=reason;decisions.append(d)
        invalid=selected['selected_person'] is None
        broken=feat['frame_index']>0 and not feat['temporal_pair']['valid']
        if invalid or broken:self.label=None;self.reason=selected['status'] if invalid else 'feature_continuity:'+str(feat['temporal_pair']['reason'])
        return selected,decisions,time.perf_counter()-begin


def paint(frame,people,selected,engine,rate,age):
    image=frame.copy()
    for p in people:
        x1,y1,x2,y2=map(int,p['box_xyxy']);cv2.rectangle(image,(x1,y1),(x2,y2),(150,150,150),1);cv2.putText(image,'D'+str(p['detection_index']),(x1,max(18,y1)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    if selected and selected['selected_person']:
        p=selected['selected_person'];x1,y1,x2,y2=map(int,p['box_xyxy']);cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,255),3);j=p['keypoints']
        for a,b in EDGES:
            if j[a]['passes_quality_gate'] and j[b]['passes_quality_gate']:cv2.line(image,(int(j[a]['x']),int(j[a]['y'])),(int(j[b]['x']),int(j[b]['y'])),(0,255,0),2)
    footer=np.zeros((150,image.shape[1],3),np.uint8)
    for i,text in enumerate(['EXPERIMENTAL DEVELOPMENT | NOT GROUND TRUTH',f"{engine.label or 'UNAVAILABLE'} | {engine.reason or 'available window'}",f'Processing {rate:.2f}/s | host frame age {age:.3f}s', '0-9: seed displayed D | r: reset | q/ESC: stop','No calibrated confidence; no sensor latency estimate']):
        cv2.putText(footer,text,(6,24+i*28),cv2.FONT_HERSHEY_SIMPLEX,min(.55,(image.shape[1]-12)/(max(len(text),1)*11)),(255,255,255),1)
    return np.vstack([image,footer])


def run(args):
    if args.camera<0 or (args.duration is not None and (not math.isfinite(args.duration) or args.duration<=0)):raise ValueError('Invalid camera/duration')
    experimental_gap=getattr(args,'experimental_association_gap',None)
    if experimental_gap is not None:experimental_gap=association_gap(experimental_gap)
    pose_backend=getattr(args,'pose_backend','pt')
    imgsz=image_size(getattr(args,'imgsz',640))
    weights=Path(args.weights);modeldir=Path(args.model_dir);out=Path(args.output_dir)
    if out.exists():raise FileExistsError(f'Output exists: {out}. Choose a NEW --output-dir; nothing overwritten.')
    out.mkdir(parents=True,exist_ok=False)
    settings_paths=prepare_settings(out)  # existing absolute writable dirs BEFORE any runtime/Ultralytics import
    load_runtime()
    from .live_pose import validate_pose,load_pose,ncnn_snapshot,package_versions
    predict_config={**PREDICT_CONFIG,'imgsz':imgsz}
    info=dict(status='incomplete',errors=[],processed=0,raw_video_recorded=False,camera=args.camera,backend=args.backend,pose_backend=pose_backend,association_mode='strict' if experimental_gap is None else 'experimental_low_rate',association_max_gap_seconds=.5 if experimental_gap is None else experimental_gap,feature_window_max_gap_seconds=.5,packages=package_versions(),pose_config=predict_config,settings_paths=settings_paths,thread_snapshots=[],telemetry_samples=0);write(out/'summary.json',info)
    warning_filter=None;pose_logger=None;capture=None;log=None;started=time.monotonic();ages=[];poses=[];class_times=[];statuses=Counter();last_stamp=None;last_picture=None
    engine=None
    performance=Performance();last_telemetry=None
    def threads(stage):
        snapshot=thread_snapshot(stage,torch);info['thread_snapshots'].append(snapshot);write(out/'summary.json',info)
    def telemetry(event):
        nonlocal last_telemetry
        sample=sample_telemetry();last_telemetry=time.monotonic();info['telemetry_samples']+=1
        info['latest_telemetry']=sample
        if log:log.add(dict(type='telemetry',event=event,monotonic_ns=time.monotonic_ns(),**sample))
    try:
        identity=validate_pose(weights,pose_backend,imgsz);info['pose_identity']=identity
        frozen=verify_live_manifest(modeldir)
        threads('before_classifier_load')
        model,prep,schema=load_bundle(modeldir)
        threads('after_classifier_load')
        engine=LiveEngine(model,prep,schema,experimental_gap)
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER
        pose_logger=LOGGER;warning_filter=HalfWarningOnce();pose_logger.addFilter(warning_filter)
        pose=load_pose(weights,pose_backend,out,predict_config)
        if pose_backend=='ncnn':info['pose_runtime_after_load']=ncnn_snapshot(pose,predict_config)
        threads('after_yolo_initialization')
        log=Log(out/'diagnostics.jsonl');telemetry('before_capture');capture=Capture(args.camera,args.backend);opened=time.monotonic();first_stamp=None
        while True:
            elapsed=time.monotonic()-opened
            if args.duration and elapsed>=args.duration:break
            if capture.status.value<0:raise RuntimeError('Camera open/read failure (capture process status=-1)')
            for stage in capture.diagnostics():log.add(dict(type='capture_stage',**stage))
            timeout=capture.timeout_reason()
            if timeout:raise TimeoutError(timeout+' (10s for current stage; inspect startup_stages)')
            item=capture.newest()
            if item is None:
                if last_stamp is not None and (time.monotonic_ns()-last_stamp)/1e9>.5:
                    engine.label=None;engine.reason='waiting_for_fresh_processed_frame'
                    if not args.no_display and last_picture is not None:
                        cv2.imshow('Patient monitoring development',paint(last_picture,[],None,engine,info['processed']/max(elapsed,.001),(time.monotonic_ns()-last_stamp)/1e9))
                if not args.no_display and cv2.waitKey(1)&255 in (27,ord('q')):break
                continue
            seq,stamp,frame=item;last_picture=frame
            if last_stamp is not None and stamp<=last_stamp:raise ValueError('Capture timestamp did not increase')
            interval=None if last_stamp is None else (stamp-last_stamp)/1e9;last_stamp=stamp
            if first_stamp is None:first_stamp=stamp
            before=time.monotonic_ns();start=time.perf_counter()
            source=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            predict_started=time.perf_counter()
            prediction=pose.predict(source=source,**predict_config)[0]
            predict_seconds=time.perf_counter()-predict_started
            people=quality_people(prediction,frame.shape[1],frame.shape[0]);pose_seconds=time.perf_counter()-start
            performance.observe(predict_seconds,interval)
            if info['processed']==0:
                if pose_backend=='ncnn':info['pose_runtime_after_first_predict']=ncnn_snapshot(pose,predict_config)
                threads('after_first_predict')
            selected,decisions,class_seconds=engine.step(people,frame.shape[1],frame.shape[0],stamp)
            info['processed']+=1;status=selected['status'] if selected else 'awaiting_explicit_seed';statuses[status]+=1
            age=(time.monotonic_ns()-stamp)/1e9;rate_elapsed=time.monotonic()-opened;rate=info['processed']/rate_elapsed if rate_elapsed>0 else 0.
            # Fixed-size measurement rings; summary contains total count plus latest256-sample statistics.
            ages.append(age);poses.append(pose_seconds);class_times.append(class_seconds)
            if len(ages)>256:ages.pop(0);poses.pop(0);class_times.pop(0)
            log.add(dict(type='frame',capture_sequence=seq,host_acquired_ns=stamp,processed_index=info['processed']-1,processed_interval_seconds=interval,tracking_diagnostics=engine.latest_step,exceeds_continuity_limit=None if interval is None else interval>.5,predict_seconds=predict_seconds,predict_phase='first' if info['processed']==1 else 'subsequent',ultralytics_speed_ms=getattr(prediction,'speed',None),epoch=engine.epoch,target_status=status,label=engine.label,reason=engine.reason,pose_seconds=pose_seconds,classifier_aggregation_seconds=class_seconds,age_before_pose_seconds=(before-stamp)/1e9,host_age_before_display_seconds=age,captured=capture.counters[0],overwritten=capture.counters[1],dropped_incoming=capture.counters[2],drained=capture.drained,people=len(people),selected_detection=None if selected is None else selected['selected_source_detection_index']))
            if info['processed']==1 or time.monotonic()-last_telemetry>=5:telemetry('after_first_predict' if info['processed']==1 else 'periodic_after_predict')
            for d in decisions:log.add(dict(type='decision',**d))
            if not args.no_display:
                cv2.imshow('Patient monitoring development',paint(frame,people,selected,engine,rate,age));key=cv2.waitKey(1)&255
                if key in (27,ord('q')):break
                if cv2.getWindowProperty('Patient monitoring development',cv2.WND_PROP_VISIBLE)<1:break
                if key==ord('r'):engine.reset();log.add(dict(type='explicit_reset',host_ns=time.monotonic_ns(),epoch=engine.epoch))
                elif ord('0')<=key<=ord('9'):
                    detection=key-ord('0')
                    # Seed the exact just-displayed image/detection, never a future frame.
                    selected,ds,_=engine.step(people,frame.shape[1],frame.shape[0],stamp,seed=detection)
                    log.add(dict(type='explicit_seed',host_acquired_ns=stamp,capture_sequence=seq,detection=detection,epoch=engine.epoch,status=selected['status'],tracking_diagnostics=engine.latest_step))
                    for d in ds:log.add(dict(type='decision',**d))
            if info['processed']%10==0:print('Processed',info['processed'],'rate',round(rate,2),'target',status,flush=True)
        if not info['processed']:raise RuntimeError('No camera frames processed within run duration')
        info.update(status='complete',acquisition_span_seconds=(last_stamp-first_stamp)/1e9,capture_rate_over_observed_span=None)
        verify_live_manifest(modeldir)
        if validate_pose(weights,pose_backend,imgsz)!=identity:raise ValueError('Pose artifacts changed')
        info.update(pose_config=predict_config,pose_sha256=identity['sha256'],frozen_manifest_sha256=sha(modeldir/'frozen_manifest.json'))
    except KeyboardInterrupt:info['status']='stopped_by_user'
    except BaseException as exc:info['status']='incomplete';info['errors'].append(repr(exc))
    finally:
        if capture:
            info.update(capture.close())
            span=(info['capture_last_host_ns']-info['capture_first_host_ns'])/1e9
            info['capture_rate_over_observed_span']=(info['captured']-1)/span if span>0 else None
            info['unprocessed_on_stop']=max(0,info['captured']-info['processed']-info['overwritten_pending']-info['dropped_incoming']-info['drained_older'])
        if log:telemetry('after_capture_cleanup')
        info.update(performance.summary())
        info['continuity_note']='Adjacent gaps >0.5s invalidate feature/window continuity. Target lifetime is separately configured; compare association reference age to association_max_gap_seconds.'
        if engine:info.update(decision_counts={k:engine.decision_counts[k] for k in ('available','unavailable')},unavailable_decision_reasons=dict(engine.unavailable_reasons),skipped_grid_decisions=engine.skipped_decisions,explicit_seed_events=engine.seed_events,history_reset_events=engine.reset_events,reset_count_note='Includes reseed-triggered resets, excludes construction')
        if not args.no_display:cv2.destroyAllWindows()
        if warning_filter:
            pose_logger.removeFilter(warning_filter);info['duplicate_half_warnings_suppressed']=warning_filter.suppressed
        if log:info['log_omitted_records']=log.omitted;info['log_bytes']=log.size;log.close()
        processing_elapsed=time.monotonic()-opened if capture else 0.
        info.update(elapsed_seconds=time.monotonic()-started,target_status_counts=dict(statuses),processing_rate_per_second=info['processed']/processing_elapsed if processing_elapsed>0 else None,
            latest_sample_count=len(ages),latest_mean_frame_host_age_seconds=float(np.mean(ages)) if ages else None,latest_mean_pose_seconds=float(np.mean(poses)) if poses else None,latest_mean_classifier_aggregation_seconds=float(np.mean(class_times)) if class_times else None,
            timestamp_policy='monotonic_ns immediately after capture.read; host acquisition, not verified sensor exposure',latency_policy='host read-return to pre-display only; excludes sensor/backend/display refresh; NOT sensor-to-screen latency',buffer_policy='one pending frame; history<=256processed frames and approximately2s; logs<=10MB; measurement rings256',source_sha256=sha(__file__))
        write(out/'summary.json',info)
    print(json.dumps(info),flush=True);return info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experimental-association-gap',type=association_gap,help='Explicit experimental association lifetime >0.5 to 2 seconds; feature/window limit stays0.5');p.add_argument('--camera',type=int,required=True);p.add_argument('--pose-backend',choices=('pt','ncnn'),default='pt');p.add_argument('--imgsz',type=image_size,default=640);p.add_argument('--backend',choices=BACKENDS,default='auto');p.add_argument('--weights',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--duration',type=float);p.add_argument('--no-display',action='store_true',help='smoke test only; remains unseeded, no automatic person selection')
    a=p.parse_args()
    try:result=run(a)
    except FileExistsError as exc:p.exit(2,str(exc)+'\n')
    if result['status']=='incomplete':raise SystemExit(1)


if __name__=='__main__':main()
