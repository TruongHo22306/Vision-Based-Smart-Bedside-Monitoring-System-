"""Explicit experimental low-rate live engine; shared offline feature/window functions."""
import json,time,math
from pathlib import Path
from collections import deque,Counter
from fractions import Fraction
import numpy as np
import torch
from .low_rate_experiment import recompute,window,apply,ORDER,CLASSES
from .supervised_dataset import sha
from .live_demo import LiveEngine,association_gap,BASE
from .target_selection import TargetAssociator
from .video_inspection import TimelineAudit
PINS = {'config.json': '9dcea4c12ebed893d5c9c1a45f8d9f2b8e944f233990ca247b4e4a7061aba938', 'schema.json': '9036f55c1fae9cc1dc001821afee71edcf543288f35eaf7ca1555c577d666647', 'model_state.pt': '0144231665fb07a7227af977f6ab242015d9b8648eba0a72d67b934c361c5f14', 'preprocessing.json': '968cee2261155d468818fc0fe711ceed143e05717f4ebb39f339e2565d719e1c'}

def verify_bundle(folder):
    folder=Path(folder)
    for name,digest in PINS.items():
        if not (folder/name).is_file() or sha(folder/name)!=digest:raise ValueError('Frozen low-rate artifact hash mismatch: '+name)
    return dict(PINS)

def validate_schema(schema,prep):
    if schema.get('order')!=ORDER or schema.get('dimensions')!=35 or prep.get('order')!=ORDER or prep.get('classes')!=CLASSES:raise ValueError('Low-rate schema/class order mismatch')
    for k in ('medians','centers','scales'):
        a=np.asarray(prep[k]);
        if a.shape!=(14,) or not np.isfinite(a).all():raise ValueError('Invalid low-rate preprocessing '+k)
    if any(s<=0 for s in prep['scales']):raise ValueError('Invalid scales')

def load_low_rate(folder):
    verify_bundle(folder);folder=Path(folder)
    config=json.loads((folder/'config.json').read_text());schema=json.loads((folder/'schema.json').read_text());prep=json.loads((folder/'preprocessing.json').read_text());validate_schema(schema,prep)
    model=torch.nn.Linear(35,4,device='cpu',dtype=torch.float64)
    model.load_state_dict(torch.load(folder/'model_state.pt',map_location='cpu',weights_only=True),strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite model')
    model.eval();return model,prep,schema,config

class LowRateEngine(LiveEngine):
    classifier_mode='low-rate'
    def __init__(self,model,prep,schema,config,experimental_association_gap=None):
        validate_schema(schema,prep);self.config=config
        self.model,self.prep,self.schema=model,prep,schema
        self.association_gap=.5 if experimental_association_gap is None else association_gap(experimental_association_gap)
        self.decision_counts=Counter();self.unavailable_reasons=Counter();self.seed_events=0;self.reset_events=0;self.skipped_decisions=0;self.epoch=0
        self.retention=Fraction(str(config['window_seconds']+2*config['max_gap_seconds']+config['stride_seconds']))
        self.reset()
    def reset(self):
        if self.epoch:self.reset_events+=1
        self.epoch+=1;self.association=None;self.audit=TimelineAudit();self.selected_history=deque(maxlen=256);self.history=[]
        self.origin=None;self.last_host=None;self.next_end=Fraction(0);self.label=None;self.reason='explicit_seed_required';self.latest_step=None;self.display_support_host_ns=None
    def expire_display(self,host_now):
        if self.label is not None and (self.display_support_host_ns is None or (host_now-self.display_support_host_ns)/1e9>self.config['max_latest_age_seconds']):
            self.label=None;self.reason='display_support_expired'
    def step(self,people,width,height,stamp,seed=None):
        begin=time.perf_counter()
        if seed is not None:
            self.reset();self.seed_events+=1;self.association=TargetAssociator(0,seed,max_gap=self.association_gap)
        if self.last_host is not None and stamp<=self.last_host:raise ValueError('Nonincreasing host acquisition timestamp')
        self.last_host=stamp
        if self.association is None:return None,[],0.
        if self.origin is None:self.origin=stamp
        row=self.audit.observe(stamp-self.origin,BASE);row.update(source_path='live-webcam',stream_index=0,width=width,height=height,people=people,person_count=len(people),inference_status='complete')
        selected=self.association.step(row);self.selected_history.append(selected);now=Fraction(stamp-self.origin,1000000000)
        while len(self.selected_history)>2 and self.selected_history[1]['pts']*BASE<now-self.retention:self.selected_history.popleft()
        saved=list(self.selected_history)
        self.history=recompute(saved,list(range(len(saved))),'live',self.config['max_gap_seconds'])
        times=[r['pts']*BASE for r in saved]
        feat=self.history[-1];self.latest_step=dict(association={k:selected.get(k) for k in ('status','reason','elapsed_since_reference_seconds','reference_frame_index','best_iou','runner_up_iou','iou_separation','candidates','association_ended')},feature_temporal_pair=feat['temporal_pair'],mode='low-rate')
        stride=Fraction(str(self.config['stride_seconds']));decisions=[];missed=0
        while self.next_end<now-self.retention:self.next_end+=stride;missed+=1
        self.skipped_decisions+=missed
        if missed:decisions.append(dict(skipped_grid_decisions=missed,reason='processing_stall_no_retained_support',epoch=self.epoch,mode='low-rate'))
        while self.next_end<=now:
            end=self.next_end;self.next_end+=stride
            value,reason=window(self.history,times,end,self.config)
            obs=[t for t in times[:-1] if end-Fraction(str(self.config['window_seconds']))<=t<end]
            gaps=[float(b-a) for a,b in zip(times,times[1:]) if a<end and b>end-Fraction(str(self.config['window_seconds']))]
            d=dict(mode='low-rate',epoch=self.epoch,support_start=str(end-Fraction(str(self.config['window_seconds']))),support_end=str(end),decision_host_ns=self.origin+int(end*1000000000),computed_host_ns=time.monotonic_ns(),latest_arrived_host_ns=stamp,available=value is not None,reason=reason,label=None,observation_count=len(obs),observed_span=float(obs[-1]-obs[0]) if len(obs)>1 else None,latest_age=float(end-obs[-1]) if obs else None,max_support_gap=max(gaps) if gaps else None,model_sha256=PINS['model_state.pt'],config_sha256=PINS['config.json'])
            if value:
                x=apply(np.asarray([value['vector']],dtype=float),self.prep)
                with torch.no_grad():logits=self.model(torch.from_numpy(x))[0];scores=logits.softmax(0)
                if not torch.isfinite(logits).all():raise ValueError('Nonfinite classifier output')
                d.update(label=CLASSES[int(scores.argmax())],logits=logits.tolist(),scores_uncalibrated=scores.tolist(),vector=value['vector'],channels=value['channels'])
                self.display_support_host_ns=self.origin+int(obs[-1]*1000000000)
            else:self.display_support_host_ns=None
            self.label=d['label'];self.reason=reason;self.decision_counts['available' if value else 'unavailable']+=1
            if not value:self.unavailable_reasons[reason]+=1
            decisions.append(d)
        if selected['selected_person'] is None or (len(self.history)>1 and not feat['temporal_pair']['valid']):
            self.label=None;self.reason=selected['status'] if selected['selected_person'] is None else 'feature_continuity:'+str(feat['temporal_pair']['reason']);self.display_support_host_ns=None
        self.expire_display(stamp)
        return selected,decisions,time.perf_counter()-begin
