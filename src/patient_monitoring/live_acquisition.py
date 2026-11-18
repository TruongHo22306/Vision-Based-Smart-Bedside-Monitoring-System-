"""Opt-in acquisition-v002; frozen model and shared causal feature adapter."""
import copy,json,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import torch
from .acquisition_features import AcquisitionFeatures,CONFIG,VERSION,freshness
from .acquisition_action import load_bundle,vectorize
from .low_rate_experiment import apply,CLASSES
from .live_low_rate import LowRateEngine
from .live_demo import BASE
from .target_selection import TargetAssociator
from .supervised_dataset import sha

PINS = {'model_state.pt': 'c9d4eb488d3f197ee98101e2f19d5b30e99871c25a6d237d37c9b0bc7329c29f', 'preprocessing.json': '39d358c8c9bd3ef9904fed2cd01457d5e54e8c948703ecd4c10bde363c71e8c5', 'feature_schema.json': '829c9b69fa2d65cb79f382a5927c6b960eaa807f8e70240034af1985e97bb8e9', 'experiment_config.json': 'cda17dd4c70c1e10a424d951101c025142a22a31dd5fd6d7bc05f3873a36eb0d'}
MANIFEST_SHA='91a4a4c82c439001445a5c53fb0d70acbd2d580b45c7607c15b3bab5e0207284'
FEATURE_CODE_SHA='ba00e7cee21271af94eff4430be31628f9d8b85f56ad07b13fe11875ab2cc293'

def verify_bundle(folder):
    folder=Path(folder)
    for n,h in dict(PINS,**{'frozen_manifest.json':MANIFEST_SHA}).items():
        if sha(folder/n)!=h:raise ValueError('Acquisition frozen hash mismatch: '+n)
    if sha(Path(__file__).with_name('acquisition_features.py'))!=FEATURE_CODE_SHA:raise ValueError('Acquisition feature implementation changed')
    return dict(PINS)

def load_acquisition(folder):
    verify_bundle(folder)
    model,prep,schema=load_bundle(folder,expected_version=VERSION)
    for k in ('medians','centers','scales'):
        a=np.asarray(prep[k])
        if a.shape!=(14,) or not np.isfinite(a).all():raise ValueError('Invalid preprocessing')
    if any(x<=0 for x in prep['scales']) or any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Invalid model/scales')
    # Frozen CONFIG shared with offline. Alias only for existing async UI interface.
    config=copy.deepcopy(CONFIG);config['max_latest_age_seconds']=config['ttl_seconds']
    return model,prep,schema,config

class AcquisitionEngine(LowRateEngine):
    classifier_mode='acquisition-v002'
    def reset(self):
        super().reset()
        self.features=AcquisitionFeatures('live',CONFIG)
        self.current_record=None;self.channel_support={};self.display_deadline_host_ns=None
    def expire_display(self,host_now):
        if self.label is not None:
            ok,_=freshness(self.current_record,F(host_now-self.origin,1000000000))
            if not ok:self.label=None;self.reason='display_support_expired'
    def step(self,people,width,height,stamp,seed=None):
        begin=time.perf_counter()
        if seed is not None:
            self.reset();self.seed_events+=1;self.association=TargetAssociator(0,seed,max_gap=self.association_gap)
        if self.last_host is not None and stamp<=self.last_host:raise ValueError('Nonincreasing host acquisition timestamp')
        self.last_host=stamp
        if self.association is None:return None,[],0.
        if self.origin is None:self.origin=stamp
        row=self.audit.observe(stamp-self.origin,BASE)
        row.update(source_path='live-webcam',stream_index=0,width=width,height=height,people=people,person_count=len(people),inference_status='complete')
        selected=self.association.step(row)
        feat,record=self.features.step(selected)
        self.history=self.features.history
        self.latest_step=dict(mode=self.classifier_mode,association={k:selected.get(k) for k in ('status','reason','elapsed_since_reference_seconds','reference_frame_index','best_iou','runner_up_iou','iou_separation','candidates','association_ended')},feature_temporal_pair=feat['temporal_pair'])
        if record is None:
            self.expire_display(stamp)
            return selected,[],time.perf_counter()-begin
        self.current_record=record;self.label=None;self.reason=record['reason']
        self.channel_support={}
        for k,c in record['channels'].items():
            support=None if c['latest_valid_timestamp'] is None else self.origin+int(F(c['latest_valid_timestamp'])*1000000000)
            self.channel_support[k]=dict(support_host_ns=support,deadline_host_ns=None if support is None else support+round(CONFIG['ttl_seconds']*1e9),used=c['mean_valid'],mean_valid=c['mean_valid'],std_valid=c['std_valid'],reason=c['reason'],std_reason=c['std_reason'])
        used=[c['support_host_ns'] for c in self.channel_support.values() if c['used']]
        self.display_support_host_ns=min(used) if used else None
        self.display_deadline_host_ns=None if record['expires_at'] is None else self.origin+int(F(record['expires_at'])*1000000000)
        d=dict(mode=self.classifier_mode,epoch=self.epoch,feature_version=VERSION,support_start=record.get('start'),support_end=record['end'],decision_host_ns=stamp,compute_started_host_ns=time.monotonic_ns(),latest_arrived_host_ns=stamp,available=record['available'],reason=record['reason'],label=None,observation_count=record.get('observation_count'),channels=record['channels'],channel_support=copy.deepcopy(self.channel_support),support_deadline_host_ns=self.display_deadline_host_ns,latest_support_host_ns=self.display_support_host_ns,model_sha256=PINS['model_state.pt'],config_sha256=PINS['experiment_config.json'],vector=record['vector'])
        if record['available']:
            with torch.no_grad():logits=self.model(torch.from_numpy(apply(vectorize([record],self.schema),self.prep)))[0]
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite classifier output')
            self.label=CLASSES[int(logits.argmax())]
            d.update(label=self.label,logits=logits.tolist(),scores_uncalibrated=logits.softmax(0).tolist())
        d['computed_host_ns']=time.monotonic_ns()
        self.display_decision_host_ns=stamp;self.display_computed_host_ns=d['computed_host_ns']
        self.decision_counts['available' if record['available'] else 'unavailable']+=1
        if not record['available']:self.unavailable_reasons[record['reason']]+=1
        return selected,[d],time.perf_counter()-begin
