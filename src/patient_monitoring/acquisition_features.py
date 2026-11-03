"""Experimental acquisition-anchored features ONLY; no classifier or live-default change."""
import copy
import math
from fractions import Fraction as F
from .pose_features import FeatureExtractor
from .supervised_dataset import CHANNELS

VERSION='acquisition_features_v002'
CONFIG=dict(version=VERSION,window_seconds=3,stride_seconds=.25,max_gap_seconds=1.5,
            ttl_seconds=1.25,min_observations=3,min_observed_span_seconds=1.5,
            min_mean_observations=2,min_std_observations=3,min_channel_coverage=.5)
ORDER=[f'{k}.{s}' for k in CHANNELS for s in ('mean','std')]+[f'{k}.coverage' for k in CHANNELS]+[f'{k}.{s}_missing' for k in CHANNELS for s in ('mean','std')]

def stamp(row):
    return int(row['pts'])*F(row['time_base'])

def aggregate(rows, end, config=CONFIG):
    """Pure prefix function. Reject future rows; never borrow their features or duration."""
    end=F(end);start=end-F(str(config['window_seconds']))
    times=[stamp(r) for r in rows]
    if not rows or times[-1]!=end or any(t>end for t in times):raise ValueError('Window must end at latest actual acquisition')
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Nonincreasing timestamps')
    obs=[i for i,t in enumerate(times) if start<t<=end]
    relevant=[i for i,t in enumerate(times) if t>=start]
    reason=None
    if times[0]>start:reason='startup_less_than_three_seconds'
    elif any(not rows[i]['timestamp_valid'] or rows[i]['target_status'] not in ('seeded','matched') for i in relevant):reason='unresolved_target'
    elif any(b-a>F(str(config['max_gap_seconds'])) for a,b in zip(times,times[1:]) if b>start):reason='timestamp_gap'
    elif any(not rows[i]['temporal_pair']['valid'] for i in relevant if i and times[i]>start):reason='track_break'
    elif len(obs)<config['min_observations']:reason='insufficient_observations'
    elif times[obs[-1]]-times[obs[0]]<F(str(config['min_observed_span_seconds'])):reason='insufficient_observed_span'
    result=dict(version=VERSION,start=str(start),end=str(end),endpoint='(start,end]',observation_count=len(obs),
                temporal_available=reason is None,reason=reason,available=False,vector=None,channels={},expires_at=None,
                source_frame_indices=[rows[i]['source_frame_index'] for i in obs],contributing_source_indices=[])
    if reason:return result
    means=[];coverages=[];flags=[];deadlines=[];contributing=set()
    for name,(feature,component) in CHANNELS.items():
        velocity='velocity' in feature;samples=[]
        for i in obs:
            if velocity:
                if i==0 or times[i-1]<start or not rows[i]['temporal_pair']['valid']:continue
                weight=float(times[i]-times[i-1])
            else:weight=1.
            f=rows[i]['features'][feature]
            if not f['valid'] or f['value'] is None:continue
            value=f['value'] if component is None else f['value'][component]
            if not isinstance(value,(int,float)) or not math.isfinite(value):continue
            samples.append((i,value,weight))
        total=sum(w for _,_,w in samples)
        coverage=total/float(end-start) if velocity else len(samples)/len(obs)
        mean=sum(v*w for _,v,w in samples)/total if total else None
        std=math.sqrt(sum(w*(v-mean)**2 for _,v,w in samples)/total) if total else None
        support=times[samples[-1][0]] if samples else None
        age=None if support is None else float(end-support)
        fresh=age is not None and age<=config['ttl_seconds']
        mean_valid=len(samples)>=config['min_mean_observations'] and coverage>=config['min_channel_coverage'] and fresh
        std_valid=mean_valid and len(samples)>=config['min_std_observations']
        why=None if mean_valid else ('missing' if not samples else 'stale_support' if not fresh else 'insufficient_count_or_coverage')
        result['channels'][name]=dict(mean=mean if mean_valid else None,std=std if std_valid else None,raw_mean=mean,raw_std=std,
            coverage=coverage,coverage_unit='observed_interval_seconds/3s' if velocity else 'valid_observations/observations',
            observations=len(samples),weight_sum=total,weight_unit='seconds' if velocity else 'observations',
            latest_valid_timestamp=None if support is None else str(support),support_age_seconds=age,
            mean_valid=mean_valid,std_valid=std_valid,reason=why,std_reason=None if std_valid else why or 'fewer_than_three_observations')
        means.extend([mean if mean_valid else None,std if std_valid else None]);coverages.append(coverage);flags.extend([float(not mean_valid),float(not std_valid)])
        if mean_valid:
            deadlines.append(support+F(str(config['ttl_seconds'])))
            for i,_,_ in samples:
                contributing.add(rows[i]['source_frame_index'])
                if velocity:contributing.add(rows[i-1]['source_frame_index'])
    result.update(vector=means+coverages+flags,available=bool(deadlines),reason=None if deadlines else 'no_usable_channel',
                  expires_at=str(min(deadlines)) if deadlines else None,contributing_source_indices=sorted(contributing))
    return result

def freshness(record, now):
    if not record['available']:return False,record['reason']
    if F(now)<F(record['end']):return False,'before_acquisition'
    if F(now)>F(record['expires_at']):return False,'support_expired'
    return True,None

class AcquisitionFeatures:
    """Shared offline/live adapter for saved/streamed selected poses; no model loading."""
    def __init__(self,video_id,config=None):
        self.video_id=video_id;self.config=copy.deepcopy(config or CONFIG);self.sequence=0;self.last=None;self.epoch=0
        self.reset()
    def reset(self):
        self.epoch+=1;self.history=[];self.next_due=None;self.last=None
        self.extractor=FeatureExtractor(self.video_id,self.config['max_gap_seconds'])
    def step(self,target,hidden_barrier=False):
        row=copy.deepcopy(target);t=stamp(row)
        if self.last is not None and t<=self.last:raise ValueError('Nonincreasing acquisition')
        barrier=hidden_barrier or row.get('track_break') or row.get('status')=='seeded' or (self.last is not None and t-self.last>F(str(self.config['max_gap_seconds'])))
        if barrier:self.reset()
        self.last=t;row['source_frame_index']=row['frame_index'];row['frame_index']=self.sequence;row['reference_frame_index']=self.sequence-1 if self.history else None
        self.sequence+=1
        f=self.extractor.step(row);f['source_frame_index']=row['source_frame_index']
        if not f['timestamp_valid'] or f['target_status'] not in ('seeded','matched'):
            self.reset();self.last=t
            return f,dict(version=VERSION,end=str(t),available=False,temporal_available=False,reason='unresolved_or_invalid_timestamp',channels={},vector=None,expires_at=None,epoch=self.epoch)
        if self.history and not f['temporal_pair']['valid']:
            # Context/identity/reference discontinuity cannot retain previous window support.
            self.reset();self.last=t
            row['reference_frame_index']=None;f=self.extractor.step(row);f['source_frame_index']=row['source_frame_index']
        self.history.append(f)
        start=t-F(str(self.config['window_seconds']))
        while len(self.history)>2 and stamp(self.history[1])<=start:self.history.pop(0)
        if len(self.history)>256:self.history=self.history[-256:]
        if self.next_due is None:self.next_due=t
        if t<self.next_due:return f,None
        while self.next_due<=t:self.next_due+=F(str(self.config['stride_seconds']))
        record=aggregate(self.history,t,self.config);record.update(epoch=self.epoch,pts=row['pts'],time_base=row['time_base'],source_frame_index=row['source_frame_index'])
        return f,record
