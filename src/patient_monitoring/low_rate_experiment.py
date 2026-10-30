"""Separate frozen-config low-rate development experiment; never changes live defaults."""
import argparse,copy,json,time
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import numpy as np
import torch
from .supervised_dataset import make_window,aligned_frames,csv_rows,CHANNELS,sha,write
from .supervised_action import load_bundle,vectorize,transform,CLASSES
from .sparse_replay import eligible_label
from .pose_features import FeatureExtractor
PIN='1cd7e002e3fe32a249ada07ef838fc056e6c99b71fb7a3fafe3a5edc8abfb1d9'
ORDER=[f'{k}.{s}' for k in CHANNELS for s in ('mean','std')]+[f'{k}.coverage' for k in CHANNELS]+[f'{k}.{s}_missing' for k in CHANNELS for s in ('mean','std')]

def select(times,cycle):
    due=times[0];ids=[];slot=0
    for i,t in enumerate(times):
        if t>=due:
            ids.append(i)
            while due<=t:due+=F(str(cycle[slot%len(cycle)]));slot+=1
    return ids

def recompute(rows,ids,video,gap):
    extractor=FeatureExtractor(video,max_time_gap=gap);result=[];previous=None
    for k,i in enumerate(ids):
        row=copy.deepcopy(rows[i]);row['source_frame_index']=i;row['frame_index']=k
        barrier=previous is not None and any(r.get('selected_person') is None or r.get('status')=='seeded' or r.get('track_break') for r in rows[previous+1:i+1])
        if barrier and result:
            # Conservative offline validity audit: never assign held support across a hidden loss.
            result[-1]['target_status']='unresolved_source_interval'
            result[-1]['hidden_barrier_in_following_interval']=True
        row['track_break']=bool(barrier)
        row['reference_frame_index']=k-1 if k else None
        f=extractor.step(row);f['source_frame_index']=i;f['skipped_source_track_barrier']=barrier;result.append(f);previous=i
    return result


def window(rows,times,end,c):
    start=end-F(str(c['window_seconds']))
    if start<0:return None,'startup'
    obs=[i for i,t in enumerate(times[:-1]) if start<=t<end]
    if len(obs)<c['min_observations']:return None,'insufficient_observations'
    if end-times[obs[-1]]>F(str(c['max_latest_age_seconds'])):return None,'stale_latest_sample'
    if times[obs[-1]]-times[obs[0]]<F(str(c['min_observed_span_seconds'])):return None,'insufficient_observed_span'
    value,reason=make_window(rows[:-1],times,start,end,max_gap=F(str(c['max_gap_seconds'])))
    if reason:return None,reason
    means=[];cov=[];flags=[];details={}
    for k,v in value['model_inputs'].items():
        mean_valid=v['valid'] and v['valid_observations']>=c['min_mean_observations'] and v['coverage']>=c['min_channel_coverage']
        std_valid=mean_valid and v['valid_observations']>=c['min_std_observations']
        means += [v['mean'] if mean_valid else None,v['std'] if std_valid else None]
        cov.append(v['coverage']);flags += [float(not mean_valid),float(not std_valid)]
        details[k]=dict(mean=means[-2],std=means[-1],coverage=v['coverage'],observations=v['valid_observations'],mean_valid=mean_valid,std_valid=std_valid)
    return dict(vector=means+cov+flags,channels=details,observation_count=len(obs),observed_span=float(times[obs[-1]]-times[obs[0]]),latest_age=float(end-times[obs[-1]]),source_indices=[rows[i]['source_frame_index'] for i in value['frame_indices']]),None

def fit(x):
    a=x[:,:14];fallback=np.isnan(a).all(0);med=np.array([0. if fallback[j] else np.nanmedian(a[:,j]) for j in range(14)])
    filled=np.where(np.isnan(a),med,a);center=filled.mean(0);scale=filled.std(0);constant=scale==0;scale[constant]=1
    return dict(medians=med.tolist(),fallback=fallback.tolist(),centers=center.tolist(),scales=scale.tolist(),constant=constant.tolist(),fit_rows=len(x),order=ORDER,classes=CLASSES,fit_scope='training only')

def apply(x,p):
    if p['order']!=ORDER:raise ValueError('schema mismatch')
    y=x.copy();y[:,:14]=(np.where(np.isnan(y[:,:14]),p['medians'],y[:,:14])-p['centers'])/p['scales']
    if not np.isfinite(y).all():raise ValueError('nonfinite input')
    return y

def run(config,out,train=False):
    config=Path(config);out=Path(out)
    if sha(config)!=PIN:raise ValueError('config hash mismatch')
    out.mkdir(parents=True,exist_ok=False);c=json.loads(config.read_text());write(out/'config.json',c);write(out/'schema.json',dict(order=ORDER,dimensions=35,units='degrees; image-fraction velocity/s; box image fractions; coverage/missing unitless',mean_std='duration weighted; std null below3 observations, mean null below2 or coverage.5'))
    if train:
        torch.set_num_threads(c['threads']);torch.set_num_interop_threads(c['interop_threads'])
    write(out/'run_info.json',dict(status='incomplete',training_executed=False));hashes={str(config):sha(config)}
    def check(p,h=None):
        p=Path(p);a=sha(p)
        if h and a!=h:raise ValueError('changed '+str(p))
        hashes[str(p)]=a;return p
    try:
        ds=Path(c['dataset']);manifest=json.loads(check(ds/'manifest.json').read_text())
        for n,h in manifest['output_sha256'].items():check(ds/n,h)
        for n,h in manifest['input_sha256'].items():check(n,h)
        frozen=json.loads(check(Path(c['model_reference'])/'frozen_manifest.json').read_text())
        for n,h in frozen['artifact_sha256'].items():check(Path(c['model_reference'])/n,h)
        strict_model,strict_prep,strict_schema=load_bundle(c['model_reference'])
        records=[];training=[];inventory=json.loads((ds/'inventory.json').read_text());supports={k:set() for k in CLASSES};seconds={k:{} for k in CLASSES}
        cycles={str(p):[p] for p in c['periods']};cycles['irregular']=c['irregular_cycle']
        for v in inventory:
            prov=json.loads(check(Path(v['features']).parent/'provenance.json').read_text());tp=check(prov['input_path'],prov['input_sha256']);rows=[json.loads(s) for s in tp.read_text().splitlines()];bounds=aligned_frames(rows,csv_rows(check(v['timeline'])))
            for cadence,cycle in cycles.items():
                ids=select(bounds[:-1],cycle);times=[bounds[i] for i in ids];low=recompute(rows,ids,v['video_id'],c['max_gap_seconds']);strict=recompute(rows,ids,v['video_id'],.5)
                folder=out/v['video_id']/cadence;folder.mkdir(parents=True)
                with (folder/'features.jsonl').open('x') as ff:
                    for r in low:ff.write(json.dumps(r,allow_nan=False)+'\n')
                end=F(0)
                while end<=times[-1]:
                    w,reason=window(low,times,end,c)
                    sw,sreason=(None,'startup') if end<1 else make_window(strict[:-1],times,end-1,end)
                    slabel=None
                    if sw:
                        with torch.no_grad():slabel=CLASSES[int(strict_model(torch.from_numpy(transform(vectorize([sw],strict_schema),strict_prep))).argmax())]
                    truth=eligible_label(end,end+F(1,1000000000),v['segments'],bounds)
                    label=eligible_label(end-3,end,v['segments'],bounds) if end>=3 else None
                    # No source observation from outside the eligible activity segment may train.
                    seg=next((s for s in v['segments'] if s['direct'] and bounds[s['lo']]<=end-3 and end<=bounds[s['hi']]),None) if label else None
                    eligible=bool(w and seg and min(w['source_indices'])>=seg['lo'])
                    rec=dict(video_id=v['video_id'],occasion=v['recording_occasion_group_id'],cadence=cadence,end=str(end),start=str(end-3),available=w is not None,reason=reason,window=w,strict_available=sw is not None,strict_reason=sreason,strict_label=slabel,truth_at_decision=truth,training_label=label if eligible else None,training_eligible=eligible)
                    records.append(rec)
                    if eligible:
                        training.append(rec);key=(v['video_id'],seg['annotation']['segment_id']);supports[label].add(key);seconds[label][key]=seg['duration']
                    end+=F(str(c['stride_seconds']))
        counts=Counter(r['training_label'] for r in training)
        support={k:dict(windows=counts[k],distinct_segments=len(supports[k]),unique_eligible_segment_seconds=sum(seconds[k].values()),occasions=sorted({r['occasion'] for r in training if r['training_label']==k})) for k in CLASSES}
        write(out/'support_before_training.json',support);print('SUPPORT BEFORE TRAINING',json.dumps(support),flush=True)
        write(out/'input_hashes.json',hashes)
        if any(counts[k]<c['minimum_training_windows_per_class'] or len(supports[k])<c['minimum_segments_per_class'] for k in CLASSES):
            write(out/'run_info.json',dict(status='blocked_insufficient_support',training_executed=False));return support
        if not train:
            write(out/'run_info.json',dict(status='prepared',training_executed=False));return support
        (out/'TRAINING_STARTED').write_text('One authorized fixed run; no restart')
        torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True)
        raw=np.asarray([r['window']['vector'] for r in training],dtype=float);prep=fit(raw);x=torch.from_numpy(apply(raw,prep));y=torch.tensor([CLASSES.index(r['training_label']) for r in training]);model=torch.nn.Linear(35,4,dtype=torch.float64);optimizer=torch.optim.Adam(model.parameters(),lr=c['lr']);losses=[];begin=time.perf_counter()
        for step in range(c['steps']):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y)+c['l2']/2*model.weight.square().sum()
            if not torch.isfinite(loss):raise ValueError('nonfinite training; no retry')
            loss.backward();optimizer.step();losses.append(float(loss))
            if (step+1)%100==0:print('step',step+1,'loss',losses[-1],flush=True)
        model.eval();torch.save(model.state_dict(),out/'model_state.pt');write(out/'preprocessing.json',prep);write(out/'loss_history.json',losses)
        reloaded=torch.nn.Linear(35,4,dtype=torch.float64);reloaded.load_state_dict(torch.load(out/'model_state.pt',weights_only=True));torch.testing.assert_close(model(x),reloaded(x),rtol=0,atol=0)
        summary={};confusions={};missing={}
        with (out/'decisions.jsonl').open('x') as f:
            for r in records:
                label=None
                if r['available']:
                    with torch.no_grad():label=CLASSES[int(model(torch.from_numpy(apply(np.asarray([r['window']['vector']],dtype=float),prep))).argmax())]
                r['label']=label;f.write(json.dumps(r,allow_nan=False)+'\n')
        for cadence in cycles:
            rs=[r for r in records if r['cadence']==cadence];known=[r for r in rs if r['truth_at_decision']]
            summary[cadence]=dict(decisions=len(rs),available=sum(r['available'] for r in rs),strict_available=sum(r['strict_available'] for r in rs),unavailable_reasons=dict(Counter(r['reason'] for r in rs if not r['available'])),known_decisions=len(known),correct=sum(r['label']==r['truth_at_decision'] for r in known),strict_correct=sum(r['strict_label']==r['truth_at_decision'] for r in known),unknown=sum(r['label'] is None for r in known),strict_unknown=sum(r['strict_label'] is None for r in known))
            confusions[cadence]={kind:dict(Counter(r['truth_at_decision']+' -> '+str(r[key] or 'unknown') for r in known)) for kind,key in [('low_rate','label'),('strict','strict_label')]}
            ws=[r['window'] for r in rs if r['available']];missing[cadence]={k:dict(windows=len(ws),mean_missing=sum(not w['channels'][k]['mean_valid'] for w in ws),std_missing=sum(not w['channels'][k]['std_valid'] for w in ws),median_coverage=float(np.median([w['channels'][k]['coverage'] for w in ws])) if ws else None) for k in CHANNELS}
        write(out/'comparison.json',summary);write(out/'confusions.json',confusions);write(out/'coverage_missingness.json',missing)
        for p,h in hashes.items():
            if sha(p)!=h:raise ValueError('protected input changed')
        write(out/'run_info.json',dict(status='complete',training_executed=True,training_runs=1,steps=c['steps'],training_rows=len(training),elapsed_seconds=time.perf_counter()-begin,reload_equal=True,initial_loss=losses[0],final_preupdate_loss=losses[-1],scope='development/resubstitution, no heldout split; strict comparison same timestamps, known label denominators include unknown/startup',config_sha256=PIN))
        write(out/'artifact_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()});print(json.dumps(summary));return summary
    except BaseException as e:write(out/'run_info.json',dict(status='incomplete',error=repr(e),training_started=(out/'TRAINING_STARTED').exists()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--train',action='store_true');a=p.parse_args();run(a.config,a.output_dir,a.train)
