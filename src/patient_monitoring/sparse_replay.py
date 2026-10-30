"""Development-only sparse saved-target replay; no pose inference or policy changes."""
import argparse,copy,json
from fractions import Fraction
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from .supervised_dataset import sha,write,csv_rows,aligned_frames,make_window,CHANNELS
from .supervised_action import load_bundle,vectorize,transform,CLASSES,ORDER
from .pose_features import FeatureExtractor

def select_indices(times,period):
    # First observed frame at/after each anchored grid; skip missed slots, never duplicate.
    due=times[0];selected=[]
    for i,t in enumerate(times):
        if t>=due:
            selected.append(i)
            while due<=t:due+=period
    return selected

def sparse_features(rows,ids,video):
    extractor=FeatureExtractor(video);result=[];previous=None
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

def eligible_label(start,end,segments,bounds):
    for s in segments:
        if s['direct'] and bounds[s['lo']]<=start and end<=bounds[s['hi']]:return s['annotation']['activity_label']
    return None

def distribution(values):
    x=np.asarray(values,dtype=float)
    return dict(count=int(len(x)),median=float(np.median(x)) if len(x) else None,p10=float(np.quantile(x,.1)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None)

def run(dataset,modeldir,out):
    dataset=Path(dataset);out=Path(out);out.mkdir(parents=True,exist_ok=False);write(out/'summary.json',{'status':'incomplete'})
    hashes={};results=[]
    def checked(p,h=None):
        p=Path(p);actual=sha(p)
        if h is not None and actual!=h:raise ValueError('hash mismatch '+str(p))
        hashes[str(p)]=actual;return p
    try:
        manifest=json.loads((dataset/'manifest.json').read_text())
        for n,h in manifest['output_sha256'].items():checked(dataset/n,h)
        for n,h in manifest['input_sha256'].items():checked(n,h)
        frozen=json.loads((Path(modeldir)/'frozen_manifest.json').read_text());checked(Path(modeldir)/'frozen_manifest.json')
        for n,h in frozen['artifact_sha256'].items():checked(Path(modeldir)/n,h)
        model,prep,schema=load_bundle(modeldir)
        training=[json.loads(s) for s in (dataset/'windows.jsonl').read_text().splitlines()];train_x=vectorize(training,schema)
        write(out/'training_distribution.json',{c:distribution(train_x[np.isfinite(train_x[:,j]),j]) for j,c in enumerate(ORDER)})
        inventory=json.loads((dataset/'inventory.json').read_text())
        pools={str(p):[] for p in (.25,.5,.75,1.)}
        for video in inventory:
            vid=video['video_id'];prov=json.loads(checked(Path(video['features']).parent/'provenance.json').read_text());target=checked(prov['input_path'],prov['input_sha256']);rows=[json.loads(s) for s in target.read_text().splitlines()]
            bounds=aligned_frames(rows,csv_rows(checked(video['timeline'])))
            for period in (.25,.5,.75,1.):
                folder=out/vid/str(period);folder.mkdir(parents=True);ids=select_indices(bounds[:-1],Fraction(str(period)));features=sparse_features(rows,ids,vid);times=[bounds[i] for i in ids]
                with (folder/'features.jsonl').open('x') as f:
                    for row in features:f.write(json.dumps(row,allow_nan=False)+'\n')
                reasons=Counter();labels=Counter();confusion=Counter();available=0;scored=0;end=Fraction(0);windows=[];source_counts=[]
                with (folder/'decisions.jsonl').open('x') as f:
                    while end<=times[-1]:
                        start=end-1
                        value,reason=(None,'startup_less_than_one_second') if end<1 else make_window(features[:-1],times,start,end)
                        truth=eligible_label(start,end,video['segments'],bounds) if end>=1 else None
                        label=None
                        if value:
                            available+=1;windows.append(value);source_counts.append(len(value['frame_indices']))
                            with torch.no_grad():label=CLASSES[int(model(torch.from_numpy(transform(vectorize([value],schema),prep))).argmax(1)[0])]
                            labels[label]+=1
                        else:reasons[reason]+=1
                        if truth:scored+=1;confusion[truth+' -> '+(label or 'unknown')]+=1
                        f.write(json.dumps(dict(support_start=str(start),support_end=str(end),available=value is not None,reason=reason,label=label,eligible_truth=truth,model_inputs=value['model_inputs'] if value else None),allow_nan=False)+'\n');end+=Fraction(1,4)
                if windows:pools[str(period)].extend(windows)
                gaps=[float(b-a) for a,b in zip(times,times[1:])]
                results.append(dict(video_id=vid,period=period,selected_frames=len(ids),source_indices=ids,available=available,unavailable=sum(reasons.values()),reasons=dict(reasons),feature_pairs_valid=sum(r['temporal_pair']['valid'] for r in features),gap_seconds=distribution(gaps),gaps_over_half=sum(x>.5 for x in gaps),observations_per_available_window=distribution(source_counts),eligible_scored_windows=scored,confusion=dict(confusion),labels=dict(labels)))
        write(out/'per_video.json',results)
        distributions={}
        for period,windows in pools.items():
            x=vectorize(windows,schema) if windows else np.empty((0,28))
            distributions[period]={c:distribution(x[np.isfinite(x[:,j]),j]) for j,c in enumerate(ORDER)}
        write(out/'sparse_distribution.json',distributions)
        totals={str(p):dict(available=sum(r['available'] for r in results if r['period']==p),unavailable=sum(r['unavailable'] for r in results if r['period']==p),reasons=dict(sum((Counter(r['reasons']) for r in results if r['period']==p),Counter()))) for p in (.25,.5,.75,1.)}
        for p,h in hashes.items():
            if sha(p)!=h:raise ValueError('input changed '+p)
        write(out/'provenance.json',dict(input_hashes=hashes,code_sha256=sha(__file__),identity_policy='Saved target identity retained ONLY as sampling-isolation diagnostic, not sparse association validation; every intervening unresolved/reseed barrier preserved. No automatic reseed.',sampling='First source PTS at/after anchored grid; skip missed slots, never duplicate/interpolate. Windows end no later than last sampled PTS; unobserved tail not held.',annotation_policy='v005 direct eligible segments only; includes authorized assistant-selected walking interiors, not independently exact user boundaries. SE06 preliminary excluded.',scope='development, training-overlapping, no independent accuracy'))
        write(out/'summary.json',dict(status='complete',videos=len(inventory),totals=totals));return totals
    except BaseException as exc:write(out/'summary.json',dict(status='incomplete',error=repr(exc)));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();print(json.dumps(run(a.dataset,a.model_dir,a.output_dir)))
