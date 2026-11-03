"""Replay saved targets for experimental acquisition features; never train/infer pose."""
import argparse,json,statistics,time
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
from patient_monitoring.acquisition_features import AcquisitionFeatures,CONFIG,ORDER,stamp,aggregate,freshness
from patient_monitoring.low_rate_experiment import select
from patient_monitoring.supervised_dataset import sha,aligned_frames,csv_rows,write,CHANNELS

DATASET=Path('outputs/SE11_V002_annotations_dataset_v005_20261021/dataset')
def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write(out/'run_info.json',dict(status='incomplete',training=False));write(out/'config.json',CONFIG)
    write(out/'schema.json',dict(version=CONFIG['version'],dimensions=35,order=ORDER,compatible_with_frozen_model=False))
    hashes={}
    def check(p,expected=None):
        p=Path(p);h=sha(p)
        if expected and h!=expected:raise ValueError('Hash mismatch '+str(p))
        hashes[str(p)]=h;return p
    try:
        manifest=json.loads(check(DATASET/'manifest.json').read_text())
        for n,h in manifest['output_sha256'].items():check(DATASET/n,h)
        for n,h in manifest['input_sha256'].items():check(n,h)
        inventory=json.loads(check(DATASET/'inventory.json').read_text())
        stats={};support=Counter();occasions=defaultdict(Counter);segments=defaultdict(set);details={}
        cycles={'0.5':[.5],'0.75':[.75],'1.0':[1.],'irregular':[.5,.75,1.,1.25]}
        for cadence,cycle in cycles.items():
            count=Counter();why=Counter();channels={k:dict(coverage=[],age=[],mean_missing=0,std_missing=0) for k in CHANNELS}
            for v in inventory:
                prov=json.loads(check(Path(v['features']).parent/'provenance.json').read_text());tp=check(prov['input_path'],prov['input_sha256'])
                rows=[json.loads(x) for x in tp.read_text().splitlines()];bounds=aligned_frames(rows,csv_rows(check(v['timeline'])))
                ids=select(bounds[:-1],cycle);engine=AcquisitionFeatures(v['video_id']);offset=stamp(rows[0]);previous=None
                folder=out/v['video_id']/cadence;folder.mkdir(parents=True);vc=Counter()
                with (folder/'features.jsonl').open('x') as ff,(folder/'windows.jsonl').open('x') as wf:
                    for i in ids:
                        hidden=previous is not None and any(x.get('selected_person') is None or x.get('status')=='seeded' or x.get('track_break') for x in rows[previous+1:i+1])
                        f,r=engine.step(rows[i],hidden_barrier=hidden);previous=i
                        ff.write(json.dumps(f,allow_nan=False)+'\n')
                        if r is None:continue
                        r.update(video_id=v['video_id'],cadence=cadence,recording_occasion=v['recording_occasion_group_id'],training_label=None,training_eligible=False)
                        count['decisions']+=1;vc['decisions']+=1
                        if r['temporal_available']:
                            # Independently call the pure offline prefix API for parity.
                            expected=aggregate(engine.history,stamp(rows[i]))
                            assert expected['vector']==r['vector'] and expected['channels']==r['channels']
                            count['parity_checks']+=1
                            for k,c in r['channels'].items():
                                channels[k]['coverage'].append(c['coverage']);channels[k]['mean_missing']+=not c['mean_valid'];channels[k]['std_missing']+=not c['std_valid']
                                if c['support_age_seconds'] is not None:channels[k]['age'].append(c['support_age_seconds'])
                        if r['available']:
                            count['available']+=1;vc['available']+=1
                            start,end=F(r['start'])-offset,F(r['end'])-offset
                            seg=next((s for s in v['segments'] if s['direct'] and bounds[s['lo']]<=start and end<bounds[s['hi']] and all(s['lo']<=j<s['hi'] for j in r['contributing_source_indices'])),None)
                            if seg:
                                label=seg['annotation']['activity_label'];r.update(training_label=label,training_eligible=True,annotation_provenance='existing authorized eligible annotation; assistant-selected interiors retain existing provenance')
                                support[label]+=1;occasions[v['recording_occasion_group_id']][label]+=1;segments[label].add((v['video_id'],seg['annotation']['segment_id']));count['eligible_'+label]+=1
                        else:why[r['reason']]+=1;vc['unavailable']+=1
                        wf.write(json.dumps(r,allow_nan=False)+'\n')
                details[v['video_id']+'/'+cadence]=dict(vc)
            for c in channels.values():
                for key in ['coverage','age']:
                    vals=c[key];c[key+'_population']=len(vals);c[key+'_mean']=statistics.mean(vals) if vals else None;c[key+'_median']=statistics.median(vals) if vals else None;del c[key]
            stats[cadence]=dict(counts=dict(count),unavailable_reasons=dict(why),channels=channels)
            print(cadence,dict(count),flush=True)
        write(out/'summary.json',stats);write(out/'per_video.json',details)
        write(out/'training_support.json',dict(class_windows=dict(support),distinct_segments={k:len(x) for k,x in segments.items()},by_occasion={k:dict(x) for k,x in occasions.items()},note='Correlated cadences/windows; not independent samples; development only. No fit. Preliminary SE06 not promoted.'))
        for p,h in hashes.items():
            if sha(p)!=h:raise ValueError('Input changed '+p)
        write(out/'input_hashes.json',hashes);write(out/'run_info.json',dict(status='complete',training=False,pose_inference=False,config_sha256=sha(out/'config.json'),schema_sha256=sha(out/'schema.json'),code_sha256=sha(Path(__file__).parents[1]/'src/patient_monitoring/acquisition_features.py'),source_videos=len(inventory),irregular_schedule='simulated .5/.75/1/1.25s cycle; first real PTS at or after due; not measured Pi timestamps',identity='saved target identity; hidden gaps excluded, not a tracking benchmark'))
    except BaseException as e:
        write(out/'run_info.json',dict(status='incomplete',training=False,error=repr(e)));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.output_dir)
