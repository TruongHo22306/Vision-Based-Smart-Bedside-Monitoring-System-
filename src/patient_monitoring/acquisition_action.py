"""One fixed development training run for acquisition_features_v002, not live wiring."""
import argparse,json,time,copy
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
from .acquisition_features import VERSION,ORDER,CONFIG,AcquisitionFeatures,aggregate,stamp
from .low_rate_experiment import select,fit,apply
from .supervised_dataset import sha,write,csv_rows,aligned_frames,segments,CLASSES

ROOT=Path('outputs/acquisition_features_v002_20261103')
DATASET=Path('outputs/SE11_V002_annotations_dataset_v005_20261021/dataset')
CYCLES={'0.5':[.5],'0.75':[.75],'1.0':[1.],'irregular':[.5,.75,1.,1.25]}

def validate_schema(s):
    if s.get('version')!=VERSION or s.get('dimensions')!=len(ORDER) or s.get('order')!=ORDER:raise ValueError('Feature version/schema mismatch')

def vectorize(records,schema):
    validate_schema(schema);vectors=[]
    for r in records:
        if r.get('version')!=VERSION or not r.get('available') or not isinstance(r.get('vector'),list) or len(r['vector'])!=schema['dimensions']:raise ValueError('Invalid feature row/version')
        a=np.asarray(r['vector'],dtype=float)
        if np.isinf(a).any() or not np.isfinite(a[14:]).all() or np.any((a[14:21]<0)|(a[14:21]>1+1e-9)):raise ValueError('Invalid feature values')
        if not np.array_equal(a[21:],np.isnan(a[:14]).astype(float)):raise ValueError('Missing flags disagree with null values')
        vectors.append(a)
    return np.asarray(vectors,dtype=np.float64)

def audit_inputs(root=ROOT):
    root=Path(root);hashes={}
    def check(p,expected=None):
        p=Path(p);h=sha(p)
        if expected and h!=expected:raise ValueError('Hash mismatch: '+str(p))
        hashes[str(p)]=h;return p
    outputs=json.loads(check(root/'output_hashes.json').read_text(encoding='utf-8-sig'))
    for n,h in outputs.items():check(root/n,h)
    for n,h in json.loads(check(root/'input_hashes.json').read_text(encoding='utf-8-sig')).items():check(n,h)
    info=json.loads((root/'run_info.json').read_text(encoding='utf-8-sig'));check(Path(__file__).with_name('acquisition_features.py'),info['code_sha256'])
    if info['status']!='complete' or json.loads((root/'config.json').read_text(encoding='utf-8-sig'))!=CONFIG:raise ValueError('Incomplete/changed feature replay')
    schema=json.loads((root/'schema.json').read_text(encoding='utf-8-sig'));validate_schema(schema)
    inventory=json.loads(check(DATASET/'inventory.json').read_text(encoding='utf-8-sig'));all_rows=[];stream_rows=[];checks=Counter();support=Counter();by_cadence=defaultdict(Counter);provenance=[]
    for v in inventory:
        prov=json.loads(check(Path(v['features']).parent/'provenance.json').read_text(encoding='utf-8-sig'));target=check(prov['input_path'],prov['input_sha256']);rows=[json.loads(l) for l in target.read_text(encoding='utf-8-sig').splitlines()]
        bounds=aligned_frames(rows,csv_rows(check(v['timeline'])));offset=stamp(rows[0]);annotations=[]
        for ap in v['annotations']:
            annotations += [dict(a,_direct_confirmation_source=ap in v.get('direct_confirmation_sources',[])) for a in csv_rows(check(ap)) if a['video_id']==v['video_id']]
        ss=segments(annotations,bounds)
        if ss!=v['segments']:raise ValueError('Annotation inventory mismatch')
        for cadence,cycle in CYCLES.items():
            folder=root/v['video_id']/cadence;saved=[json.loads(l) for l in (folder/'windows.jsonl').read_text(encoding='utf-8-sig').splitlines()];saved_features=[json.loads(l) for l in (folder/'features.jsonl').read_text(encoding='utf-8-sig').splitlines()]
            engine=AcquisitionFeatures(v['video_id']);previous=None;j=0
            ids=select(bounds[:-1],cycle)
            if len(ids)!=len(saved_features):raise ValueError('Selected frame count mismatch')
            for fi,i in enumerate(ids):
                hidden=previous is not None and any(x.get('selected_person') is None or x.get('status')=='seeded' or x.get('track_break') for x in rows[previous+1:i+1]);previous=i
                f,r=engine.step(rows[i],hidden_barrier=hidden)
                if f!=saved_features[fi]:raise ValueError('Recomputed feature mismatch')
                checks['source_feature_rows']+=1
                if r is None:continue
                s=saved[j];j+=1
                if any(s.get(k)!=val for k,val in r.items()):raise ValueError('Offline/stream feature record mismatch')
                if r['temporal_available']:
                    pure=aggregate(engine.history,stamp(rows[i]))
                    if pure['vector']!=s['vector'] or pure['channels']!=s['channels']:raise ValueError('Pure aggregate parity mismatch')
                    checks['pure_prefix_checks']+=1
                end=F(s['end'])-offset;seg=None
                if s['available']:
                    start=F(s['start'])-offset
                    seg=next((a for a in ss if a['direct'] and bounds[a['lo']]<=start and end<bounds[a['hi']] and all(a['lo']<=k<a['hi'] for k in s['contributing_source_indices'])),None)
                label=seg['annotation']['activity_label'] if seg else None
                if bool(seg)!=s['training_eligible'] or label!=s['training_label']:raise ValueError('Training eligibility/label mismatch')
                truth=next((a['annotation']['activity_label'] for a in ss if a['direct'] and bounds[a['lo']]<=end<bounds[a['hi']]),None)
                rec=dict(s,truth_at_decision=truth,subject_id=v['subject_id'],relative_end=str(end))
                all_rows.append(rec);stream_rows.append(dict(r,video_id=v['video_id'],cadence=cadence))
                if seg:
                    support[label]+=1;by_cadence[cadence][label]+=1
                    provenance.append(dict(video_id=v['video_id'],cadence=cadence,end=s['end'],segment_id=seg['annotation']['segment_id'],annotation=seg['annotation'],occasion=s['recording_occasion']))
            if j!=len(saved):raise ValueError('Decision count mismatch')
    expected={'standing':432,'sitting':378,'lying':423,'moving':78}
    if dict(support)!=expected or sum(support.values())!=1311:raise ValueError('1311 support mismatch: '+str(support))
    vectorize([r for r in all_rows if r['available']],schema)
    return all_rows,stream_rows,schema,hashes,dict(checks=dict(checks),support=dict(support),by_cadence={k:dict(v) for k,v in by_cadence.items()},provenance=provenance)

def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows,stream,schema,hashes,audit=audit_inputs();write(out/'audit.json',audit);write(out/'input_hashes.json',hashes);write(out/'feature_schema.json',schema)
    old=json.loads(Path('outputs/low_rate_development_v001_20261030/config.json').read_text(encoding='utf-8-sig'))
    c=dict(feature_version=VERSION,dimensions=schema['dimensions'],classes=list(CLASSES),seed=old['seed'],threads=old['threads'],interop_threads=old['interop_threads'],steps=old['steps'],lr=old['lr'],l2=old['l2'],dtype='float64',device='cpu',optimizer='Adam',betas=[.9,.999],eps=1e-8,weight_decay=0,full_batch=True,final_step=True,training_rows=1311,split=None,
           preprocessing='14 mean/std columns: training-only median (all-missing fallback0) then population center/scale (constant scale1). Coverage and14 missing flags unchanged.',scope='development/resubstitution only',changes_from_low_rate_v001='New acquisition feature semantics;1311 rows instead of3633; optimizer/seed/threads/steps/35-input Linear unchanged; fresh preprocessing.',dataset=str(ROOT))
    write(out/'experiment_config.json',c)
    frozen={n:sha(out/n) for n in ('audit.json','input_hashes.json','feature_schema.json','experiment_config.json')};frozen['training_code']=sha(__file__);write(out/'preparation_manifest.json',frozen)
    print(json.dumps(dict(support=audit['support'],config_sha256=frozen['experiment_config.json'])),flush=True)

def metrics(records,truth_key):
    known=[r for r in records if r.get(truth_key) in CLASSES];matrix={k:{j:0 for j in (*CLASSES,'unavailable')} for k in CLASSES}
    for r in known:matrix[r[truth_key]][r['predicted_label'] or 'unavailable']+=1
    classes={}
    for k in CLASSES:
        tp=matrix[k][k];actual=sum(matrix[k].values());pred=sum(matrix[t][k] for t in CLASSES)
        classes[k]=dict(tp=tp,actual=actual,predicted=pred,precision=tp/pred if pred else None,recall=tp/actual if actual else None,f1=2*tp/(actual+pred) if actual+pred else None,undefined_reason=None if actual and pred else 'no_actual_or_predicted_support; affected denominators zero')
    correct=sum(matrix[k][k] for k in CLASSES);unknown=sum(matrix[k]['unavailable'] for k in CLASSES)
    return dict(known=len(known),correct=correct,wrong_label=len(known)-correct-unknown,unavailable=unknown,accuracy_including_unavailable=correct/len(known) if known else None,confusion=matrix,per_class=classes)

def load_bundle(folder,expected_version=VERSION):
    folder=Path(folder);manifest=json.loads((folder/'frozen_manifest.json').read_text(encoding='utf-8-sig'))
    if expected_version!=VERSION or manifest['feature_version']!=expected_version:raise ValueError('Feature version mismatch')
    for n,h in manifest['artifact_sha256'].items():
        if sha(folder/n)!=h:raise ValueError('Frozen artifact hash mismatch')
    schema=json.loads((folder/'feature_schema.json').read_text(encoding='utf-8-sig'));validate_schema(schema);prep=json.loads((folder/'preprocessing.json').read_text(encoding='utf-8-sig'));config=json.loads((folder/'experiment_config.json').read_text(encoding='utf-8-sig'))
    if prep['feature_version']!=VERSION or prep['order']!=ORDER or prep['classes']!=list(CLASSES) or config['classes']!=list(CLASSES):raise ValueError('Class/preprocessing schema mismatch')
    model=torch.nn.Linear(schema['dimensions'],len(CLASSES),dtype=torch.float64)
    model.load_state_dict(torch.load(folder/'model_state.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
    return model,prep,schema

def train(config,out):
    config=Path(config);prepdir=config.parent;frozen=json.loads((prepdir/'preparation_manifest.json').read_text(encoding='utf-8-sig'))
    for n,h in frozen.items():
        if sha(__file__ if n=='training_code' else prepdir/n)!=h:raise ValueError('Preparation hash mismatch')
    c=json.loads(config.read_text(encoding='utf-8-sig'));out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write(out/'run_info.json',dict(status='incomplete',training_started=False))
    try:
        rows,stream,schema,hashes,audit=audit_inputs(c['dataset'])
        if hashes!=json.loads((prepdir/'input_hashes.json').read_text(encoding='utf-8-sig')):raise ValueError('Inputs changed since freeze')
        write(out/'experiment_config.json',c);write(out/'feature_schema.json',schema);write(out/'input_hashes.json',hashes);write(out/'audit.json',audit)
        train_rows=[r for r in rows if r['training_eligible']];raw=vectorize(train_rows,schema)
        prep=fit(raw);prep.update(feature_version=VERSION,classes=list(CLASSES));write(out/'preprocessing.json',prep)
        x=torch.from_numpy(apply(raw,prep));y=torch.tensor([CLASSES.index(r['training_label']) for r in train_rows])
        torch.set_num_threads(c['threads']);torch.set_num_interop_threads(c['interop_threads']);torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True)
        model=torch.nn.Linear(c['dimensions'],len(CLASSES),dtype=torch.float64);opt=torch.optim.Adam(model.parameters(),lr=c['lr'],betas=tuple(c['betas']),eps=c['eps'],weight_decay=c['weight_decay'])
        (out/'TRAINING_STARTED').write_text('Exactly one authorized fixed development run; no automatic retry.')
        start=time.perf_counter();losses=[]
        for step in range(c['steps']):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y)+c['l2']/2*model.weight.square().sum()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss; no retry')
            loss.backward();opt.step();losses.append(float(loss))
            if (step+1)%100==0:print('step',step+1,'objective',losses[-1],flush=True)
        elapsed=time.perf_counter()-start;model.eval()
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite final weights')
        with torch.no_grad():final_logits=model(x);final_objective=float(torch.nn.functional.cross_entropy(final_logits,y)+c['l2']/2*model.weight.square().sum())
        torch.save(model.state_dict(),out/'model_state.pt');write(out/'loss_history.json',dict(preupdate_objective=losses,final_postupdate_objective=final_objective))
        artifacts={n:sha(out/n) for n in ('model_state.pt','preprocessing.json','feature_schema.json','experiment_config.json')};write(out/'frozen_manifest.json',dict(status='frozen_complete',feature_version=VERSION,artifact_sha256=artifacts))
        reloaded,rp,ss=load_bundle(out)
        with torch.no_grad():logits2=reloaded(torch.from_numpy(apply(vectorize(train_rows,ss),rp)))
        torch.testing.assert_close(final_logits,logits2,rtol=0,atol=0)
        predictions=[];parity=0
        with (out/'predictions.jsonl').open('x') as f:
            for r,online in zip(rows,stream):
                label=None;logits=None;scores=None
                if r['available']:
                    a=vectorize([r],schema);b=vectorize([online],schema);np.testing.assert_array_equal(a,b)
                    with torch.no_grad():z=model(torch.from_numpy(apply(a,prep)))[0];zz=reloaded(torch.from_numpy(apply(b,rp)))[0]
                    torch.testing.assert_close(z,zz,rtol=0,atol=0);parity+=1;logits=z.tolist();scores=z.softmax(0).tolist();label=CLASSES[int(z.argmax())]
                rec=dict(video_id=r['video_id'],occasion=r['recording_occasion'],cadence=r['cadence'],end=r['end'],relative_end=r['relative_end'],version=VERSION,available=r['available'],reason=r['reason'],expires_at=r['expires_at'],training_eligible=r['training_eligible'],training_label=r['training_label'],truth_at_decision=r['truth_at_decision'],predicted_label=label,logits=logits,scores_uncalibrated=scores)
                predictions.append(rec);f.write(json.dumps(rec,allow_nan=False)+'\n')
        training=[r for r in predictions if r['training_eligible']]
        report=dict(scope='TRAINING/RESUBSTITUTION; replay is development, not heldout. Unknown included on known point labels; unlabelled points not scored.',training=metrics(training,'training_label'),training_by_cadence={k:metrics([r for r in training if r['cadence']==k],'training_label') for k in CYCLES},replay_by_cadence={k:dict(total=sum(r['cadence']==k for r in predictions),available=sum(r['cadence']==k and r['available'] for r in predictions),unavailable_reasons=dict(Counter(r['reason'] for r in predictions if r['cadence']==k and not r['available'])),known_point_diagnostics=metrics([r for r in predictions if r['cadence']==k],'truth_at_decision')) for k in CYCLES},training_by_video={k:metrics([r for r in training if r['video_id']==k],'training_label') for k in sorted({r['video_id'] for r in training})},training_by_occasion={k:metrics([r for r in training if r['occasion']==k],'training_label') for k in sorted({r['occasion'] for r in training})})
        write(out/'metrics.json',report);write(out/'errors.json',[r for r in predictions if r['truth_at_decision'] is not None and r['predicted_label']!=r['truth_at_decision']])
        write(out/'missingness.json',dict(training_rows=len(raw),by_column=dict(zip(ORDER[:14],np.isnan(raw[:,:14]).sum(0).tolist())),by_cadence={k:dict(zip(ORDER[:14],np.isnan(vectorize([r for r in train_rows if r['cadence']==k],schema)[:,:14]).sum(0).tolist())) for k in CYCLES}))
        for p,h in hashes.items():
            if sha(p)!=h:raise ValueError('Protected input changed')
        write(out/'run_info.json',dict(status='complete',training_started=True,training_runs=1,steps=c['steps'],training_rows=len(train_rows),initial_objective=losses[0],final_objective=final_objective,optimization_seconds=elapsed,model_bytes=(out/'model_state.pt').stat().st_size,reload_logits_equal=True,reload_labels_equal=True,replay_logit_label_parity_checks=parity,torch_version=torch.__version__,numpy_version=np.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),training_code_sha256=sha(__file__),preparation_manifest_sha256=sha(prepdir/'preparation_manifest.json'),comparison_note='No direct old-model metric comparison: .25s historical grid differs from acquisition-anchored decisions. No resampling/alignment or false performance gain claimed.'))
        write(out/'output_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
        print(json.dumps(dict(training=report['training'],optimization_seconds=elapsed)),flush=True)
    except BaseException as e:
        write(out/'run_info.json',dict(status='incomplete',training_started=(out/'TRAINING_STARTED').exists(),error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);a=sub.add_parser('prepare');a.add_argument('--output-dir',required=True);a=sub.add_parser('train');a.add_argument('--config',required=True);a.add_argument('--output-dir',required=True);a=p.parse_args()
    prepare(a.output_dir) if a.command=='prepare' else train(a.config,a.output_dir)
