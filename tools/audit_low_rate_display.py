"""Read-only legacy Pi-log timing reconstruction; no model inference."""
import argparse,collections,hashlib,json,statistics
from pathlib import Path

def audit(source,out):
    source=Path(source);out=Path(out);out.mkdir(exist_ok=False,parents=True)
    summary=json.loads((source/'summary.json').read_text())
    rows=[json.loads(x) for x in (source/'diagnostics.jsonl').read_text().splitlines()]
    frames=[r for r in rows if r['type']=='frame'];decisions=[r for r in rows if r['type']=='decision']
    groups=collections.defaultdict(list)
    for d in decisions:groups[(d['epoch'],d['latest_arrived_host_ns'])].append(d)
    checks=[];latest_unavailable=[];schedule_errors=[]
    for i,f in enumerate(frames):
        ds=groups.get((f['epoch'],f['host_acquired_ns']),[])
        if not ds:continue
        d=ds[-1];ends=[x['decision_host_ns'] for x in ds]
        if ends!=sorted(ends) or not 0<=f['host_acquired_ns']-ends[-1]<250000000:schedule_errors.append(i)
        if not d['available']:
            if any(x['available'] for x in ds):latest_unavailable.append(dict(frame=i,reason=d['reason'],historical_available=sum(x['available'] for x in ds),frame_label=f['label']))
            continue
        support=d['decision_host_ns']-round(d['latest_age']*1e9)
        assessment=f['host_acquired_ns']+round(f['host_age_before_display_seconds']*1e9)
        age=(assessment-support)/1e9
        checks.append(dict(frame=f['processed_index'],epoch=f['epoch'],decision_host_ns=d['decision_host_ns'],legacy_computed_host_ns=d['computed_host_ns'],latest_support_host_ns=support,assessment_host_ns=assessment,support_age_seconds=age,expired=age>summary['classifier_config']['max_latest_age_seconds'],frame_label=f['label'],frame_reason=f['reason'],lag_at_capture_seconds=(f['host_acquired_ns']-support)/1e9,source_frame_age_seconds=f['host_age_before_display_seconds'],matches_previous_capture=(i>0 and support==frames[i-1]['host_acquired_ns'])))
    assert len(frames)==summary['processed'] and len(decisions)==sum(summary['decision_counts'].values())
    assert not schedule_errors
    assert all(c['expired']==(c['frame_reason']=='display_support_expired') and c['expired']==(c['frame_label'] is None) and c['matches_previous_capture'] for c in checks)
    result=dict(source=str(source),input_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('summary.json','diagnostics.jsonl')},processed_frames=len(frames),decision_counts=dict(collections.Counter('available' if d['available'] else 'unavailable' for d in decisions)),frame_label_counts=dict(collections.Counter(f['label'] or 'UNAVAILABLE' for f in frames)),frame_non_display_reasons=dict(collections.Counter(f['reason'] for f in frames if f['label'] is None)),latest_available_frames=len(checks),timing_replay_mismatches=0,schedule_errors=schedule_errors,latest_unavailable_overrides=latest_unavailable,log_omitted_records=summary['log_omitted_records'],telemetry_flags=dict(collections.Counter(r['throttled_hex'] for r in rows if r['type']=='telemetry')),limits='Frame labels were logged before telemetry/final expiry/UI. They are not proof of screen visibility. Legacy computed_host_ns precedes classifier forward, not completion. Timing replay only; log has no raw poses for feature replay.')
    for name,expired in [('expired',True),('fresh',False)]:
        a=[c['support_age_seconds'] for c in checks if c['expired']==expired]
        result[name]=dict(count=len(a),support_age_min=min(a),support_age_median=statistics.median(a),support_age_max=max(a))
    (out/'timing_replay.json').write_text(json.dumps(checks,indent=2))
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output-dir',required=True);a=p.parse_args();print(json.dumps(audit(a.source,a.output_dir),indent=2))
