"""Read-only report for the scheduling-update logs; stdlib only, no inference."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def distribution(values, population):
    return dict(count=len(values),population=population,missing=population-len(values),
                mean_seconds=statistics.mean(values) if values else None,
                median_seconds=statistics.median(values) if values else None,
                min_seconds=min(values) if values else None,max_seconds=max(values) if values else None)


def analyze(summary, rows):
    frames=[r for r in rows if r['type']=='frame']
    fm={(r.get('target_epoch'),r.get('capture_sequence'),r.get('completed_host_ns')):r for r in frames}
    unique={}
    for r in rows:
        if r['type']=='ui_update' and r.get('frame_id') is not None:
            unique.setdefault((r.get('target_epoch'),r['frame_id'],r.get('completed_host_ns')),r)
    received=[r for key,r in unique.items() if key in fm]
    def delta(items,a,b):
        return distribution([(r[b]-r[a])/1e9 for r in items if r.get(a) is not None and r.get(b) is not None],len(items))
    timing=dict(frame_age_at_worker_start=delta(frames,'host_acquired_ns','process_started_host_ns'),
                predict=distribution([r['predict_seconds'] for r in frames if r.get('predict_seconds') is not None],len(frames)),
                classifier=delta(frames,'classifier_started_host_ns','classifier_completed_host_ns'),
                completion_to_publication=delta(received,'completed_host_ns','published_host_ns'),
                publication_to_ui_receipt=delta(received,'published_host_ns','ui_received_host_ns'),
                receipt_to_ui_assessment=delta(received,'ui_received_host_ns','assessed_host_ns'))
    for group,part in [('first',frames[:1]),('subsequent',frames[1:])]:
        timing[group+'_predict']=distribution([r['predict_seconds'] for r in part if r.get('predict_seconds') is not None],len(part))
    return dict(run_status=summary.get('status'),errors=summary.get('errors'),
                log_omitted_records=summary.get('log_omitted_records'),
                log_coverage='partial' if summary.get('log_omitted_records') or len(frames)!=summary.get('processed') else 'frame count matches; inspect run status',
                logged_pose_frames=len(frames),received_unique_pose_results=len(received),non_pose_seed_or_unmatched_results=len(unique)-len(received),
                timings=timing,decision_counts=summary.get('decision_counts'),
                first_result_display_assessments=summary.get('display_summary'),repeated_ui_updates=summary.get('async_ui'),
                accounting={k:summary.get(k) for k in ('captured','processed','overwritten_pending','dropped_incoming','drained_older','async_queue_counts','legacy_unprocessed_residual','async_frames_discarded_before_processing','capture_accounting_residual','unprocessed_on_stop')},
                shutdown={k:summary.get(k) for k in ('async_worker_stopped','forced_capture_termination','capture_process_alive','capture_exitcode')},
                notes=['Missing timestamps remain null; no estimates from old logs.',
                       'Publication/UI timing population is unique received pose results only; excludes seed-only and overwritten/discarded results.',
                       'First-result frame assessments differ from repeated UI updates. Available historical decisions are not fresh live labels.',
                       'Fresh submission duration is clipped at source-support expiry; not proof of physical display/human perception.',
                       'completion_to_publication includes worker logging/telemetry. publication_to_ui_receipt includes scheduling/polling, not pure rendering.'])


def run(source,output):
    source=Path(source);output=Path(output)
    s=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (source/'diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
    result=analyze(s,rows)
    result['input_sha256']={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('summary.json','diagnostics.jsonl')}
    output.mkdir(parents=True,exist_ok=False)
    (output/'timing_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir');p.add_argument('--output-dir',required=True);a=p.parse_args()
    print(json.dumps(run(a.run_dir,a.output_dir),indent=2))
