"""Offline, timestamp-based supervised windows. No predictions or fitting."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

CLASSES = ('standing', 'sitting', 'lying', 'moving')
CHANNELS = {
    'left_knee_deg': ('left_knee_angle_deg', None),
    'right_knee_deg': ('right_knee_angle_deg', None),
    'torso_deg': ('torso_inclination_deg', None),
    'hip_vx_per_s': ('hip_image_velocity_per_second', 0),
    'hip_vy_per_s': ('hip_image_velocity_per_second', 1),
    'box_width_fraction': ('box_size_image', 0),
    'box_height_fraction': ('box_size_image', 1),
}
CONFIG = dict(version='supervised_windows_v001', window_seconds=1.0,
              stride_seconds=0.25, max_time_gap_seconds=0.5,
              eligibility='reviewed_clear_activity_and_direct_user_confirmation',
              schedule='segment_start + window + k * stride',
              channels=CHANNELS, weighting='causal_observed_support',
              minimum_feature_coverage=None, split=None)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def timestamp(row):
    return int(row['pts']) * Fraction(row['time_base'])


def aligned_frames(rows, timeline):
    if not rows or len(rows) != len(timeline):
        raise ValueError('empty or mismatched frame counts')
    origin = timestamp(timeline[0])
    times = []
    for i, (r, t) in enumerate(zip(rows, timeline)):
        if int(t['frame_index']) != i or r['frame_index'] != i:
            raise ValueError('noncontiguous frame indices')
        if r['pts'] != int(t['pts']) or Fraction(r['time_base']) != Fraction(t['time_base']):
            raise ValueError('PTS/time_base mismatch')
        now = timestamp(t)
        if Fraction(t['time_base']) <= 0 or (times and now-origin <= times[-1]):
            raise ValueError('invalid timestamp order')
        for key, expected in [('presentation_time_seconds', now),
                              ('relative_time_seconds', now-origin)]:
            if not math.isclose(float(r[key]), float(expected), abs_tol=1e-9, rel_tol=0):
                raise ValueError('feature timestamp mismatch')
        times.append(now-origin)
    duration = int(timeline[-1]['source_duration']) * Fraction(timeline[-1]['time_base'])
    if duration <= 0:
        raise ValueError('missing final frame duration')
    return times + [times[-1]+duration]


def segments(annotations, bounds):
    occupied = set()
    result = []
    for a in annotations:
        lo, hi = int(a['start_frame_index']), int(a['end_frame_index_exclusive'])
        if not 0 <= lo < hi <= len(bounds)-1:
            raise ValueError('annotation outside timeline')
        for key, value in [('start_sec', bounds[lo]), ('end_sec', bounds[hi])]:
            if not math.isclose(float(a[key]), float(value), abs_tol=1e-9, rel_tol=0):
                raise ValueError('annotation boundary mismatch')
        indices = set(range(lo, hi))
        if occupied & indices:
            raise ValueError('overlapping annotations')
        occupied |= indices
        reviewed = (a['annotation_status']=='reviewed' and a['label_certainty']=='clear'
                    and a['segment_kind']=='activity' and a['activity_label'] in CLASSES
                    and all(a.get(k) for k in ('annotator_id','reviewer_id',
                                               'annotated_at_utc','reviewed_at_utc')))
        # A reviewed, assistant-selected evaluation subset is not promoted to
        # an explicitly user-confirmed supervised training annotation.
        direct = reviewed and a.get('_direct_confirmation_source') is True
        result.append(dict(annotation=a, lo=lo, hi=hi, reviewed=reviewed, direct=direct,
                           start=float(bounds[lo]), end=float(bounds[hi]),
                           duration=float(bounds[hi]-bounds[lo])))
    return result


def make_window(rows, bounds, start, end, max_gap=Fraction(1, 2)):
    """Return features over [start,end); never use a feature at/after end."""
    indices = [i for i in range(len(rows)) if bounds[i] < end and bounds[i+1] > start]
    if not indices or bounds[indices[0]] > start or bounds[indices[-1]+1] < end:
        return None, 'incomplete_temporal_support'
    for k, i in enumerate(indices):
        r = rows[i]
        if not r.get('timestamp_valid') or r['target_status'] not in ('seeded','matched'):
            return None, 'unresolved_or_invalid_timestamp'
        if bounds[i+1]-bounds[i] > max_gap:
            return None, 'timestamp_gap'
        if k and (r['target_status']=='seeded' or not r['temporal_pair']['valid']
                  or r['temporal_pair']['previous_frame_index'] != i-1):
            return None, 'track_break'
    values = {}
    for name, (feature, component) in CHANNELS.items():
        samples = []
        reasons = Counter()
        for i in indices:
            r = rows[i]
            f = r['features'][feature]
            a, b = max(start, bounds[i]), min(end, bounds[i+1])
            if 'velocity' in feature:
                # Backward difference belongs to [previous PTS,current PTS].
                # Require its entire support inside this window; no boundary leak.
                if i==0 or bounds[i-1] < start or not r['temporal_pair']['valid']:
                    reasons['derivative_support_outside_window_or_invalid'] += 1
                    continue
                a, b = bounds[i-1], bounds[i]
            if not f['valid'] or f['value'] is None:
                reasons[f.get('reason') or 'invalid_saved_feature'] += 1
                continue
            v = f['value'] if component is None else f['value'][component]
            if not isinstance(v, (int,float)) or not math.isfinite(v):
                raise ValueError('valid feature contains nonfinite value')
            samples.append((float(v), float(b-a)))
        support = sum(w for _,w in samples)
        mean = sum(v*w for v,w in samples)/support if support else None
        values[name] = dict(mean=mean,
            std=math.sqrt(sum(w*(v-mean)**2 for v,w in samples)/support) if support else None,
            valid=bool(support), valid_duration_seconds=support,
            coverage=support/float(end-start), valid_observations=len(samples),
            missing_reasons=dict(reasons))
    return dict(model_inputs=values, frame_indices=indices,
                observed_frame_times=[dict(frame_index=i,pts=rows[i]['pts'],
                    time_base=rows[i]['time_base'], relative_time_seconds=float(bounds[i])) for i in indices],
                temporal_support_seconds=float(end-start)), None


def build(manifest, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'summary.json', {'status':'incomplete'})
    inputs = json.loads(Path(manifest).read_text(encoding='utf-8'))
    inventory, rejected, windows = [], [], []
    hashes = {str(manifest):sha(manifest), str(Path(__file__)):sha(__file__)}
    try:
        for video in inputs['videos']:
            fp, tp = Path(video['features']), Path(video['timeline'])
            for p in (fp,tp): hashes[str(p)] = sha(p)
            rows = [json.loads(s) for s in fp.read_text(encoding='utf-8').splitlines()]
            if any(r['video_id'] != video['video_id'] for r in rows):
                raise ValueError('feature video identity mismatch')
            bounds = aligned_frames(rows, csv_rows(tp))
            annotations = []
            for ap in video['annotations']:
                hashes[ap] = sha(ap)
                annotations += [dict(a, _direct_confirmation_source=
                    ap in video.get('direct_confirmation_sources', []))
                    for a in csv_rows(ap) if a['video_id']==video['video_id']]
            ss = segments(annotations, bounds)
            availability = {k:sum(r['features'][f]['valid'] for r in rows)
                            for k,(f,_) in CHANNELS.items()}
            inventory.append(dict(**video, frames=len(rows), duration=float(bounds[-1]),
                feature_valid_frames=availability, segments=ss))
            for s in ss:
                a = s['annotation']
                meta = dict(video_id=video['video_id'], subject_id=video['subject_id'],
                    recording_occasion_group_id=video['recording_occasion_group_id'],
                    segment_id=a['segment_id'], label=a['activity_label'])
                if not s['direct']:
                    rejected.append(dict(**meta, start=s['start'],end=s['end'],
                        reason='not_direct_user_confirmed',
                        also_shorter_than_window=s['duration']<CONFIG['window_seconds']))
                    continue
                start, stop = bounds[s['lo']], bounds[s['hi']]
                duration, stride = Fraction(str(CONFIG['window_seconds'])), Fraction(str(CONFIG['stride_seconds']))
                if stop-start < duration:
                    rejected.append(dict(**meta, start=float(start),end=float(stop),reason='segment_shorter_than_window'))
                    continue
                end = start+duration
                while end <= stop:
                    value, reason = make_window(rows,bounds,end-duration,end)
                    record = dict(**meta,start=float(end-duration),end=float(end))
                    if reason:
                        rejected.append(dict(**record,reason=reason))
                    else:
                        windows.append(dict(**record,**value))
                    end += stride
        for filename, records in [('windows.jsonl',windows),('rejections.jsonl',rejected)]:
            with (output/filename).open('w',encoding='utf-8') as f:
                for r in records: f.write(json.dumps(r,allow_nan=False)+'\n')
        write(output/'inventory.json',inventory)
        classes = {}
        for label in CLASSES:
            segs = [s for v in inventory for s in v['segments'] if s['annotation']['activity_label']==label]
            classes[label] = dict(accepted_windows=sum(w['label']==label for w in windows),
                direct_confirmed_seconds=sum(s['duration'] for s in segs if s['direct']),
                preliminary_reviewed_seconds=sum(s['duration'] for s in segs if s['reviewed'] and not s['direct']),
                confirmed_segment_lengths=[s['duration'] for s in segs if s['direct']])
        write(output/'summary.json',dict(status='complete',classes=classes,
            accepted_windows=len(windows),rejection_records=len(rejected),
            rejection_reasons=dict(Counter(r['reason'] for r in rejected)),
            labeled_occasion_count=len({w['recording_occasion_group_id'] for w in windows}),
            classes_without_windows=[c for c in CLASSES if not classes[c]['accepted_windows']],
            split='unassigned; assess occasion grouping and class support before any split'))
        write(output/'manifest.json',dict(config=CONFIG,
            config_sha256=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest(),
            input_sha256=hashes, output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
            no_fitting=True, overlapping_windows_are_correlated=True))
    except Exception as e:
        write(output/'summary.json',dict(status='incomplete',error=str(e)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    build(args.manifest,args.output_dir)
