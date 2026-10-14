"""Confirmed-only evaluation of saved predictions; never runs inference."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path

from .action_baseline import sha256, write_json

CLASSES = ('standing', 'sitting', 'lying', 'moving')
COLUMNS = (*CLASSES, 'unknown')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def identity(row):
    base = Fraction(row['time_base'])
    if base <= 0:
        raise ValueError('Nonpositive time_base')
    time = int(row['pts']) * base
    if not math.isclose(float(time), float(row['presentation_time_seconds']), abs_tol=1e-9, rel_tol=0):
        raise ValueError('Timestamp inconsistent with PTS')
    return int(row['frame_index']), int(row['pts']), base, time


def ratio(numerator, denominator):
    return dict(value=numerator / denominator if denominator else None,
                numerator=numerator, denominator=denominator,
                reason=None if denominator else 'zero_denominator')


def metrics(matrix):
    total = sum(sum(row.values()) for row in matrix.values())
    if not total:
        return dict(confusion_matrix=None, per_class=None, prediction_coverage=None,
                    overall_accuracy=None, reason='no_confirmed_known_class_intervals')
    per_class = {}
    for label in CLASSES:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[truth][label] for truth in CLASSES)
        per_class[label] = dict(precision=ratio(tp, predicted), recall=ratio(tp, support),
                               f1=ratio(2*tp, support+predicted), support=support,
                               predicted_count_or_duration=predicted)
    return dict(confusion_matrix=matrix, per_class=per_class,
                prediction_coverage=ratio(total-sum(r['unknown'] for r in matrix.values()), total),
                overall_accuracy=ratio(sum(matrix[c][c] for c in CLASSES), total), reason=None)


def evaluate_video(video_id, timeline, predictions, annotations):
    """Use half-open frame ranges, exact PTS, actual source frame durations."""
    n = len(timeline)
    if not n or len(predictions) != n:
        raise ValueError('Missing/extra prediction or empty timeline')
    times, durations = [], []
    for index, (t, p) in enumerate(zip(timeline, predictions)):
        ti, pi = identity(t), identity(p)
        if ti != pi or ti[0] != index or p['video_id'] != video_id:
            raise ValueError('Prediction/frame/video alignment mismatch')
        if p['predicted_label'] not in COLUMNS:
            raise ValueError('Unsupported prediction label; do not silently map it')
        duration = int(t['source_duration']) * ti[2]
        if duration <= 0 or (times and (ti[3] <= times[-1] or times[-1]+durations[-1] > ti[3])):
            raise ValueError('Invalid or overlapping frame timing')
        times.append(ti[3]); durations.append(duration)
    boundaries = [t-times[0] for t in times] + [times[-1]+durations[-1]-times[0]]
    assignments = [None] * n
    ids = set()
    for a in annotations:
        if a['video_id'] != video_id:
            raise ValueError('Annotation video mismatch')
        if not a['segment_id'] or a['segment_id'] in ids:
            raise ValueError('Missing/duplicate segment ID')
        ids.add(a['segment_id'])
        start, end = int(a['start_frame_index']), int(a['end_frame_index_exclusive'])
        if not 0 <= start < end <= n:
            raise ValueError('Annotation out of bounds')
        for key, boundary in (('start_sec', boundaries[start]), ('end_sec', boundaries[end])):
            if not math.isclose(float(a[key]), float(boundary), abs_tol=1e-6, rel_tol=0):
                raise ValueError('Annotation seconds do not match source frame boundaries')
        status = a['annotation_status']
        if status not in ('unannotated', 'draft', 'needs_review', 'reviewed'):
            raise ValueError('Unsupported annotation status')
        if a['activity_label'] not in (*CLASSES, 'unknown', 'no_person'):
            raise ValueError('Unsupported annotation label')
        if status == 'reviewed' and not all(a.get(k) for k in ('reviewer_id','reviewed_at_utc','annotator_id','annotated_at_utc')):
            raise ValueError('Reviewed annotation requires explicit provenance')
        confirmed = (status == 'reviewed' and a['label_certainty'] == 'clear'
                     and a['segment_kind'] == 'activity' and a['activity_label'] in CLASSES)
        category = ('confirmed_known' if confirmed else 'confirmed_excluded' if status == 'reviewed'
                    else 'proposed' if status == 'draft' and a['segment_kind'] == 'activity' else 'uncertain_pending')
        reason = None if confirmed else a.get('uncertainty_reason') or a.get('notes') or category
        for i in range(start, end):
            if assignments[i] is not None:
                raise ValueError('Overlapping annotations, including pending intervals')
            assignments[i] = (category, a['activity_label'], reason)
    matrices = [{c:{p:0 for p in COLUMNS} for c in CLASSES} for _ in range(2)]
    coverage, exclusions = {}, {}
    for i, assignment in enumerate(assignments):
        category, label, reason = assignment or ('unannotated', None, 'no_annotation')
        duration = float(durations[i])
        bucket = coverage.setdefault(category, dict(frames=0, seconds=0.0))
        bucket['frames'] += 1; bucket['seconds'] += duration
        if category == 'confirmed_known':
            matrices[0][label][predictions[i]['predicted_label']] += 1
            matrices[1][label][predictions[i]['predicted_label']] += duration
        else:
            bucket = exclusions.setdefault(reason, dict(frames=0, seconds=0.0))
            bucket['frames'] += 1; bucket['seconds'] += duration
    return dict(video_id=video_id, frame_count=n, source_frame_duration_seconds=float(sum(durations)),
                source_span_seconds=float(boundaries[-1]), uncovered_timing_gap_seconds=float(boundaries[-1]-sum(durations)),
                annotation_coverage=coverage, excluded_from_metrics_by_reason=exclusions,
                frame_weighted=metrics(matrices[0]), duration_weighted=metrics(matrices[1]),
                _matrices=matrices)


def run(package_path, annotations_path, output_dir):
    package_path, annotations_path, output_dir = map(Path, (package_path, annotations_path, output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='incomplete', errors=[], results=[])
    try:
        package = json.loads(package_path.read_text(encoding='utf-8'))
        annotations = read_csv(annotations_path)
        videos = package['videos']
        if len({v['video_id'] for v in videos}) != len(videos):
            raise ValueError('Duplicate video ID')
        if any(a['video_id'] not in {v['video_id'] for v in videos} for a in annotations):
            raise ValueError('Annotation for unknown video')
        hashes = {str(package_path):sha256(package_path), str(annotations_path):sha256(annotations_path)}
        combined = [{c:{p:0 for p in COLUMNS} for c in CLASSES} for _ in range(2)]
        for v in videos:
            paths = {}
            for key in ('timeline', 'predictions'):
                path = Path(v[key]); paths[key] = path
                digest = sha256(path)
                if digest != v[key+'_sha256']:
                    raise ValueError(f'{key} hash mismatch')
                hashes[str(path)] = digest
            pred = [json.loads(x) for x in paths['predictions'].read_text(encoding='utf-8').splitlines()]
            if any(p['config_sha256'] != package['config_sha256'] for p in pred):
                raise ValueError('Frozen configuration hash mismatch')
            result = evaluate_video(v['video_id'], read_csv(paths['timeline']), pred,
                                    [a for a in annotations if a['video_id']==v['video_id']])
            matrices = result.pop('_matrices')
            for mode in range(2):
                for c in CLASSES:
                    for p in COLUMNS: combined[mode][c][p] += matrices[mode][c][p]
            report['results'].append(result)
        coverage, exclusions = {}, {}
        for result in report['results']:
            for field, target in (('annotation_coverage', coverage), ('excluded_from_metrics_by_reason', exclusions)):
                for key, values in result[field].items():
                    bucket = target.setdefault(key, dict(frames=0, seconds=0.0))
                    for unit in ('frames', 'seconds'): bucket[unit] += values[unit]
        report.update(status='complete', input_hashes=hashes, evaluator_sha256=sha256(__file__),
                      annotation_coverage=coverage, excluded_from_metrics_by_reason=exclusions,
                      config_sha256=package['config_sha256'],
                      aggregate=dict(frame_weighted=metrics(combined[0]), duration_weighted=metrics(combined[1])),
                      confirmed_labels_available=bool(sum(sum(r.values()) for r in combined[0].values())),
                      matrix_rows=list(CLASSES), matrix_columns=list(COLUMNS),
                      policy='Only reviewed/clear/activity known classes; startup and unresolved unknown predictions retained. Duration weights are source frame durations, not nominal FPS. No accuracy when no confirmed labels.')
    except Exception as exc:
        report['errors'].append(str(exc))
        raise
    finally:
        write_json(output_dir/'evaluation.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args=parser.parse_args()
    result=run(args.package,args.annotations,args.output_dir)
    print('Evaluation complete; confirmed labels available:', result['confirmed_labels_available'])


if __name__ == '__main__':
    main()
