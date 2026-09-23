"""Small local-backend consistency check, never an accuracy assessment."""
from importlib import metadata
import hashlib
import math
from pathlib import Path
import platform

from .pose_benchmark import load_local_model, model_identity, prepare_frames, runtime_snapshot
from .pose_extraction import quality_people, write_report
from .pose_preview import PREDICT_CONFIG, sha256_file
from .target_selection import box_iou
from .video_inspection import InspectionError


def match_people(pt, ncnn, min_iou=.5):
    """Greedy descending IoU, one-to-one; tie by frame-local array indices."""
    def valid(person):
        box = person['box_xyxy']
        return all(v is not None and math.isfinite(v) for v in box) and box[2] > box[0] and box[3] > box[1]
    candidates = sorted((-box_iou(a['box_xyxy'], b['box_xyxy']), i, j)
                        for i, a in enumerate(pt) for j, b in enumerate(ncnn) if valid(a) and valid(b))
    left, right, matches = set(), set(), []
    for negative, i, j in candidates:
        if -negative < min_iou or i in left or j in right:
            continue
        left.add(i)
        right.add(j)
        joints = []
        for a, b in zip(pt[i]['keypoints'], ncnn[j]['keypoints']):
            finite_xy = all(v is not None and math.isfinite(v) for v in (a['x'], a['y'], b['x'], b['y']))
            joints.append(dict(index=a['index'], name=a['name'], pt_confidence=a['confidence'],
                ncnn_confidence=b['confidence'],
                dx_ncnn_minus_pt=b['x']-a['x'] if finite_xy else None,
                dy_ncnn_minus_pt=b['y']-a['y'] if finite_xy else None,
                distance_pixels=math.hypot(b['x']-a['x'], b['y']-a['y']) if finite_xy else None))
        distances = [p['distance_pixels'] for p in joints if p['distance_pixels'] is not None]
        matches.append(dict(pt_detection_index=pt[i]['detection_index'], ncnn_detection_index=ncnn[j]['detection_index'],
            box_iou=-negative, pt_box=pt[i]['box_xyxy'], ncnn_box=ncnn[j]['box_xyxy'],
            pt_box_confidence=pt[i]['box_confidence'], ncnn_box_confidence=ncnn[j]['box_confidence'],
            keypoints=joints, finite_joint_pair_count=len(distances),
            mean_joint_distance_pixels=sum(distances)/len(distances) if distances else None,
            max_joint_distance_pixels=max(distances) if distances else None))
    return dict(matches=matches, unmatched_pt=[p['detection_index'] for i,p in enumerate(pt) if i not in left],
                unmatched_ncnn=[p['detection_index'] for j,p in enumerate(ncnn) if j not in right])


def compare_backends(source: Path, pt: Path, ncnn: Path, output: Path, imgsz=320, frames=3):
    source, pt, ncnn, output = (p.expanduser().resolve() for p in (source, pt, ncnn, output))
    if output.exists():
        raise InspectionError('Choose a new comparison output directory')
    output.mkdir(parents=True, exist_ok=False)
    config = {**PREDICT_CONFIG, 'imgsz': imgsz}
    report = dict(status='incomplete', completed=False, errors=[], frames=[], config=config,
        python=platform.python_version(), platform=platform.platform(),
        package_versions={d.metadata['Name']: d.version for d in metadata.distributions()},
        matching='Greedy descending IoU >=0.5, one-to-one, deterministic array-index tie break; no tracking.',
        interpretation='Consistency only, NOT accuracy. All 17 raw joint values and confidences retained; low-confidence differences are not trusted joint errors.',
        preprocessing='Identical decoded PIL RGB objects and pixels for both backends, rect=False. Library converts RGB->BGR->normalized RGB.',
        runtime_snapshots={})
    prepared = []
    write_report(output/'comparison.json', report)
    try:
        if type(frames) is not int or not 1 <= frames <= 30 or type(imgsz) is not int or imgsz <= 0 or imgsz % 32:
            raise InspectionError('Require frames 1..30 and positive imgsz multiple of 32')
        identities = {'pt': model_identity(pt, imgsz), 'ncnn': model_identity(ncnn, imgsz)}
        if identities['pt']['backend'] != 'pt' or identities['ncnn']['backend'] != 'ncnn':
            raise InspectionError('Require one local PT file and one NCNN directory')
        report.update(models=identities, video_sha256=sha256_file(source))
        report['timeline'] = prepare_frames(source, frames, prepared)
        predictions = {}
        for name, weights in (('pt', pt), ('ncnn', ncnn)):
            model, _ = load_local_model(weights, output, config)
            predictions[name] = []
            for row, image in prepared:
                before = hashlib.sha256(image.tobytes()).hexdigest()
                results = model.predict(source=image, **config)
                if len(results) != 1 or tuple(results[0].orig_shape) != (row['height'], row['width']):
                    raise InspectionError('Result dimensions/count mismatch')
                if hashlib.sha256(image.tobytes()).hexdigest() != before:
                    raise InspectionError('Backend mutated source pixels')
                predictions[name].append(quality_people(results[0], row['width'], row['height']))
            report['runtime_snapshots'][name] = runtime_snapshot(model, config)
            del model
        for index, (row, image) in enumerate(prepared):
            a, b = predictions['pt'][index], predictions['ncnn'][index]
            report['frames'].append(dict(**row, rgb_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                pt_person_count=len(a), ncnn_person_count=len(b), pt_people=a, ncnn_people=b, **match_people(a,b)))
        if sha256_file(source) != report['video_sha256'] or any(model_identity(p, imgsz) != identities[n] for n,p in (('pt',pt),('ncnn',ncnn))):
            raise InspectionError('Input changed during comparison')
        report.update(status='complete', completed=True)
    except (Exception, KeyboardInterrupt) as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        for _, image in prepared:
            image.close()
        write_report(output/'comparison.json', report)
    return report
