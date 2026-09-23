"""Review only the eight CSV PNGs: verified saved PT640, PT320 and NCNN320."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys

from PIL import Image, ImageDraw, ImageFont

from patient_monitoring.pose_benchmark import load_local_model, model_identity, runtime_snapshot
from patient_monitoring.pose_extraction import quality_keypoint, write_report
from patient_monitoring.pose_preview import MODEL_SHA256, PREDICT_CONFIG, read_samples, serialize_people, sha256_file
from patient_monitoring.pose_review import annotate_review
from patient_monitoring.video_inspection import InspectionError

VARIANTS = (('pt640', 'Original PT | imgsz=640', 640),
            ('pt320', 'Original PT | imgsz=320', 320),
            ('ncnn320', 'NCNN export | imgsz=320', 320))
FLAG_KEYS = ('finite_components', 'finite_values', 'confidence_pass',
             'at_or_outside_boundary', 'passes_quality_gate')


def verified_reuse(folder, sources, csv_hash, weights_hash):
    """Fail closed: never reuse predictions whose identity/config cannot be verified."""
    run = json.loads((folder/'run_info.json').read_text(encoding='utf-8'))
    saved = json.loads((folder/'poses.json').read_text(encoding='utf-8'))['images']
    if (run['status'] != 'complete' or run['processed_image_count'] != len(sources)
            or run['input_csv_sha256'] != csv_hash or run['model']['sha256'] != weights_hash
            or any(run['config'].get(k) != v for k,v in PREDICT_CONFIG.items())
            or run['actual_model_device'] != 'cpu' or run['actual_model_dtype'] != 'torch.float32'):
        raise InspectionError('Saved PT640 completion/source/model/config mismatch')
    for key, value in PREDICT_CONFIG.items():
        if key != 'half' and run['effective_predict_config'].get(key) != value:
            raise InspectionError(f'Saved PT640 effective config mismatch: {key}')
    versions = {k.lower():v for k,v in run['package_versions'].items()}
    for name in ('ultralytics', 'torch', 'torchvision', 'numpy', 'opencv-python', 'pillow'):
        if versions.get(name) != metadata.version(name):
            raise InspectionError(f'Saved PT640 package version mismatch: {name}')
    if len(saved) != len(sources):
        raise InspectionError('Saved PT640 image count mismatch')
    for source, record in zip(sources, saved):
        for key in ('frame_index', 'actual_time_seconds', 'source_sha256', 'width', 'height'):
            if record[key] != source[key]:
                raise InspectionError(f'Saved PT640 source mismatch: {key}')
        if record['person_count'] != len(record['people']):
            raise InspectionError('Saved detection count mismatch')
    return saved, dict(method='reused_verified_saved_prediction', path=str(folder),
        files_sha256={n:sha256_file(folder/n) for n in ('run_info.json','poses.json')},
        original_run_info=run)


def split_quality(people, width, height):
    """Separate original numeric predictions from flags; only nonfinite -> JSON null."""
    raw, flags, drawing = [], [], []
    for index, person in enumerate(people):
        if person['detection_index'] != index or len(person['keypoints']) != 17:
            raise InspectionError('Detection indices/17-joint schema mismatch')
        saved = deepcopy(person)
        box_finite = [v is not None and math.isfinite(v) for v in person['box_xyxy']]
        conf = person['box_confidence']
        conf_finite = conf is not None and math.isfinite(conf)
        saved['box_xyxy'] = [v if finite else None for v,finite in zip(person['box_xyxy'],box_finite)]
        saved['box_confidence'] = conf if conf_finite else None
        quality = dict(detection_index=index, box_finite_values=box_finite,
                       box_confidence_finite=conf_finite, keypoints=[])
        drawn = deepcopy(saved)
        drawn['keypoints'] = []
        for joint, point in enumerate(person['keypoints']):
            if point['index'] != joint:
                raise InspectionError('Keypoint order mismatch')
            numeric = {**point, **{k:float('nan') if point[k] is None else point[k] for k in ('x','y','confidence')}}
            checked = quality_keypoint(numeric, width, height)
            saved['keypoints'][joint] = {k:checked[k] for k in ('index','name','x','y','confidence')}
            quality['keypoints'].append(dict(index=joint,name=point['name'],**{k:checked[k] for k in FLAG_KEYS}))
            drawn['keypoints'].append(checked)
        raw.append(saved)
        flags.append(quality)
        drawing.append(drawn)
    return raw, flags, drawing


def render_row(output, source, variants):
    width, height = source['width'], source['height']
    panels = []
    try:
        for name, label, _ in VARIANTS:
            with Image.open(source['source_image']) as original:
                picture = original.convert('RGB')
            record = dict(frame_index=source['frame_index'], presentation_time_seconds=source['actual_time_seconds'],
                          person_count=len(variants[name]), people=variants[name])
            issues = annotate_review(picture, record)
            if issues:
                raise InspectionError('; '.join(issues))
            panel = Image.new('RGB', (width, height+80), '#10151b')
            panel.paste(picture,(0,80))
            picture.close()
            draw=ImageDraw.Draw(panel)
            draw.text((16,10),label,font=ImageFont.load_default(size=30),fill='white')
            draw.text((16,48),f"Frame {source['frame_index']} | t={source['actual_time_seconds']:.6f} s | detections={record['person_count']}",font=ImageFont.load_default(size=22),fill='white')
            panels.append(panel)
        filename=f"comparison_frame_{source['frame_index']:06d}.png"
        with Image.new('RGB',(3*width,height+80),'#10151b') as row:
            for i,panel in enumerate(panels): row.paste(panel,(i*width,0))
            with (output/filename).open('xb') as file: row.save(file,format='PNG')
        return filename
    finally:
        for panel in panels: panel.close()


def contact_sheet(output, records):
    # Three columns, one frame per row; source-size PNGs remain available for detailed review.
    cell_height=400
    filename='backend_review_sheet.jpg'
    with Image.new('RGB',(1920,96+cell_height*len(records)),'#10151b') as sheet:
        draw=ImageDraw.Draw(sheet)
        draw.text((16,10),'YOLO11n-pose preview | PT640 / PT320 / NCNN320',font=ImageFont.load_default(size=30),fill='white')
        draw.text((16,50),'All detections | gate: finite, confidence >=0.5, strictly inside | pink X: boundary knee/ankle | human review PENDING',font=ImageFont.load_default(size=22),fill='white')
        for i,record in enumerate(records):
            with Image.open(output/record['side_by_side_image']) as row:
                row.thumbnail((1920,cell_height),Image.Resampling.LANCZOS)
                sheet.paste(row,((1920-row.width)//2,96+i*cell_height))
        with (output/filename).open('xb') as file: sheet.save(file,format='JPEG',quality=95)
    return filename


def generate(input_dir, output, reuse):
    root=Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (root/'.venv-ncnn').resolve():
        raise InspectionError('Use the existing .venv-ncnn interpreter')
    input_dir,output,reuse=(p.resolve() for p in (input_dir,output,reuse))
    if output.exists(): raise InspectionError('Choose a new output directory')
    sources=read_samples(input_dir)
    if len(sources)!=8: raise InspectionError('This review requires exactly 8 CSV-referenced PNGs')
    csv_hash=sha256_file(input_dir/'samples.csv')
    weights=root/'models/yolo11n-pose.pt'
    ncnn=root/'models/ncnn_trial_320/yolo11n_pose_320_ncnn_model'
    identities={'pt':model_identity(weights),'ncnn':model_identity(ncnn,320)}
    if identities['pt']['sha256']!=MODEL_SHA256: raise InspectionError('Original M2 PT hash mismatch')
    output.mkdir(parents=True,exist_ok=False)
    run=dict(status='incomplete',human_review='pending',errors=[],variants={},models=identities,
        python=platform.python_version(),platform=platform.platform(),executable=sys.executable,
        package_versions={d.metadata['Name']:d.version for d in metadata.distributions()},
        started_at_utc=datetime.now(timezone.utc).isoformat(),input_csv_sha256=csv_hash,
        source_frames=sources,new_predict_calls=0,reused_predictions=0,images=[],
        drawing_rule='Existing pose_review.annotate_review: draw all finite boxes; joints/edges require passes_quality_gate; pink X marks finite boundary knee/ankle, display clipping only.',
        quality_gate='Provisional conservative filter: finite x/y/confidence, confidence>=0.5, strict interior. Not accuracy or proof of visibility.',
        limitations='Confidence and agreement are not accuracy. Partial body persists. Detection indices are frame-local, not tracking IDs. New inference is Windows only.',
        raw_predictions='poses.json',quality_flags='quality_flags.json')
    raw_rows,flag_rows=[],[]
    write_report(output/'run_info.json',run)
    try:
        cached, provenance=verified_reuse(reuse,sources,csv_hash,identities['pt']['sha256'])
        predictions={'pt640':[r['people'] for r in cached]}
        run['variants']['pt640']=dict(config=dict(PREDICT_CONFIG),**provenance)
        run['reused_predictions']=len(cached)
        pixel_hashes=[]
        for source in sources:
            with Image.open(source['source_image']) as original, original.convert('RGB') as picture:
                pixel_hashes.append(hashlib.sha256(picture.tobytes()).hexdigest())
        for name,_,size in VARIANTS[1:]:
            config={**PREDICT_CONFIG,'imgsz':size}
            model,_=load_local_model(weights if name=='pt320' else ncnn,output,config)
            predictions[name]=[]
            for index,source in enumerate(sources):
                with Image.open(source['source_image']) as original, original.convert('RGB') as picture:
                    if hashlib.sha256(picture.tobytes()).hexdigest()!=pixel_hashes[index]: raise InspectionError('Source pixel mismatch')
                    results=model.predict(source=picture,**config)
                    run['new_predict_calls']+=1
                    if len(results)!=1 or tuple(results[0].orig_shape)!=(source['height'],source['width']): raise InspectionError('Result shape mismatch')
                    predictions[name].append(serialize_people(results[0]))
                    if hashlib.sha256(picture.tobytes()).hexdigest()!=pixel_hashes[index]: raise InspectionError('Model mutated source pixels')
            run['variants'][name]=dict(config=config,method='new_inference_on_csv_pngs',runtime=runtime_snapshot(model,config))
            del model
        for index,source in enumerate(sources):
            raw_record={**source,'rgb_pixels_sha256':pixel_hashes[index],'variants':{}}
            flag_record=dict(frame_index=source['frame_index'],actual_time_seconds=source['actual_time_seconds'],variants={})
            drawings={}
            for name,_,_ in VARIANTS:
                raw,flags,drawing=split_quality(predictions[name][index],source['width'],source['height'])
                raw_record['variants'][name]=dict(person_count=len(raw),people=raw)
                flag_record['variants'][name]=dict(person_count=len(flags),people=flags)
                drawings[name]=drawing
            row=render_row(output,source,drawings)
            raw_rows.append(raw_record);flag_rows.append(flag_record)
            run['images'].append(dict(frame_index=source['frame_index'],actual_time_seconds=source['actual_time_seconds'],
                side_by_side_image=row,person_counts={k:len(v) for k,v in drawings.items()}))
        run['contact_sheet']=contact_sheet(output,run['images'])
        if csv_hash!=sha256_file(input_dir/'samples.csv') or any(sha256_file(Path(s['source_image']))!=s['source_sha256'] for s in sources): raise InspectionError('Input changed')
        if model_identity(weights)!=identities['pt'] or model_identity(ncnn,320)!=identities['ncnn']: raise InspectionError('Model changed')
        for filename,digest in provenance['files_sha256'].items():
            if sha256_file(reuse/filename)!=digest: raise InspectionError('Reused predictions changed')
        run['status']='complete'
    except (Exception,KeyboardInterrupt) as exc:
        run['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        write_report(output/'poses.json',dict(images=raw_rows,nonfinite_policy='Only nonfinite values become null; see separate quality flags. No clamping, interpolation or deletion.'))
        write_report(output/'quality_flags.json',dict(images=flag_rows,confidence_threshold=.5,gate_definition=run['quality_gate']))
        run['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        write_report(output/'run_info.json',run)
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('outputs/pilot01_preview'))
    parser.add_argument('--reuse-pt640',type=Path,default=Path('outputs/pilot01_pose_m2_1'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=generate(args.input_dir,args.output_dir,args.reuse_pt640)
    print(json.dumps({k:result[k] for k in ('status','errors','new_predict_calls','reused_predictions','images')}))
    sys.exit(0 if result['status']=='complete' else 1)
