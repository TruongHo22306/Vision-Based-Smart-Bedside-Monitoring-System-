"""Video-only v002 review from saved predictions/targets, preserving source timing."""
import argparse
import csv
from fractions import Fraction
from itertools import zip_longest
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from .action_baseline import sha256, write_json
from .pose_review import gate_edges, read_records
from .target_selection import valid_box
from .video_inspection import COLUMNS, TimelineAudit


FOOTER = 180


def display_lines(prediction, target):
    reasons = prediction['reasons']
    descriptions = {
        'supported_posture_independent_of_image_motion': 'supported posture; image motion assessed separately',
        'quiet_with_geometry_support': 'quiet image motion with geometry support',
        'upright_with_temporal_motion': 'upright with image motion (walking not verified)',
        'motion_threshold_band': 'image motion in uncertain threshold band',
        'motion_posture_conflict': 'image motion / posture conflict',
        'ambiguous_posture_geometry': 'insufficient posture geometry',
    }
    reason = ('insufficient feature evidence / time coverage' if any(r.startswith('insufficient:') for r in reasons)
              else '; '.join(descriptions.get(r, r) for r in reasons))
    cautions = list(prediction.get('image_motion_cautions', []))
    cautions += [r for r in reasons if r in ('motion_threshold_band', 'motion_posture_conflict') and r not in cautions]
    warnings = '; '.join(descriptions.get(r, r) for r in cautions) or 'none recorded'
    return [
        'EXPERIMENTAL v002 PREDICTIONS - NOT GROUND TRUTH',
        f"{prediction['video_id']} | frame {prediction['frame_index']} | source {prediction['presentation_time_seconds']:.6f}s | PTS {prediction['pts']} @ {prediction['time_base']}",
        f"Predicted: {prediction['predicted_label']} | posture evidence: {prediction['posture_evidence'] or 'unknown'} | target D{target['selected_source_detection_index']}",
        f'Reason: {reason}',
        f"Image motion: {prediction['motion_evidence'] or 'unknown'} | Warning: {warnings}",
        f"Locomotion: {prediction['locomotion_evidence']['status']} | yellow = saved target; joint gate is not accuracy",
    ]


def render(frame_image, prediction, target):
    image = Image.new('RGB', (frame_image.width, frame_image.height + FOOTER), '#10151b')
    image.paste(frame_image, (0, 0))
    draw = ImageDraw.Draw(image)
    person = target.get('selected_person')
    if person is not None:
        if valid_box(person['box_xyxy'], frame_image.width, frame_image.height) is not None:
            raise ValueError('Saved selected box is invalid')
        draw.rectangle(person['box_xyxy'], outline='#ffe44d', width=3)
        points = person['keypoints']
        for a, b in gate_edges(points):
            draw.line((points[a]['x'], points[a]['y'], points[b]['x'], points[b]['y']), fill='#ffe44d', width=3)
        for point in points:
            if point['passes_quality_gate']:
                x, y = point['x'], point['y']
                draw.ellipse((x-3, y-3, x+3, y+3), fill='#ffe44d')
    for n, line in enumerate(display_lines(prediction, target)):
        font = ImageFont.load_default(size=22)
        # Fit complete text; never crop a warning or paint the source image with a footer.
        size = 22
        while draw.textlength(line, font=font) > image.width-16 and size > 8:
            size -= 1
            font = ImageFont.load_default(size=size)
        if draw.textlength(line, font=font) > image.width-16:
            raise ValueError('Overlay text cannot fit; refusing clipped review text')
        color = '#ffb454' if n == 4 else '#ffe44d' if n == 2 else 'white'
        draw.text((8, frame_image.height+5+n*29), line, font=font, fill=color)
    return image


def check_alignment(frame, timing, prediction, target, video, stream_index):
    if (prediction.get('baseline_version') != 'action_rules_v002'
            or any(timing[k] != prediction[k] or prediction[k] != target[k] for k in COLUMNS)
            or any(Path(r['source_path']).resolve() != video for r in (prediction, target))
            or any(r['stream_index'] != stream_index for r in (prediction, target))
            or any((r['width'],r['height']) != (frame.width,frame.height) for r in (prediction,target))
            or prediction['selected_source_detection_index'] != target['selected_source_detection_index']
            or prediction['target_status'] != target['status'] or frame.is_corrupt):
        raise ValueError(f"Video/prediction/target alignment mismatch at frame {timing['frame_index']}")
    if frame.duration is None or frame.duration <= 0:
        raise ValueError('Missing source duration; do not invent frame timing')


def export_review(video, predictions, targets, output):
    video, predictions, targets, output = (Path(p).resolve() for p in (video,predictions,targets,output))
    if output.exists():
        raise ValueError('Output exists; choose a NEW directory')
    provenance_path = predictions.parent/'provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance['status'] != 'complete' or provenance['baseline_version'] != 'action_rules_v002':
        raise ValueError('Need complete saved v002 predictions')
    if provenance['output_sha256'] != sha256(predictions):
        raise ValueError('Prediction checksum differs from saved provenance')
    hashes = {str(p):sha256(p) for p in (video,predictions,targets,provenance_path)}
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='incomplete', errors=[], aligned_frames=0, verified_output_frames=0,
        input_sha256=hashes, config=provenance['config'], config_sha256=provenance['config_sha256'],
        baseline_code_sha256=provenance['code_sha256'], review_code_sha256=sha256(__file__),
        timing_policy='Original PTS and per-frame duration; no rebasing, fixed-FPS resampling or frame drops. Verify by decoding output.',
        encoding=dict(codec='libx264',crf=18,preset='fast',b_frames=0,pixel_format='yuv420p'),
        audio='Video-only review. Original audio and source unchanged.',
        inference_performed=False, annotations_confirmed=False, human_playback_review='pending',
        versions=dict(av=av.__version__), preview_frames=[])
    write_json(output/'review_manifest.json',manifest)
    mapping = []
    durations = {}
    outfile = output/'review.mp4'
    try:
        with av.open(str(video)) as source, av.open(str(outfile),'w',options={'movflags':'+faststart'}) as dest:
            first = next(read_records(predictions))
            stream = next(s for s in source.streams.video if s.index == first['stream_index'])
            base = Fraction(stream.time_base)
            encoded = dest.add_stream('libx264', rate=stream.average_rate or 30)
            encoded.width, encoded.height = stream.width, stream.height+FOOTER
            encoded.pix_fmt='yuv420p'
            encoded.time_base = encoded.codec_context.time_base = base
            encoded.codec_context.max_b_frames=0
            encoded.options={'crf':'18','preset':'fast'}
            manifest['source_dimensions']=[stream.width,stream.height]
            manifest['output_dimensions']=[encoded.width,encoded.height]
            audit=TimelineAudit()

            def mux(packet):
                stamp=packet.pts*Fraction(packet.time_base)
                duration=durations[stamp]/Fraction(packet.time_base)
                if duration.denominator!=1:
                    raise ValueError('Packet duration cannot exactly represent source timing')
                packet.duration=duration.numerator
                dest.mux(packet)

            warning_preview=False
            for frame,pred,target in zip_longest(source.decode(stream),read_records(predictions),read_records(targets)):
                if any(v is None for v in (frame,pred,target)):
                    raise ValueError('Source/prediction/target frame counts differ')
                timing=audit.observe(frame.pts,frame.time_base)
                check_alignment(frame,timing,pred,target,video,stream.index)
                if pred['config_sha256'] != manifest['config_sha256']:
                    raise ValueError('Per-frame config hash mismatch')
                manifest['aligned_frames']+=1
                durations[frame.pts*Fraction(frame.time_base)]=frame.duration*Fraction(frame.time_base)
                rendered=render(frame.to_image(),pred,target)
                if timing['frame_index']==0 or (pred.get('image_motion_cautions') and not warning_preview):
                    name=f"preview_{timing['frame_index']:06d}.png"
                    rendered.save(output/name)
                    manifest['preview_frames'].append(name)
                    warning_preview=warning_preview or bool(pred.get('image_motion_cautions'))
                outframe=av.VideoFrame.from_image(rendered)
                outframe.pts, outframe.time_base, outframe.duration=frame.pts,frame.time_base,frame.duration
                mapping.append(dict(**timing,source_duration=frame.duration,source_duration_seconds=float(frame.duration*Fraction(frame.time_base)),
                                    predicted_label=pred['predicted_label'],selected_source_detection_index=pred['selected_source_detection_index']))
                for packet in encoded.encode(outframe):mux(packet)
                if manifest['aligned_frames']%100==0:print(f"Review encoded {manifest['aligned_frames']} frames",flush=True)
            for packet in encoded.encode():mux(packet)
        if not mapping:raise ValueError('Empty source video')
        with av.open(str(outfile)) as result:
            for actual, expected in zip_longest(result.decode(video=0),mapping):
                if actual is None or expected is None:raise ValueError('Review frame count differs')
                if (actual.pts!=expected['pts'] or Fraction(actual.time_base)!=Fraction(expected['time_base'])
                        or actual.duration!=expected['source_duration']):
                    raise ValueError(f"Output PTS/duration mismatch at {expected['frame_index']}")
                manifest['verified_output_frames']+=1
                expected.update(output_pts=actual.pts,output_time_base=str(actual.time_base),output_duration=actual.duration)
        for path,digest in hashes.items():
            if sha256(path)!=digest:raise ValueError('Input changed during export')
        manifest.update(status='complete',all_pts_and_durations_exact=True,output_sha256=sha256(outfile),
                        verified_end_seconds=float((mapping[-1]['pts']+mapping[-1]['source_duration'])*Fraction(mapping[-1]['time_base'])))
    except (Exception,KeyboardInterrupt) as exc:
        manifest['errors'].append(f'{type(exc).__name__}: {exc}')
    if mapping:
        fields=list(dict.fromkeys(k for row in mapping for k in row))
        with (output/'frame_mapping.csv').open('x',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(mapping)
    write_json(output/'review_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--targets',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    try:result=export_review(args.video,args.predictions,args.targets,args.output_dir)
    except (ValueError,OSError,KeyError) as exc:parser.exit(1,f'Review failed: {exc}\n')
    print(f"Review {result['status']}: aligned={result['aligned_frames']}, verified={result['verified_output_frames']}")
    if result['status']!='complete':parser.exit(1,f"{result['errors']}\n")


if __name__=='__main__':main()
