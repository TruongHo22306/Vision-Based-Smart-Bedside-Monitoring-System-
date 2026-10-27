"""Annotation-independent offline causal supervised prediction and timed review."""
import argparse
import json
import time
from collections import Counter
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .supervised_dataset import aligned_frames, csv_rows, make_window, sha, write
from .supervised_action import read, load_bundle, vectorize, transform, CLASSES


def windows(rows, bounds):
    """Grid anchored to first source PTS, including explicit startup decisions."""
    end = Fraction(0)
    while end <= bounds[-1]:
        start = end-1
        value, reason = (None, 'startup_less_than_one_second') if start < 0 else make_window(rows, bounds, start, end)
        yield dict(support_start=str(start), support_end=str(end), decision_time=str(end),
                   support_start_seconds=float(start), support_end_seconds=float(end),
                   decision_timestamp_seconds=float(end), available=value is not None,
                   availability_reason=reason, support=value)
        end += Fraction(1, 4)


def display_rows(rows, decisions):
    """Hold latest decision only after end; clear on any unavailable decision/track break."""
    pointer = 0
    latest = None
    reason = 'startup_less_than_one_second'
    origin = int(rows[0]['pts'])*Fraction(rows[0]['time_base'])
    for i, r in enumerate(rows):
        now = int(r['pts'])*Fraction(r['time_base'])-origin
        while pointer < len(decisions) and Fraction(decisions[pointer]['decision_time']) <= now:
            d = decisions[pointer]
            latest = d if d['available'] else None
            reason = d['availability_reason']
            pointer += 1
        invalid = r['target_status'] not in ('seeded', 'matched') or not r['timestamp_valid']
        broken = i > 0 and (r['target_status']=='seeded' or not r['temporal_pair']['valid'] or r['temporal_pair']['previous_frame_index'] != i-1)
        if invalid or broken:
            latest = None
            reason = 'current_target_unavailable' if invalid else 'current_track_break'
        yield dict(frame_index=r['frame_index'], pts=r['pts'], time_base=r['time_base'],
                   source_timestamp_seconds=float(now+origin),
                   displayed_label=latest['label'] if latest else None,
                   displayed_decision_time=latest['decision_time'] if latest else None,
                   unavailable_reason=None if latest else reason or 'awaiting_new_valid_window',
                   target_status=r['target_status'])


def render(picture, row):
    w,h=picture.size
    canvas=Image.new('RGB',(w,h+200),'#11151b');canvas.paste(picture,(0,0));draw=ImageDraw.Draw(canvas)
    lines=['SUPERVISED DEVELOPMENT DEMO | NOT GROUND TRUTH',
           f"Frame {row['frame_index']} | PTS {row['pts']} @ {row['time_base']} | {row['source_timestamp_seconds']:.6f}s",
           'Label: '+(row['displayed_label'] or 'UNAVAILABLE'),
           'Decision end: '+str(row['displayed_decision_time'])+'s (relative source clock)',
           'Target: '+row['target_status']+' | Reason: '+str(row['unavailable_reason']),
           '1s trailing / 0.25s grid | held causally; cleared at gaps',
           'Softmax scores uncalibrated | no smoothing / new threshold']
    for j,line in enumerate(lines):
        size=22
        while size>10 and draw.textlength(line,font=ImageFont.load_default(size=size))>w-16:size-=1
        draw.text((8,h+6+j*27),line,font=ImageFont.load_default(size=size),fill='#ffe44d' if j==2 else 'white')
    return canvas


def export_review(source, rows, display, timeline, output):
    mapping=[];durations={}
    with av.open(str(source)) as src,av.open(str(output/'review.mp4'),'w',options={'movflags':'+faststart','video_track_timescale':str(Fraction(rows[0]['time_base']).denominator)}) as dst:
        stream=src.streams.video[0];base=Fraction(stream.time_base)
        enc=dst.add_stream('libx264',rate=stream.average_rate or 30);enc.width=stream.width;enc.height=stream.height+200
        enc.pix_fmt='yuv420p';enc.time_base=enc.codec_context.time_base=base;enc.codec_context.max_b_frames=0
        enc.options={'crf':'18','preset':'fast'}
        def mux(packet):
            duration=durations[packet.pts*Fraction(packet.time_base)]/Fraction(packet.time_base)
            if duration.denominator!=1:raise ValueError('Nonintegral packet duration')
            packet.duration=duration.numerator;dst.mux(packet)
        for i,(frame,r,d,t) in enumerate(zip_longest(src.decode(stream),rows,display,timeline)):
            if any(x is None for x in (frame,r,d,t)):raise ValueError('Source/display frame count mismatch')
            if (i,frame.pts,Fraction(frame.time_base)) != (r['frame_index'],r['pts'],Fraction(r['time_base'])):raise ValueError('Source PTS mismatch')
            if frame.duration!=int(t['source_duration']) or frame.is_corrupt:raise ValueError('Source duration/corrupt mismatch')
            if d['displayed_decision_time'] is not None and Fraction(d['displayed_decision_time']) > frame.pts*Fraction(frame.time_base)-int(rows[0]['pts'])*Fraction(rows[0]['time_base']):raise ValueError('Future display')
            durations[frame.pts*Fraction(frame.time_base)]=frame.duration*Fraction(frame.time_base)
            picture=render(frame.to_image(),d)
            if i in (0,120):picture.save(output/f'preview_{i:06d}.png')
            f=av.VideoFrame.from_image(picture);f.pts=frame.pts;f.time_base=frame.time_base;f.duration=frame.duration
            mapping.append(dict(frame_index=i,pts=frame.pts,time_base=str(frame.time_base),duration=frame.duration))
            for packet in enc.encode(f):mux(packet)
            if i%250==0:print('Review encoded',i,flush=True)
        for packet in enc.encode():mux(packet)
    checked=0
    with av.open(str(output/'review.mp4')) as src:
        for f,m in zip_longest(src.decode(video=0),mapping):
            if f is None or m is None or (f.pts,str(f.time_base),f.duration)!=(m['pts'],m['time_base'],m['duration']):raise ValueError('Export PTS/duration mismatch')
            checked+=1
    write(output/'review_timing_verification.json',dict(verified_frames=checked,all_integer_pts_timebase_durations_exact=True,source_panels='native dimensions, separate footer; video only',output_sha256=sha(output/'review.mp4')))


def run(features, timeline, model_dir, output_dir, source=None):
    started=time.perf_counter();out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    info=dict(status='incomplete',errors=[],scope='development demonstration, NOT held-out evaluation',metrics_computed=False)
    write(out/'summary.json',info)
    try:
        model_dir=Path(model_dir);frozen=read(model_dir/'frozen_manifest.json')
        if frozen['status']!='frozen_complete':raise ValueError('Model not frozen complete')
        for name,h in frozen['artifact_sha256'].items():
            if sha(model_dir/name)!=h:raise ValueError('Frozen artifact changed: '+name)
        inputs={str(p):sha(p) for p in [Path(features),Path(timeline),model_dir/'frozen_manifest.json']}
        if source:inputs[str(source)]=sha(source)
        rows=[json.loads(x) for x in Path(features).read_text(encoding='utf-8').splitlines()]
        if len({r['video_id'] for r in rows})!=1:raise ValueError('Multiple videos')
        tl=csv_rows(timeline);bounds=aligned_frames(rows,tl)
        for i,t in enumerate(tl):
            end=bounds[i]+int(t['source_duration'])*Fraction(t['time_base'])
            if end!=bounds[i+1]:raise ValueError('Noncontiguous source duration support')
        if source and Path(rows[0]['source_path']).resolve()!=Path(source).resolve():raise ValueError('Source path mismatch')
        model,prep,schema=load_bundle(model_dir)
        decisions=[];prediction_started=time.perf_counter()
        for index,d in enumerate(windows(rows,bounds)):
            d.update(window_index=index,label=None,softmax_scores_uncalibrated=None,video_id=rows[0]['video_id'],
                     source_origin_pts=rows[0]['pts'],source_time_base=rows[0]['time_base'],
                     absolute_decision_timestamp_seconds=float(Fraction(d['decision_time'])+int(rows[0]['pts'])*Fraction(rows[0]['time_base'])))
            if d['available']:
                x=transform(vectorize([d['support']],schema),prep)
                with torch.no_grad():logits=model(torch.from_numpy(x));scores=logits.softmax(1)[0]
                if not torch.isfinite(scores).all():raise ValueError('Nonfinite model scores')
                d['label']=CLASSES[int(logits.argmax(1).item())];d['softmax_scores_uncalibrated']=dict(zip(CLASSES,scores.tolist()))
                d['partial_coverage_channels']=[k for k,v in d['support']['model_inputs'].items() if v['coverage']<1-1e-9]
                d['missing_channels']=[k for k,v in d['support']['model_inputs'].items() if not v['valid']]
            decisions.append(d)
        prediction_seconds=time.perf_counter()-prediction_started
        display=list(display_rows(rows,decisions))
        for name,records in [('window_predictions.jsonl',decisions),('frame_display.jsonl',display)]:
            with (out/name).open('x',encoding='utf-8') as f:
                for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
        if source:export_review(source,rows,display,tl,out)
        for p,h in inputs.items():
            if sha(p)!=h:raise ValueError('Input changed during execution')
        for name,h in frozen['artifact_sha256'].items():
            if sha(model_dir/name)!=h:raise ValueError('Frozen artifact changed during execution')
        info.update(status='complete',video_id=rows[0]['video_id'],source_frames=len(rows),scheduled_windows=len(decisions),
            available_windows=sum(d['available'] for d in decisions),availability_reasons=dict(Counter(d['availability_reason'] for d in decisions if not d['available'])),
            window_label_counts=dict(Counter(d['label'] for d in decisions if d['available'])),
            frame_display_counts=dict(Counter(d['displayed_label'] or 'unavailable' for d in display)),
            partial_feature_coverage_windows=sum(bool(d.get('partial_coverage_channels')) for d in decisions),
            prediction_seconds=prediction_seconds,total_seconds=time.perf_counter()-started,input_sha256=inputs,
            source_code_sha256={str(Path(__file__)):sha(__file__)},frozen_model_artifacts_verified=True,
            preprocessing_fitted=False,annotation_inputs=False,window_policy='relative first PTS grid0,0.25,...<=exact end; [end-1,end); startup explicit; no off-grid tail window',
            display_policy='At frame PTS consume decisions whose end<=PTS; newest unavailable clears label. Current unresolved/reseed/invalid pair clears held label until a new valid decision. No anticipation or smoothing; no display at an earlier frame for final decision at source end.',
            maximum_grid_tail_seconds=float(bounds[-1]-Fraction(decisions[-1]['decision_time'])))
    except BaseException as e:info['errors'].append(repr(e));raise
    finally:
        info['total_seconds']=time.perf_counter()-started;write(out/'summary.json',info)
    print(json.dumps(info),flush=True)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',required=True);p.add_argument('--timeline',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--source',help='optional source video for timestamp-preserving review export')
    a=p.parse_args();run(a.features,a.timeline,a.model_dir,a.output_dir,a.source)


if __name__=='__main__':main()
