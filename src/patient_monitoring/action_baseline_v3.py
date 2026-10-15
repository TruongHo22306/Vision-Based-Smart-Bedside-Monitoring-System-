"""Experimental body-transport AND lower-body-change locomotion evidence.

All inputs are saved features/masks. No inference, fitting, identifiers as labels,
future frames, smoothing or automatic reseeding. V002 remains untouched.
"""
import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from . import action_baseline as v1
from .action_baseline_v2 import BaselineV2


@dataclass(frozen=True)
class LocomotionConfig:
    window_seconds: float = 1.0
    minimum_span_seconds: float = .8
    minimum_intervals: int = 3
    body_speed: float = .1
    log_scale_speed: float = .05
    ankle_range_moving: float = .15
    ankle_range_quiet: float = .1

    def __post_init__(self):
        if any(not v1.finite(x) or x<=0 for x in asdict(self).values()):
            raise ValueError('Config must be finite and positive')
        if self.minimum_span_seconds>self.window_seconds or self.ankle_range_quiet>=self.ankle_range_moving or type(self.minimum_intervals) is not int:
            raise ValueError('Invalid duration/range configuration')


def landmarks(row):
    """Use saved masks and finite coordinates; never rethreshold confidence."""
    data={}
    for key in ('hip_midpoint_px','shoulder_midpoint_px','torso_scale_px'):
        field=row['features'].get(key,{})
        value=field.get('value')
        vector=key!='torso_scale_px'
        if field.get('valid') is not True or (vector and (not isinstance(value,list) or len(value)!=2)):
            return None
        if not all(v1.finite(x) for x in (value if vector else [value])):
            return None
        data[key]=value
    if data['torso_scale_px']<=1e-9:return None
    points={j['index']:j for j in row.get('joints',[])}
    ankles=[]
    for idx in (15,16):
        j=points.get(idx,{})
        field=j.get('body_xy',{})
        xy=field.get('value')
        if (j.get('source_quality') or {}).get('passes_quality_gate') is not True or field.get('valid') is not True:
            return None
        if not isinstance(xy,list) or len(xy)!=2 or not all(v1.finite(x) for x in xy):return None
        ankles.append(xy)
    data['ankle_separation']=[ankles[0][axis]-ankles[1][axis] for axis in (0,1)]
    return data


class BaselineV3(BaselineV2):
    def __init__(self, config=v1.Config(), locomotion=LocomotionConfig()):
        super().__init__(config)
        self.locomotion=locomotion
        self.landmark_history=deque()

    def step(self,row):
        result=super().step(row)
        result['baseline_version']='action_rules_v003'
        result['v002_decision']={k:result[k] for k in ('predicted_label','reasons')}
        history=self.landmark_history;c=self.locomotion
        reset=result['window']['reset_reason']
        if reset:history.clear()
        data=landmarks(row)
        valid_target=row.get('target_status') in ('seeded','matched') and row.get('selected_source_detection_index') is not None
        valid_time=result['timestamp_valid'] and reset not in ('invalid_timestamp','nonincreasing_timestamp','invalid_frame_index')
        motion=dict(valid=False,reason='insufficient_contiguous_landmark_span',hip_speed=None,shoulder_speed=None,
                    log_scale_speed=None,ankle_range=None,body_transport=None,lower_body_change=None,
                    span_seconds=0.,reset_reason=reset)
        if data is None or not valid_target or not valid_time:
            history.clear();motion['reason']='missing_valid_landmarks_or_target_or_time'
        else:
            now=Fraction(row['time_base'])*row['pts']
            history.append((now,row['frame_index'],data))
            cutoff=now-Fraction(str(c.window_seconds))
            while history and history[0][0]<cutoff:history.popleft()
            span=now-history[0][0]
            motion.update(span_seconds=float(span),start_frame=history[0][1],end_frame=row['frame_index'])
            if span>=Fraction(str(c.minimum_span_seconds)) and len(history)-1>=c.minimum_intervals:
                a=history[0][2];b=data;scale=(a['torso_scale_px']+b['torso_scale_px'])/2
                hip=math.dist(a['hip_midpoint_px'],b['hip_midpoint_px'])/scale/float(span)
                shoulder=math.dist(a['shoulder_midpoint_px'],b['shoulder_midpoint_px'])/scale/float(span)
                depth=abs(math.log(b['torso_scale_px']/a['torso_scale_px']))/float(span)
                ankle=math.hypot(*(max(x[2]['ankle_separation'][axis] for x in history)-min(x[2]['ankle_separation'][axis] for x in history) for axis in (0,1)))
                transport=min(hip,shoulder)>=c.body_speed or depth>=c.log_scale_speed
                gait=ankle>=c.ankle_range_moving
                motion.update(valid=True,reason=None,hip_speed=hip,shoulder_speed=shoulder,log_scale_speed=depth,
                              ankle_range=ankle,body_transport=transport,lower_body_change=gait)
        result['landmark_motion']=motion
        # Preserve v002 non-upright handling and original missing-support outcomes.
        if result['posture_evidence'] in ('sitting','lying'):
            return result
        support=result['support']
        if not all(support[n]['valid'] for n in v1.FEATURES):return result
        if support['torso_inclination_deg']['value']>self.config.upright_max_deg:return result
        if not motion['valid']:
            label,reason='unknown','insufficient_contiguous_landmark_evidence'
        elif motion['body_transport'] and motion['lower_body_change']:
            label,reason='moving','body_transport_and_lower_body_change'
        elif (result['posture_evidence']=='standing' and not motion['body_transport'] and motion['ankle_range']<=c.ankle_range_quiet):
            label,reason='standing','standing_geometry_without_transport_and_quiet_lower_body'
        else:
            label,reason='unknown','locomotion_evidence_inconclusive'
        result.update(predicted_label=label,reasons=[reason],locomotion_evidence=dict(status='candidate' if label=='moving' else 'unknown',reason=reason))
        return result


def run_baseline(path,output,config=v1.Config(),locomotion=LocomotionConfig()):
    path,output=Path(path).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    settings=dict(base_config=asdict(config),locomotion=asdict(locomotion),policy='body_transport_AND_lower_body_change',
                  temporal='causal trailing actual PTS; no debounce; reset on v002 barriers or missing landmarks')
    digest=v1.sha256(path);config_hash=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    provenance=dict(status='incomplete',input_path=str(path),input_sha256=digest,config=settings,config_sha256=config_hash,
                    code_sha256=v1.sha256(__file__),v002_code_sha256=v1.sha256(Path(__file__).with_name('action_baseline_v2.py')),
                    v001_code_sha256=v1.sha256(v1.__file__),inference_performed=False,thresholds_validated=False,annotations_used=False)
    summary=dict(status='incomplete',completed=False,frame_count=0,errors=[])
    v1.write_json(output/'provenance.json',provenance)
    labels=Counter();reasons=Counter();transitions=[];previous=None;engine=BaselineV3(config,locomotion)
    try:
        with path.open(encoding='utf-8') as src,(output/'predictions.jsonl').open('x',encoding='utf-8') as dest:
            for line in src:
                row=json.loads(line);result=engine.step(row)
                result.update(input_sha256=digest,config_sha256=config_hash)
                dest.write(json.dumps(result,allow_nan=False)+'\n')
                labels[result['predicted_label']]+=1
                if result['predicted_label']=='unknown':reasons.update(result['reasons'])
                if previous is not None and previous!=result['predicted_label']:
                    transitions.append(dict(frame_index=result['frame_index'],presentation_time_seconds=result['presentation_time_seconds'],from_label=previous,to_label=result['predicted_label']))
                previous=result['predicted_label'];summary['frame_count']+=1
        assert v1.sha256(path)==digest
        if not summary['frame_count']:raise ValueError('Empty input')
        summary.update(status='complete',completed=True)
    except (Exception,KeyboardInterrupt) as exc:
        summary['errors'].append(repr(exc));raise
    finally:
        summary.update(label_counts={k:labels[k] for k in ('standing','sitting','lying','moving','unknown')},unknown_reason_counts=dict(reasons),transitions=transitions,transition_count=len(transitions))
        provenance.update(status=summary['status'],errors=summary['errors'])
        if (output/'predictions.jsonl').exists():provenance['output_sha256']=v1.sha256(output/'predictions.jsonl')
        v1.write_json(output/'summary.json',summary);v1.write_json(output/'provenance.json',provenance)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('features',type=Path);p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();print(run_baseline(a.features,a.output_dir))
