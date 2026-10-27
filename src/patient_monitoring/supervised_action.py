"""Frozen first supervised experiment; all metrics are training/resubstitution."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .supervised_dataset import sha, write

CONFIG_HASH = '6ff4a96bd537b370ad9026126b0e85ac69ae903634fbdd2fac7692e9948d33aa'
SCHEMA_HASH = 'ef1087563130864efebb8b9a5a830fdaaa7d4281aaaffd29720a85710e998500'
CHANNELS = ['left_knee_deg', 'right_knee_deg', 'torso_deg', 'hip_vx_per_s',
            'hip_vy_per_s', 'box_width_fraction', 'box_height_fraction']
CLASSES = ['standing', 'sitting', 'lying', 'moving']
ORDER = [f'{c}.{s}' for c in CHANNELS for s in ('mean', 'std')] + [f'{c}.coverage' for c in CHANNELS] + [f'{c}.missing' for c in CHANNELS]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_schema(schema):
    if (schema['version'] != 'action_linear_input_v001' or
        [c['name'] for c in schema['channels']] != CHANNELS or
        schema['column_order'] != ORDER or schema['input_dimensions'] != 28 or
        schema['continuous_columns'] != ORDER[:14] or
        schema['coverage_columns'] != ORDER[14:21] or schema['missing_columns'] != ORDER[21:]):
        raise ValueError('Frozen feature schema/order mismatch')


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def vectorize(rows, schema):
    validate_schema(schema)
    if not rows:
        raise ValueError('No rows')
    result = []
    for i, row in enumerate(rows):
        try:
            inputs = row['model_inputs']
            if set(inputs) != set(CHANNELS):
                raise ValueError('channel keys mismatch')
            values, coverages, missing = [], [], []
            for c in CHANNELS:
                f = inputs[c]
                valid, mean, std, coverage = (f[k] for k in ('valid', 'mean', 'std', 'coverage'))
                if type(valid) is not bool or not number(coverage) or not -1e-9 <= coverage <= 1+1e-9:
                    raise ValueError('invalid mask/coverage')
                if valid:
                    if not number(mean) or not number(std) or std < 0 or coverage <= 0:
                        raise ValueError('invalid finite values/valid support')
                    values.extend([mean, std])
                else:
                    if mean is not None or std is not None or abs(coverage) > 1e-9:
                        raise ValueError('inconsistent null/mask/coverage')
                    values.extend([np.nan, np.nan])
                coverages.append(coverage)
                missing.append(float(not valid))
            result.append(values + coverages + missing)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Row {i}: {exc}') from exc
    return np.asarray(result, dtype=np.float64)


def fit_preprocessing(training_x, schema):
    validate_schema(schema)
    if training_x.ndim != 2 or training_x.shape[1] != 28 or not len(training_x):
        raise ValueError('Invalid training matrix')
    observed = training_x[:, :14]
    fallback = np.isnan(observed).all(axis=0)
    medians = np.array([0.0 if fallback[j] else np.median(observed[~np.isnan(observed[:, j]), j]) for j in range(14)])
    filled = np.where(np.isnan(observed), medians, observed)
    centers, scales = filled.mean(axis=0), filled.std(axis=0, ddof=0)
    constant = scales == 0
    scales[constant] = 1.0
    return dict(version='training_median_scaler_v001', feature_order=ORDER.copy(), class_order=CLASSES.copy(),
                medians=medians.tolist(), all_missing_fallback=fallback.tolist(),
                centers=centers.tolist(), scales=scales.tolist(), constant_columns=constant.tolist(),
                fit_row_count=len(training_x), fit_scope='training only',
                missing_counts=np.isnan(observed).sum(axis=0).tolist(),
                all_channels_missing_rows=int((training_x[:, 21:] == 1).all(axis=1).sum()))


def transform(x, prep):
    if prep['feature_order'] != ORDER or prep['class_order'] != CLASSES:
        raise ValueError('Preprocessing schema/class mismatch')
    if x.ndim != 2 or x.shape[1] != 28:
        raise ValueError('Invalid matrix dimensions')
    arrays = [np.asarray(prep[k], dtype=np.float64) for k in ('medians', 'centers', 'scales')]
    if any(a.shape != (14,) or not np.isfinite(a).all() for a in arrays) or (arrays[2] <= 0).any():
        raise ValueError('Invalid persisted preprocessing')
    y = x.copy()
    y[:, :14] = (np.where(np.isnan(y[:, :14]), arrays[0], y[:, :14])-arrays[1])/arrays[2]
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite transformed features')
    return y


def save_bundle(output, model, prep, schema_path, config_path):
    torch.save(model.state_dict(), output/'model_state.pt')
    write(output/'preprocessing.json', prep)
    shutil.copyfile(schema_path, output/'feature_schema.json')
    shutil.copyfile(config_path, output/'config.json')
    hashes = {n: sha(output/n) for n in ('model_state.pt', 'preprocessing.json', 'feature_schema.json', 'config.json')}
    write(output/'model_hashes.json', hashes)
    return hashes


def load_bundle(output, expected_schema_hash=SCHEMA_HASH):
    output = Path(output)
    hashes = read(output/'model_hashes.json')
    for name in ('model_state.pt', 'preprocessing.json', 'feature_schema.json', 'config.json'):
        if sha(output/name) != hashes[name]:
            raise ValueError('Bundle hash mismatch: '+name)
    if hashes['feature_schema.json'] != expected_schema_hash or hashes['config.json'] != CONFIG_HASH:
        raise ValueError('Frozen schema/config hash mismatch')
    schema = read(output/'feature_schema.json')
    validate_schema(schema)
    prep = read(output/'preprocessing.json')
    transform(np.zeros((1, 28)), prep)  # validates persisted order/scales; never fits
    model = torch.nn.Linear(28, 4, bias=True, device='cpu', dtype=torch.float64)
    model.load_state_dict(torch.load(output/'model_state.pt', map_location='cpu', weights_only=True), strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite saved weights')
    model.eval()
    return model, prep, schema


def metrics(truth, predicted):
    matrix = np.zeros((4, 4), dtype=int)
    for a, b in zip(truth, predicted):
        matrix[a, b] += 1
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp, support, predicted_count = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        fp, fn = predicted_count-tp, support-tp
        denom = 2*tp+fp+fn
        per_class[name] = dict(tp=tp, fp=fp, fn=fn, support=support, predicted_count=predicted_count,
            precision=tp/predicted_count if predicted_count else None,
            recall=tp/support if support else None, f1=2*tp/denom if denom else None,
            denominators=dict(precision=predicted_count, recall=support, f1=denom),
            undefined_reasons={k: reason for k, d, reason in [('precision', predicted_count, 'no predicted positives'), ('recall', support, 'no true positives support'), ('f1', denom, 'no true or predicted support')] if not d})
    return dict(metric_scope='training/resubstitution ONLY; not held-out', class_order=CLASSES,
                confusion_matrix_rows_true_columns_predicted=matrix.tolist(), window_count=len(truth),
                correct=int(matrix.trace()), accuracy=float(matrix.trace()/len(truth)) if len(truth) else None,
                per_class=per_class)


def preflight(config_path):
    config_path = Path(config_path)
    folder = config_path.parent
    pins = read(folder/'config_hash.json')
    if sha(config_path) != CONFIG_HASH or pins['config_sha256'] != CONFIG_HASH:
        raise ValueError('Prepared config hash mismatch')
    for name, digest in read(folder/'checksums.json').items():
        if sha(folder/name) != digest:
            raise ValueError('Preparation checksum mismatch: '+name)
    config, schema = read(config_path), read(folder/'feature_schema.json')
    if sha(folder/'feature_schema.json') != SCHEMA_HASH or pins['schema_sha256'] != SCHEMA_HASH:
        raise ValueError('Prepared schema hash mismatch')
    validate_schema(schema)
    dataset = Path(config['dataset'])
    if sha(dataset) != config['dataset_sha256'] or sha(dataset) != pins['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    manifest = read(dataset.parent/'manifest.json')
    if sha(dataset.parent/'manifest.json') != config['dataset_manifest_sha256'] or manifest['config_sha256'] != config['dataset_config_sha256']:
        raise ValueError('Dataset manifest/config mismatch')
    if hashlib.sha256(json.dumps(manifest['config'], sort_keys=True).encode()).hexdigest() != manifest['config_sha256']:
        raise ValueError('Dataset config content mismatch')
    for path, digest in manifest['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Dataset input changed: '+path)
    for name, digest in manifest['output_sha256'].items():
        if sha(dataset.parent/name) != digest:
            raise ValueError('Dataset output changed: '+name)
    if read(dataset.parent/'summary.json')['status'] != 'complete':
        raise ValueError('Incomplete dataset')
    if torch.__version__ != '2.9.1+cpu' or np.__version__ != '2.2.6':
        raise ValueError('Prepared numerical dependency versions changed')
    rows = [json.loads(s) for s in dataset.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 1183 or Counter(r['label'] for r in rows) != Counter(dict(standing=345, sitting=297, lying=333, moving=208)):
        raise ValueError('Unexpected training rows/classes')
    keys = {(r['video_id'], r['segment_id'], r['start'], r['end']) for r in rows}
    if len(keys) != len(rows):
        raise ValueError('Duplicate windows')
    expected_groups = read(dataset.parent.parent/'occasion_support.json')
    for group, expected in expected_groups.items():
        if Counter(r['label'] for r in rows if r['recording_occasion_group_id']==group) != Counter(expected['windows']):
            raise ValueError('Occasion support mismatch')
    return config, schema, rows


def train(config_path, output_dir):
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    info = dict(status='incomplete', metric_scope='training/resubstitution', training_steps_completed=0,
                errors=[], started_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    write(output/'run_info.json', info)
    history = []
    try:
        config, schema, rows = preflight(config_path)
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        repro = config['reproducibility']
        torch.set_num_threads(repro['num_threads']); torch.set_num_interop_threads(repro['num_interop_threads'])
        torch.use_deterministic_algorithms(repro['deterministic_algorithms'])
        raw = vectorize(rows, schema)
        prep = fit_preprocessing(raw, schema)
        x = torch.from_numpy(transform(raw, prep))
        y = torch.tensor([CLASSES.index(r['label']) for r in rows], dtype=torch.int64)
        model = torch.nn.Linear(28, 4, bias=True, device='cpu', dtype=torch.float64)
        o = config['optimization']
        optimizer = torch.optim.Adam(model.parameters(), lr=o['lr'], betas=tuple(o['betas']), eps=o['eps'],
                                     weight_decay=o['weight_decay'], foreach=o['foreach'], fused=o['fused'])
        def loss_values():
            logits = model(x)
            ce = torch.nn.functional.cross_entropy(logits, y, reduction='mean')
            penalty = o['l2']/2 * model.weight.square().sum()
            return ce+penalty, ce, penalty
        def measure(step):
            with torch.no_grad():
                loss, ce, penalty = loss_values()
            if not all(torch.isfinite(v) for v in (loss, ce, penalty)):
                raise FloatingPointError(f'Nonfinite objective step {step}')
            record = dict(step=step, objective=loss.item(), cross_entropy=ce.item(), regularization=penalty.item(), metric_scope='training/resubstitution')
            history.append(record)
            return record
        initial = measure(0)
        optimization_started = time.perf_counter()
        for step in range(1, o['steps']+1):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_values()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss before step {step}')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Nonfinite gradient step {step}')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError(f'Nonfinite weights step {step}')
            info['training_steps_completed'] = step
            rec = measure(step)
            if step == 1 or step % 100 == 0:
                print(f"Training/resubstitution step {step}/1000 objective={rec['objective']:.9f} CE={rec['cross_entropy']:.9f}", flush=True)
                write(output/'run_info.json', info)
                write(output/'loss_history.json', history)
        optimization_seconds = time.perf_counter()-optimization_started
        model.eval()
        with torch.no_grad():
            logits = model(x); predicted = logits.argmax(dim=1).numpy(); probs = logits.softmax(dim=1).numpy()
        hashes = save_bundle(output, model, prep, Path(config_path).parent/'feature_schema.json', Path(config_path))
        reloaded, prep2, schema2 = load_bundle(output)
        with torch.no_grad():
            reload_logits = reloaded(torch.from_numpy(transform(vectorize(rows, schema2), prep2)))
        torch.testing.assert_close(reload_logits, logits, rtol=1e-12, atol=1e-12)
        if not torch.equal(reload_logits.argmax(1), logits.argmax(1)):
            raise ValueError('Reload predictions differ')
        truth = y.numpy()
        diagnostics = dict(overall=metrics(truth, predicted), per_video={}, per_occasion={},
            scores='Softmax scores are not calibrated confidence', missingness=prep['missing_counts'],
            all_channels_missing_rows=prep['all_channels_missing_rows'],
            coverage={c:dict(min=float(raw[:,14+i].min()),max=float(raw[:,14+i].max())) for i,c in enumerate(CHANNELS)})
        for key, dest in [('video_id','per_video'),('recording_occasion_group_id','per_occasion')]:
            for value in sorted({r[key] for r in rows}):
                ids = [i for i,r in enumerate(rows) if r[key]==value]
                diagnostics[dest][value] = metrics(truth[ids], predicted[ids])
        write(output/'training_diagnostics.json', diagnostics)
        with (output/'training_predictions.jsonl').open('x', encoding='utf-8') as f:
            for i,r in enumerate(rows):
                record = {k:r[k] for k in ('video_id','subject_id','recording_occasion_group_id','segment_id','start','end','frame_indices','observed_frame_times')}
                record.update(row_index=i,metric_scope='training/resubstitution',true_label=r['label'],predicted_label=CLASSES[predicted[i]],softmax_scores_uncalibrated=dict(zip(CLASSES,probs[i].tolist())))
                f.write(json.dumps(record, allow_nan=False)+'\n')
        info.update(status='complete',training_rows=len(rows),seed=config['seed'],device=str(model.weight.device),dtype=str(model.weight.dtype),
            threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            python=platform.python_version(),packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            source_sha256={str(Path(__file__)):sha(__file__)},input_config_sha256=sha(config_path),dataset_sha256=config['dataset_sha256'],
            schema_sha256=hashes['feature_schema.json'],parameter_count=sum(p.numel() for p in model.parameters()),model_file_bytes=(output/'model_state.pt').stat().st_size,
            optimization_seconds=optimization_seconds,elapsed_seconds=time.perf_counter()-started,
            reload_predictions_equal=True,reload_max_abs_logit_difference=(reload_logits-logits).abs().max().item(),
            reload_tolerances=dict(rtol=1e-12,atol=1e-12),initial_objective=initial['objective'],final_objective=history[-1]['objective'],
            loss_increase_steps=sum(b['objective']>a['objective'] for a,b in zip(history,history[1:])),
            prepared_config_status_note='Copied config retains preparation-time status/no_training_executed; this run_info records actual execution.',
            split='none; all rows development; no held-out metrics',no_tuning=True)
        # Recheck source artifacts after fitting; do not run another training if changed.
        if sha(config['dataset']) != config['dataset_sha256'] or sha(config_path) != CONFIG_HASH:
            raise ValueError('Inputs changed during training')
    except BaseException as exc:
        info.update(status='incomplete',elapsed_seconds=time.perf_counter()-started)
        info['errors'].append(repr(exc))
        raise
    finally:
        write(output/'loss_history.json', history)
        write(output/'run_info.json', info)
    write(output/'frozen_manifest.json',dict(status='frozen_complete',artifact_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
          purpose='Future evaluation with no refit/tuning; all present metrics training/resubstitution'))
    print(json.dumps(dict(status=info['status'],training_accuracy=diagnostics['overall']['accuracy'],elapsed_seconds=info['elapsed_seconds'],output=str(output))),flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--output-dir',required=True)
    args = parser.parse_args()
    train(args.config, args.output_dir)


if __name__ == '__main__':
    main()
