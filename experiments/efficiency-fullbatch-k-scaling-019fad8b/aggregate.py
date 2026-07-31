#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import re

from run_shard import load_trials


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values):
    summary = {
        'count': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'p25': percentile(values, 0.25),
        'p75': percentile(values, 0.75),
    }
    summary['sample_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def nested(payload, *keys):
    value = payload
    for key in keys:
        value = value[key]
    return value


def atomic_write(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--expected-gpu', required=True)
    parser.add_argument('--expected-physical-gpus', required=True)
    parser.add_argument('--expected-sweep-id', required=True)
    parser.add_argument('--expected-code-sha', required=True)
    parser.add_argument('--expected-protocol-sha', required=True)
    args = parser.parse_args()
    expected_physical_gpus = {
        int(value) for value in args.expected_physical_gpus.split(',') if value
    }

    trials = load_trials(args.registry)
    errors = []
    terminal = []
    artifact_digests = {}
    for trial in trials:
        sidecar_path = args.output_root / 'wandb' / f"{trial['id']}.json"
        sidecar = None
        if not sidecar_path.is_file():
            errors.append(f'{trial["id"]}: missing W&B sidecar')
        else:
            artifact_digests[str(sidecar_path.relative_to(args.output_root))] = sha256(sidecar_path)
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f'{trial["id"]}: invalid W&B sidecar: {error}')
            if sidecar is not None:
                if sidecar.get('trial_id') != trial['id']:
                    errors.append(f'{trial["id"]}: W&B trial identity mismatch')
                if sidecar.get('physical_gpu_index') not in expected_physical_gpus:
                    errors.append(f'{trial["id"]}: unexpected physical GPU index')
                if sidecar.get('wandb', {}).get('sweep_id') != args.expected_sweep_id:
                    errors.append(f'{trial["id"]}: W&B sweep identity mismatch')
                if sidecar.get('code_sha') != args.expected_code_sha:
                    errors.append(f'{trial["id"]}: code SHA mismatch')
                if sidecar.get('protocol_sha') != args.expected_protocol_sha:
                    errors.append(f'{trial["id"]}: protocol SHA mismatch')
        attempts = []
        attempt_paths = sorted(
            (args.output_root / 'raw').glob(f"{trial['id']}.attempt*.json"),
            key=lambda path: int(re.search(r'attempt(\d+)\.json$', path.name).group(1)),
        )
        for path in attempt_paths:
            attempt = int(re.search(r'attempt(\d+)\.json$', path.name).group(1))
            artifact_digests[str(path.relative_to(args.output_root))] = sha256(path)
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f'{trial["id"]}: invalid JSON in attempt {attempt}: {error}')
                continue
            attempts.append((attempt, payload))
            if payload.get('method') != trial['method']:
                errors.append(f'{trial["id"]}: method identity mismatch')
            if payload.get('dataset') != trial['dataset']:
                errors.append(f'{trial["id"]}: dataset identity mismatch')
            if payload.get('repeat') != trial['repeat']:
                errors.append(f'{trial["id"]}: repeat identity mismatch')

        if not attempts:
            errors.append(f'{trial["id"]}: missing all attempts')
            continue
        completed = [item for item in attempts if item[1].get('status') == 'completed']
        if completed:
            if len(completed) != 1:
                errors.append(f'{trial["id"]}: multiple completed attempts')
                continue
            source_attempt, result = completed[0]
            runtime = result.get('runtime', {})
            gpu = runtime.get('gpu') or {}
            if runtime.get('hostname') != args.expected_hostname:
                errors.append(f'{trial["id"]}: unexpected hostname {runtime.get("hostname")!r}')
            if gpu.get('name') != args.expected_gpu:
                errors.append(f'{trial["id"]}: unexpected GPU {gpu.get("name")!r}')
            if gpu.get('index') != 0:
                errors.append(f'{trial["id"]}: child must see local GPU 0, got {gpu.get("index")!r}')
            epochs = result.get('training', {}).get('epoch_seconds', [])
            if len(epochs) != trial['measured_epochs'] or any(value <= 0 for value in epochs):
                errors.append(f'{trial["id"]}: invalid measured epoch vector')
            if trial['method'] == 'VecGAD' and result.get('config', {}).get(
                'tokenization_reference'
            ) != 'sequential_sparse':
                errors.append(f'{trial["id"]}: non-formal tokenization mode')
            training = result.get('training', {})
            config = result.get('config', {})
            if training.get('optimizer_steps_per_epoch') != 1:
                errors.append(f'{trial["id"]}: expected exactly one optimizer step per epoch')
            expected_batch_size = 0 if trial['method'] == 'RHO' else trial['num_nodes']
            if training.get('batch_size') != expected_batch_size:
                errors.append(
                    f'{trial["id"]}: expected batch size {expected_batch_size}, '
                    f'got {training.get("batch_size")!r}'
                )
            train_ratio_key = 'train_ratio' if trial['method'] == 'RHO' else 'train_rate'
            if config.get(train_ratio_key) != 0.05:
                errors.append(f'{trial["id"]}: expected 5% training ratio')
            split_seed_key = 'data_split_seed'
            if config.get(split_seed_key) != 42:
                errors.append(f'{trial["id"]}: expected data split seed 42')
            if sidecar is not None and sidecar.get('terminal_status') != 'completed':
                errors.append(f'{trial["id"]}: W&B sidecar disagrees with completed result')
            terminal.append({
                'trial': trial,
                'status': 'completed',
                'offline_seconds': result['offline']['seconds'],
                'tokenization_seconds': result['offline']['tokenization_seconds'],
                'epoch_seconds': epochs,
                'epoch_median_seconds': statistics.median(epochs) if epochs else None,
                'metrics': {
                    'offline_rss_baseline_bytes': nested(result, 'offline', 'rss', 'baseline_bytes'),
                    'offline_rss_peak_bytes': nested(result, 'offline', 'rss', 'peak_bytes'),
                    'offline_rss_delta_bytes': nested(result, 'offline', 'rss', 'delta_bytes'),
                    'offline_gpu_allocated_peak_bytes': nested(
                        result, 'offline', 'gpu_peak', 'peak', 'allocated_bytes'
                    ),
                    'offline_gpu_reserved_peak_bytes': nested(
                        result, 'offline', 'gpu_peak', 'peak', 'reserved_bytes'
                    ),
                    'offline_gpu_allocated_delta_bytes': nested(
                        result, 'offline', 'gpu_peak', 'delta', 'allocated_bytes'
                    ),
                    'offline_gpu_reserved_delta_bytes': nested(
                        result, 'offline', 'gpu_peak', 'delta', 'reserved_bytes'
                    ),
                    'training_rss_baseline_bytes': nested(
                        result, 'training', 'rss', 'baseline_bytes'
                    ),
                    'training_rss_peak_bytes': nested(result, 'training', 'rss', 'peak_bytes'),
                    'training_rss_delta_bytes': nested(result, 'training', 'rss', 'delta_bytes'),
                    'training_gpu_allocated_peak_bytes': nested(
                        result, 'training', 'gpu_peak', 'peak', 'allocated_bytes'
                    ),
                    'training_gpu_reserved_peak_bytes': nested(
                        result, 'training', 'gpu_peak', 'peak', 'reserved_bytes'
                    ),
                    'training_gpu_allocated_delta_bytes': nested(
                        result, 'training', 'gpu_peak', 'delta', 'allocated_bytes'
                    ),
                    'training_gpu_reserved_delta_bytes': nested(
                        result, 'training', 'gpu_peak', 'delta', 'reserved_bytes'
                    ),
                    'token_payload_bytes': nested(result, 'offline', 'token_payload_bytes'),
                    'parameter_count': nested(result, 'model', 'parameter_count'),
                    'trainable_parameter_count': nested(
                        result, 'model', 'trainable_parameter_count'
                    ),
                },
                'source_attempt': source_attempt,
                'physical_gpu_index': sidecar.get('physical_gpu_index') if sidecar else None,
                'wandb_run_id': sidecar.get('wandb', {}).get('run_id') if sidecar else None,
            })
        elif len(attempts) >= 2 and all(item[1].get('status') == 'gpu_oom' for item in attempts):
            if sidecar is not None and sidecar.get('terminal_status') != 'gpu_oom':
                errors.append(f'{trial["id"]}: W&B sidecar disagrees with OOM result')
            terminal.append({
                'trial': trial,
                'status': 'gpu_oom',
                'physical_gpu_index': sidecar.get('physical_gpu_index') if sidecar else None,
                'wandb_run_id': sidecar.get('wandb', {}).get('run_id') if sidecar else None,
            })
        else:
            statuses = [item[1].get('status') for item in attempts]
            errors.append(f'{trial["id"]}: invalid terminal attempts {statuses!r}')

    cells = []
    for method in sorted({trial['method'] for trial in trials}):
        for dataset in sorted({trial['dataset'] for trial in trials if trial['method'] == method}):
            matching = [
                item for item in terminal
                if item['trial']['method'] == method and item['trial']['dataset'] == dataset
            ]
            statuses = [item['status'] for item in matching]
            cell = {'method': method, 'dataset': dataset, 'statuses': statuses}
            cell['physical_gpu_indices'] = [item['physical_gpu_index'] for item in matching]
            cell['wandb_run_ids'] = [item['wandb_run_id'] for item in matching]
            completed = [item for item in matching if item['status'] == 'completed']
            if len(completed) == 3:
                cell['offline_seconds'] = summarize([item['offline_seconds'] for item in completed])
                cell['tokenization_seconds'] = summarize(
                    [item['tokenization_seconds'] for item in completed]
                )
                cell['per_run_epoch_median_seconds'] = summarize(
                    [item['epoch_median_seconds'] for item in completed]
                )
                cell['all_measured_epoch_seconds'] = summarize([
                    value for item in completed for value in item['epoch_seconds']
                ])
                cell['resources'] = {
                    metric: summarize([item['metrics'][metric] for item in completed])
                    for metric in completed[0]['metrics']
                }
            elif statuses and all(status == 'gpu_oom' for status in statuses):
                cell['terminal_status'] = 'gpu_oom'
            else:
                errors.append(f'{method}/{dataset}: incomplete repeat coverage {statuses!r}')
            cells.append(cell)

    output = {
        'schema_version': 1,
        'valid': not errors,
        'errors': errors,
        'expected_trials': len(trials),
        'terminal_trials': len(terminal),
        'cells': cells,
        'artifact_sha256': artifact_digests,
    }
    aggregate_path = args.output_root / 'aggregate.json'
    atomic_write(aggregate_path, output)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
