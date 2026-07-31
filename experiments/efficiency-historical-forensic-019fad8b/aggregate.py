#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_wandb_trial import load_trials


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
    return {
        'count': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'sample_std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'p25': percentile(values, 0.25),
        'p75': percentile(values, 0.75),
    }


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--aggregate-output', type=Path)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--expected-gpu', required=True)
    parser.add_argument('--expected-physical-gpus', required=True)
    parser.add_argument('--expected-sweep-id', required=True)
    parser.add_argument('--expected-harness-sha', required=True)
    parser.add_argument('--expected-protocol-sha', required=True)
    args = parser.parse_args()
    expected_gpus = {
        int(value) for value in args.expected_physical_gpus.split(',') if value
    }
    trials = load_trials(args.registry)
    errors = []
    terminal = []
    artifacts = {}
    for trial in trials:
        raw_path = args.output_root / 'raw' / f"{trial['id']}.json"
        sidecar_path = args.output_root / 'wandb' / f"{trial['id']}.json"
        if not raw_path.is_file():
            errors.append(f"{trial['id']}: missing raw result")
            continue
        if not sidecar_path.is_file():
            errors.append(f"{trial['id']}: missing W&B sidecar")
            continue
        artifacts[str(raw_path.relative_to(args.output_root))] = sha256(raw_path)
        artifacts[str(sidecar_path.relative_to(args.output_root))] = sha256(sidecar_path)
        try:
            result = json.loads(raw_path.read_text(encoding='utf-8'))
            sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{trial['id']}: invalid JSON: {error}")
            continue
        status = result.get('status')
        if sidecar.get('terminal_status') != status:
            errors.append(f"{trial['id']}: raw/sidecar terminal status mismatch")
        identity_checks = {
            'trial_id': trial['id'],
            'source_commit': trial['source_commit'],
            'batch_mode': trial['batch_mode'],
            'dataset': trial['dataset'],
            'repeat': trial['repeat'],
        }
        for key, expected in identity_checks.items():
            observed = result.get(key)
            if key == 'source_commit' and status == 'completed':
                observed = result.get('source', {}).get('commit')
            elif key == 'batch_mode' and status == 'completed':
                observed = result.get('config', {}).get('batch_mode')
            if observed != expected:
                errors.append(
                    f"{trial['id']}: {key} mismatch: expected {expected!r}, got {observed!r}"
                )
        if sidecar.get('physical_gpu_index') not in expected_gpus:
            errors.append(f"{trial['id']}: unexpected physical GPU")
        if sidecar.get('wandb', {}).get('sweep_id') != args.expected_sweep_id:
            errors.append(f"{trial['id']}: W&B sweep mismatch")
        if sidecar.get('harness_sha') != args.expected_harness_sha:
            errors.append(f"{trial['id']}: harness SHA mismatch")
        if sidecar.get('protocol_sha') != args.expected_protocol_sha:
            errors.append(f"{trial['id']}: protocol SHA mismatch")
        if status == 'completed':
            runtime = result.get('runtime', {})
            gpu = runtime.get('gpu', {})
            config = result.get('config', {})
            training = result.get('training', {})
            if runtime.get('hostname') != args.expected_hostname:
                errors.append(f"{trial['id']}: hostname mismatch")
            if gpu.get('name') != args.expected_gpu or gpu.get('index') != 0:
                errors.append(f"{trial['id']}: child GPU identity mismatch")
            contract = trial['loader']
            for key in ('num_workers', 'persistent_workers', 'pin_memory'):
                if config.get(key) != contract[key]:
                    errors.append(f"{trial['id']}: loader {key} mismatch")
            if config.get('batch_size') != trial['batch_size']:
                errors.append(f"{trial['id']}: batch-size mismatch")
            if config.get('batch_mode') != trial['batch_mode']:
                errors.append(f"{trial['id']}: batch-mode mismatch")
            if config.get('num_nodes') != trial['num_nodes']:
                errors.append(f"{trial['id']}: node-count mismatch")
            expected_steps = math.ceil(trial['num_nodes'] / trial['batch_size'])
            if training.get('optimizer_steps_per_epoch') != expected_steps:
                errors.append(f"{trial['id']}: optimizer-step mismatch")
            async_epochs = training.get('measured_async_epoch_seconds', [])
            if len(async_epochs) != trial['measured_epochs'] or any(
                value <= 0 for value in async_epochs
            ):
                errors.append(f"{trial['id']}: invalid async epoch vector")
            for key in (
                'tqdm_terminal_rate_it_per_second',
                'synchronized_block_seconds',
                'synchronized_epoch_seconds',
                'synchronized_throughput_it_per_second',
            ):
                if not isinstance(training.get(key), (int, float)) or training[key] <= 0:
                    errors.append(f"{trial['id']}: invalid timing metric {key}")
            block = training.get('synchronized_block_seconds', 0)
            epoch = training.get('synchronized_epoch_seconds', 0)
            throughput = training.get('synchronized_throughput_it_per_second', 0)
            if block > 0 and not math.isclose(
                epoch, block / trial['measured_epochs'], rel_tol=1e-10
            ):
                errors.append(f"{trial['id']}: synchronized epoch derivation mismatch")
            if block > 0 and not math.isclose(
                throughput, trial['measured_epochs'] / block, rel_tol=1e-10
            ):
                errors.append(f"{trial['id']}: synchronized throughput derivation mismatch")
        elif status != 'gpu_oom':
            errors.append(f"{trial['id']}: invalid terminal status {status!r}")
        terminal.append({
            'trial': trial,
            'status': status,
            'result': result,
            'physical_gpu_index': sidecar.get('physical_gpu_index'),
            'wandb_run_id': sidecar.get('wandb', {}).get('run_id'),
        })

    cells = []
    for source_key in sorted({trial['source_key'] for trial in trials}):
        for batch_mode in ('native', 'fullbatch'):
            for dataset in ('Amazon', 't_finance'):
                matching = [
                    item for item in terminal
                    if item['trial']['source_key'] == source_key
                    and item['trial']['batch_mode'] == batch_mode
                    and item['trial']['dataset'] == dataset
                ]
                statuses = [item['status'] for item in matching]
                cell = {
                    'source_key': source_key,
                    'source_commit': (
                        matching[0]['trial']['source_commit'] if matching else None
                    ),
                    'batch_mode': batch_mode,
                    'dataset': dataset,
                    'statuses': statuses,
                    'physical_gpu_indices': [
                        item['physical_gpu_index'] for item in matching
                    ],
                    'wandb_run_ids': [item['wandb_run_id'] for item in matching],
                }
                completed = [item for item in matching if item['status'] == 'completed']
                if len(completed) == 3:
                    trainings = [item['result']['training'] for item in completed]
                    offlines = [item['result']['offline'] for item in completed]
                    cell.update({
                        'batch_size': completed[0]['trial']['batch_size'],
                        'optimizer_steps_per_epoch': (
                            trainings[0]['optimizer_steps_per_epoch']
                        ),
                        'tokenization_seconds': summarize([
                            item['tokenization_seconds'] for item in offlines
                        ]),
                        'tqdm_terminal_rate_it_per_second': summarize([
                            item['tqdm_terminal_rate_it_per_second'] for item in trainings
                        ]),
                        'synchronized_block_seconds': summarize([
                            item['synchronized_block_seconds'] for item in trainings
                        ]),
                        'synchronized_epoch_seconds': summarize([
                            item['synchronized_epoch_seconds'] for item in trainings
                        ]),
                        'synchronized_throughput_it_per_second': summarize([
                            item['synchronized_throughput_it_per_second']
                            for item in trainings
                        ]),
                        'tqdm_to_synchronized_throughput_ratio': summarize([
                            item['tqdm_terminal_rate_it_per_second']
                            / item['synchronized_throughput_it_per_second']
                            for item in trainings
                        ]),
                        'gpu_allocated_peak_bytes': summarize([
                            item['gpu_allocated_peak_bytes'] for item in trainings
                        ]),
                        'gpu_reserved_peak_bytes': summarize([
                            item['gpu_reserved_peak_bytes'] for item in trainings
                        ]),
                    })
                elif len(matching) == 3 and statuses and all(
                    status == 'gpu_oom' for status in statuses
                ):
                    cell['terminal_status'] = 'gpu_oom'
                else:
                    errors.append(
                        f'{source_key}/{batch_mode}/{dataset}: incomplete coverage {statuses!r}'
                    )
                cells.append(cell)

    output = {
        'schema_version': 1,
        'valid': not errors,
        'errors': errors,
        'expected_trials': len(trials),
        'terminal_trials': len(terminal),
        'cells': cells,
        'artifact_sha256': artifacts,
    }
    aggregate_path = args.aggregate_output or (args.output_root / 'aggregate.json')
    atomic_write(aggregate_path, output)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
