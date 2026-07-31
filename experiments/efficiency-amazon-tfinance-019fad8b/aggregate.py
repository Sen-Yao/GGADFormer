#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

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
    return {
        'count': len(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'p25': percentile(values, 0.25),
        'p75': percentile(values, 0.75),
    }


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
    args = parser.parse_args()

    trials = load_trials(args.registry)
    errors = []
    terminal = []
    artifact_digests = {}
    for trial in trials:
        attempts = []
        for attempt in (1, 2):
            path = args.output_root / 'raw' / f"{trial['id']}.attempt{attempt}.json"
            if not path.exists():
                continue
            artifact_digests[str(path.relative_to(args.output_root))] = sha256(path)
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f'{trial["id"]}: invalid JSON in attempt {attempt}: {error}')
                continue
            attempts.append(payload)
            if payload.get('method') != trial['method']:
                errors.append(f'{trial["id"]}: method identity mismatch')
            if payload.get('dataset') != trial['dataset']:
                errors.append(f'{trial["id"]}: dataset identity mismatch')
            if payload.get('repeat') != trial['repeat']:
                errors.append(f'{trial["id"]}: repeat identity mismatch')

        if not attempts:
            errors.append(f'{trial["id"]}: missing all attempts')
            continue
        completed = [item for item in attempts if item.get('status') == 'completed']
        if completed:
            if len(completed) != 1:
                errors.append(f'{trial["id"]}: multiple completed attempts')
                continue
            result = completed[0]
            runtime = result.get('runtime', {})
            gpu = runtime.get('gpu') or {}
            if runtime.get('hostname') != args.expected_hostname:
                errors.append(f'{trial["id"]}: unexpected hostname {runtime.get("hostname")!r}')
            if gpu.get('name') != args.expected_gpu:
                errors.append(f'{trial["id"]}: unexpected GPU {gpu.get("name")!r}')
            epochs = result.get('training', {}).get('epoch_seconds', [])
            if len(epochs) != trial['measured_epochs'] or any(value <= 0 for value in epochs):
                errors.append(f'{trial["id"]}: invalid measured epoch vector')
            terminal.append({
                'trial': trial,
                'status': 'completed',
                'offline_seconds': result['offline']['seconds'],
                'epoch_seconds': epochs,
                'epoch_median_seconds': statistics.median(epochs) if epochs else None,
                'source_attempt': attempts.index(result) + 1,
            })
        elif len(attempts) == 2 and all(item.get('status') == 'gpu_oom' for item in attempts):
            terminal.append({'trial': trial, 'status': 'gpu_oom'})
        else:
            statuses = [item.get('status') for item in attempts]
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
            completed = [item for item in matching if item['status'] == 'completed']
            if len(completed) == 3:
                cell['offline_seconds'] = summarize([item['offline_seconds'] for item in completed])
                cell['per_run_epoch_median_seconds'] = summarize(
                    [item['epoch_median_seconds'] for item in completed]
                )
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
