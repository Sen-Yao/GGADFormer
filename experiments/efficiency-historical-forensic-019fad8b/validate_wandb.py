#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from run_wandb_trial import load_trials


REQUIRED_SUMMARY_KEYS = (
    'terminal_status',
    'child_returncode',
    'offline/tokenization_seconds',
    'training/tqdm_terminal_rate_it_per_second',
    'training/synchronized_block_seconds',
    'training/synchronized_epoch_seconds',
    'training/synchronized_throughput_it_per_second',
    'training/optimizer_steps_per_epoch',
)


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--sweep-path', required=True)
    parser.add_argument('--expected-harness-sha', required=True)
    parser.add_argument('--expected-protocol-sha', required=True)
    parser.add_argument('--expected-host', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    expected_trials = {trial['id']: trial for trial in load_trials(args.registry)}
    sweep = wandb.Api(timeout=30).sweep(args.sweep_path)
    runs = list(sweep.runs)
    errors = []
    snapshots = []
    seen_trial_ids = []
    for run in runs:
        config = dict(run.config)
        summary = dict(run.summary)
        trial_id = config.get('trial_id')
        seen_trial_ids.append(trial_id)
        trial = expected_trials.get(trial_id)
        if trial is None:
            errors.append(f'{run.id}: unknown trial_id {trial_id!r}')
            continue
        expected_config = {
            'trial_id': trial_id,
            'source_key': trial['source_key'],
            'source_commit': trial['source_commit'],
            'batch_mode': trial['batch_mode'],
            'batch_size': trial['batch_size'],
            'dataset': trial['dataset'],
            'repeat': trial['repeat'],
            'seed': trial['seed'],
            'num_nodes': trial['num_nodes'],
            'num_workers': trial['loader']['num_workers'],
            'persistent_workers': trial['loader']['persistent_workers'],
            'pin_memory': trial['loader']['pin_memory'],
            'warmup_epochs': trial['warmup_epochs'],
            'measured_epochs': trial['measured_epochs'],
            'execution_host': args.expected_host,
            'harness_sha': args.expected_harness_sha,
            'protocol_sha': args.expected_protocol_sha,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                errors.append(
                    f'{trial_id}: config {key} mismatch: '
                    f'expected {expected!r}, got {config.get(key)!r}'
                )
        if run.state != 'finished':
            errors.append(f'{trial_id}: W&B state is {run.state!r}')
        if summary.get('terminal_status') != 'completed':
            errors.append(f'{trial_id}: terminal status is not completed')
        if summary.get('child_returncode') != 0:
            errors.append(f'{trial_id}: child return code is not zero')
        for key in REQUIRED_SUMMARY_KEYS:
            if key not in summary:
                errors.append(f'{trial_id}: missing W&B summary key {key}')
        sidecar_path = args.output_root / 'wandb' / f'{trial_id}.json'
        if not sidecar_path.is_file():
            errors.append(f'{trial_id}: missing local W&B sidecar')
        else:
            sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
            if sidecar.get('wandb', {}).get('run_id') != run.id:
                errors.append(f'{trial_id}: live W&B/sidecar run identity mismatch')
        snapshots.append({
            'run_id': run.id,
            'name': run.name,
            'state': run.state,
            'config': {key: config.get(key) for key in sorted(expected_config)},
            'summary': {key: summary.get(key) for key in REQUIRED_SUMMARY_KEYS},
        })

    expected_ids = set(expected_trials)
    observed_ids = set(seen_trial_ids)
    if len(runs) != len(expected_trials):
        errors.append(f'expected {len(expected_trials)} W&B runs, got {len(runs)}')
    if len(seen_trial_ids) != len(observed_ids):
        errors.append('duplicate W&B trial_id values')
    if observed_ids != expected_ids:
        errors.append(
            f'W&B trial coverage mismatch: missing={sorted(expected_ids - observed_ids)!r}, '
            f'extra={sorted(observed_ids - expected_ids)!r}'
        )
    if sweep.state != 'FINISHED':
        errors.append(f'W&B sweep state is {sweep.state!r}, expected FINISHED')

    output = {
        'schema_version': 1,
        'valid': not errors,
        'errors': errors,
        'sweep_path': args.sweep_path,
        'sweep_state': sweep.state,
        'expected_runs': len(expected_trials),
        'observed_runs': len(runs),
        'runs': sorted(snapshots, key=lambda item: item['config']['trial_id']),
    }
    atomic_write(args.output, output)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
