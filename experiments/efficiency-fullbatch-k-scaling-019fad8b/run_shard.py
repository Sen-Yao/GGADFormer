#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def load_trials(registry_path):
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    trials = []
    for method, datasets in registry['methods'].items():
        for dataset, entry in datasets.items():
            method_args = entry['args'] if isinstance(entry, dict) else entry
            num_nodes = entry.get('num_nodes') if isinstance(entry, dict) else None
            for repeat in registry['repeats']:
                trials.append({
                    'id': f'{method.lower()}-{dataset.lower()}-r{repeat}',
                    'method': method,
                    'dataset': dataset,
                    'repeat': repeat,
                    'num_nodes': num_nodes,
                    'args': method_args,
                    'warmup_epochs': registry['warmup_epochs'],
                    'measured_epochs': registry['measured_epochs'],
                })
    return trials


def run_attempt(repo_root, rho_root, rho_data_root, output_root, trial, attempt):
    result_path = output_root / 'raw' / f"{trial['id']}.attempt{attempt}.json"
    log_path = output_root / 'logs' / f"{trial['id']}.attempt{attempt}.log"
    if trial['method'] == 'RHO':
        command = [
            sys.executable,
            str(Path(__file__).resolve().with_name('rho_efficiency.py')),
            f'--rho-root={rho_root}',
            f'--rho-data-root={rho_data_root}',
            *trial['args'],
            '--cuda=0',
            f"--warmup-epochs={trial['warmup_epochs']}",
            f"--measured-epochs={trial['measured_epochs']}",
            f"--repeat={trial['repeat']}",
            f'--output={result_path}',
        ]
        cwd = rho_data_root
    else:
        command = [
            sys.executable,
            str(repo_root / 'run.py'),
            *trial['args'],
            '--device=0',
            '--efficiency_mode',
            f"--efficiency_warmup_epochs={trial['warmup_epochs']}",
            f"--efficiency_measure_epochs={trial['measured_epochs']}",
            f"--efficiency_repeat={trial['repeat']}",
            f'--efficiency_output={result_path}',
        ]
        cwd = repo_root
    environment = os.environ.copy()
    environment['WANDB_MODE'] = 'disabled'
    environment['PYTHONUNBUFFERED'] = '1'
    with log_path.open('w', encoding='utf-8') as log_stream:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = None
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
    return process.returncode, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--rho-root', type=Path, required=True)
    parser.add_argument('--rho-data-root', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--method', action='append', choices=['VecGAD', 'GGAD', 'RHO'])
    parser.add_argument('--attempt-offset', type=int, default=0)
    parser.add_argument('--summary-label')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'raw').mkdir(exist_ok=True)
    (args.output_root / 'logs').mkdir(exist_ok=True)
    assigned = [
        trial for index, trial in enumerate(load_trials(args.registry))
        if index % args.num_shards == args.shard_index
        and (not args.method or trial['method'] in args.method)
    ]
    shard_summary = {'schema_version': 1, 'shard_index': args.shard_index, 'trials': []}
    failed = False
    for trial in assigned:
        attempts = []
        for attempt in (args.attempt_offset + 1, args.attempt_offset + 2):
            returncode, result = run_attempt(
                repo_root,
                args.rho_root.resolve(),
                args.rho_data_root.resolve(),
                args.output_root,
                trial,
                attempt,
            )
            attempts.append({
                'attempt': attempt,
                'returncode': returncode,
                'status': result.get('status') if result else 'missing_result',
            })
            if returncode == 0 and result and result.get('status') == 'completed':
                break
            if not result or result.get('status') != 'gpu_oom':
                break
        terminal_status = attempts[-1]['status']
        if terminal_status not in ('completed', 'gpu_oom'):
            failed = True
        shard_summary['trials'].append({'trial': trial, 'attempts': attempts})

    summary_label = f'.{args.summary_label}' if args.summary_label else ''
    summary_path = args.output_root / f'shard-{args.shard_index}{summary_label}.json'
    summary_path.write_text(json.dumps(shard_summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
