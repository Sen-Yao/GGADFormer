#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def atomic_write(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_command(command, cwd, log_path):
    environment = os.environ.copy()
    environment['WANDB_MODE'] = 'disabled'
    environment['PYTHONUNBUFFERED'] = '1'
    with log_path.open('w', encoding='utf-8') as stream:
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    registry = json.loads(args.registry.read_text(encoding='utf-8'))
    base_args = [
        argument for argument in registry['methods']['VecGAD']['Amazon']
        if not argument.startswith('--seed=')
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_results = []

    for seed in range(5):
        outputs = {}
        modes = {
            'sequential_sparse': 'sequential_sparse',
            'reference': 'dense_recomputation',
        }
        for label, mode in modes.items():
            output_path = args.output_root / f'amazon-seed{seed}.{label}.json'
            log_path = args.output_root / f'amazon-seed{seed}.{label}.log'
            command = [
                sys.executable,
                str(repo_root / 'run.py'),
                *base_args,
                f'--seed={seed}',
                '--device=0',
                '--efficiency_mode',
                '--efficiency_evaluate',
                '--efficiency_warmup_epochs=10',
                '--efficiency_measure_epochs=30',
                '--efficiency_repeat=0',
                f'--tokenization_reference={mode}',
                f'--efficiency_output={output_path}',
            ]
            returncode = run_command(command, repo_root, log_path)
            if returncode != 0:
                atomic_write(args.output_root / 'five-seed-gate.json', {
                    'schema_version': 1,
                    'status': 'execution_failed',
                    'seed': seed,
                    'mode': mode,
                    'returncode': returncode,
                })
                return 1
            outputs[label] = json.loads(output_path.read_text(encoding='utf-8'))

        sequential = outputs['sequential_sparse']['validation']
        reference = outputs['reference']['validation']
        seed_results.append({
            'seed': seed,
            'sequential': sequential,
            'reference': reference,
            'auc_signed_delta': sequential['auc'] - reference['auc'],
            'ap_signed_delta': sequential['ap'] - reference['ap'],
        })

    auc_deltas = [item['auc_signed_delta'] for item in seed_results]
    ap_deltas = [item['ap_signed_delta'] for item in seed_results]
    auc_mean = statistics.mean(auc_deltas)
    ap_mean = statistics.mean(ap_deltas)
    threshold = 0.005
    passed = abs(auc_mean) <= threshold and abs(ap_mean) <= threshold
    result = {
        'schema_version': 1,
        'status': 'passed' if passed else 'failed',
        'criterion': 'absolute mean paired signed delta <= threshold',
        'threshold': threshold,
        'seeds': seed_results,
        'summary': {
            'auc_mean_signed_delta': auc_mean,
            'auc_std_signed_delta': statistics.stdev(auc_deltas),
            'auc_max_absolute_seed_delta': max(abs(value) for value in auc_deltas),
            'ap_mean_signed_delta': ap_mean,
            'ap_std_signed_delta': statistics.stdev(ap_deltas),
            'ap_max_absolute_seed_delta': max(abs(value) for value in ap_deltas),
        },
    }
    atomic_write(args.output_root / 'five-seed-gate.json', result)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
