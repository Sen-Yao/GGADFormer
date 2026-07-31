#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
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
    args.output_root.mkdir(parents=True, exist_ok=True)
    result = {'schema_version': 1, 'threshold': 0.005, 'datasets': {}}

    for dataset in ('Amazon', 't_finance'):
        numeric_path = args.output_root / f'{dataset}.token-equivalence.json'
        numeric_log = args.output_root / f'{dataset}.token-equivalence.log'
        numeric_command = [
            sys.executable,
            str(Path(__file__).with_name('validate_tokenization.py')),
            f'--dataset={dataset}',
            f'--output={numeric_path}',
        ]
        if run_command(numeric_command, repo_root, numeric_log) != 0:
            result['datasets'][dataset] = {'numeric_gate': 'failed'}
            atomic_write(args.output_root / 'gate.json', result)
            return 1

        modes = {
            'sequential_sparse': 'sequential_sparse',
            'reference': (
                'dense_recomputation' if dataset == 'Amazon' else 'sparse_recomputation'
            ),
        }
        outputs = {}
        for label, mode in modes.items():
            output_path = args.output_root / f'{dataset}.{label}.json'
            log_path = args.output_root / f'{dataset}.{label}.log'
            command = [
                sys.executable,
                str(repo_root / 'run.py'),
                *registry['methods']['VecGAD'][dataset],
                '--device=0',
                '--efficiency_mode',
                '--efficiency_evaluate',
                '--efficiency_warmup_epochs=10',
                '--efficiency_measure_epochs=30',
                '--efficiency_repeat=0',
                f'--tokenization_reference={mode}',
                f'--efficiency_output={output_path}',
            ]
            if run_command(command, repo_root, log_path) != 0:
                result['datasets'][dataset] = {
                    'numeric_gate': 'passed',
                    'downstream_gate': 'execution_failed',
                    'failed_mode': mode,
                }
                atomic_write(args.output_root / 'gate.json', result)
                return 1
            outputs[label] = json.loads(output_path.read_text(encoding='utf-8'))

        auc_delta = abs(
            outputs['sequential_sparse']['validation']['auc']
            - outputs['reference']['validation']['auc']
        )
        ap_delta = abs(
            outputs['sequential_sparse']['validation']['ap']
            - outputs['reference']['validation']['ap']
        )
        passed = auc_delta <= 0.005 and ap_delta <= 0.005
        result['datasets'][dataset] = {
            'numeric_gate': 'passed',
            'downstream_gate': 'passed' if passed else 'requires_five_seed_escalation',
            'auc_delta': auc_delta,
            'ap_delta': ap_delta,
            'sequential': outputs['sequential_sparse']['validation'],
            'reference': outputs['reference']['validation'],
        }

    atomic_write(args.output_root / 'gate.json', result)
    return 0 if all(
        item['downstream_gate'] == 'passed' for item in result['datasets'].values()
    ) else 2


if __name__ == '__main__':
    raise SystemExit(main())
