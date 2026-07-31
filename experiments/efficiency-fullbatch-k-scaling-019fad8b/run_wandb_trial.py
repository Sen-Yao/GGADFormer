#!/usr/bin/env python3
import json
import os
from pathlib import Path
import socket
import statistics

import wandb

from run_shard import load_trials, run_attempt


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'missing required environment variable: {name}')
    return value


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def completed_metrics(result):
    epochs = result['training']['epoch_seconds']
    return {
        'offline/prepare_seconds': result['offline']['seconds'],
        'offline/tokenization_seconds': result['offline']['tokenization_seconds'],
        'offline/rss_peak_gib': result['offline']['rss']['peak_bytes'] / 1024 ** 3,
        'offline/gpu_allocated_peak_gib': (
            result['offline']['gpu_peak']['peak']['allocated_bytes'] / 1024 ** 3
        ),
        'offline/gpu_reserved_peak_gib': (
            result['offline']['gpu_peak']['peak']['reserved_bytes'] / 1024 ** 3
        ),
        'offline/token_payload_mib': result['offline']['token_payload_bytes'] / 1024 ** 2,
        'training/epoch_median_seconds': statistics.median(epochs),
        'training/epoch_mean_seconds': statistics.mean(epochs),
        'training/rss_peak_gib': result['training']['rss']['peak_bytes'] / 1024 ** 3,
        'training/gpu_allocated_peak_gib': (
            result['training']['gpu_peak']['peak']['allocated_bytes'] / 1024 ** 3
        ),
        'training/gpu_reserved_peak_gib': (
            result['training']['gpu_peak']['peak']['reserved_bytes'] / 1024 ** 3
        ),
        'training/optimizer_steps_per_epoch': result['training']['optimizer_steps_per_epoch'],
        'model/parameter_count': result['model']['parameter_count'],
        'model/trainable_parameter_count': result['model']['trainable_parameter_count'],
    }


def main():
    registry_path = Path(
        os.environ.get('VECGAD_EFFICIENCY_REGISTRY', EXPERIMENT_DIR / 'registry.json')
    ).resolve()
    output_root = Path(require_env('VECGAD_EFFICIENCY_OUTPUT_ROOT')).resolve()
    rho_root = Path(require_env('VECGAD_RHO_ROOT')).resolve()
    rho_data_root = Path(require_env('VECGAD_RHO_DATA_ROOT')).resolve()
    physical_gpu_index = int(require_env('VECGAD_AGENT_GPU_INDEX'))
    visible_devices = require_env('CUDA_VISIBLE_DEVICES')
    if visible_devices != str(physical_gpu_index):
        raise RuntimeError(
            f'GPU binding mismatch: CUDA_VISIBLE_DEVICES={visible_devices!r}, '
            f'agent index={physical_gpu_index}'
        )

    run = wandb.init(
        entity=os.environ.get('WANDB_ENTITY', 'HCCS'),
        project=os.environ.get('WANDB_PROJECT', 'VecGAD'),
        job_type='efficiency-fullbatch-trial',
        tags=['efficiency', 'full-batch', 'cold-start', 'hccs90'],
    )
    exit_code = 0
    try:
        trial_id = str(run.config['trial_id'])
        trials = {trial['id']: trial for trial in load_trials(registry_path)}
        if trial_id not in trials:
            raise RuntimeError(f'unknown W&B trial_id: {trial_id}')
        trial = trials[trial_id]
        run.name = trial_id
        run.config.update({
            'method': trial['method'],
            'dataset': trial['dataset'],
            'repeat': trial['repeat'],
            'num_nodes': trial['num_nodes'],
            'warmup_epochs': trial['warmup_epochs'],
            'measured_epochs': trial['measured_epochs'],
            'full_batch_required': True,
            'execution_host': 'HCCS-90',
            'physical_gpu_index': physical_gpu_index,
            'code_sha': require_env('VECGAD_CODE_SHA'),
            'protocol_sha': require_env('VECGAD_PROTOCOL_SHA'),
        }, allow_val_change=True)

        attempts = []
        terminal_result = None
        for attempt in (1, 2):
            returncode, result = run_attempt(
                REPO_ROOT, rho_root, rho_data_root, output_root, trial, attempt
            )
            status = result.get('status') if result else 'missing_result'
            attempts.append({'attempt': attempt, 'returncode': returncode, 'status': status})
            if returncode == 0 and status == 'completed':
                terminal_result = result
                break
            if status != 'gpu_oom':
                break

        if terminal_result is not None:
            terminal_status = 'completed'
        elif len(attempts) == 2 and all(item['status'] == 'gpu_oom' for item in attempts):
            terminal_status = 'gpu_oom'
        else:
            terminal_status = 'error'

        sidecar = {
            'schema_version': 1,
            'trial_id': trial_id,
            'method': trial['method'],
            'dataset': trial['dataset'],
            'repeat': trial['repeat'],
            'terminal_status': terminal_status,
            'attempts': attempts,
            'hostname': socket.gethostname(),
            'physical_gpu_index': physical_gpu_index,
            'cuda_visible_devices': visible_devices,
            'wandb': {
                'entity': run.entity,
                'project': run.project,
                'sweep_id': run.sweep_id,
                'run_id': run.id,
                'run_url': run.url,
            },
            'code_sha': require_env('VECGAD_CODE_SHA'),
            'protocol_sha': require_env('VECGAD_PROTOCOL_SHA'),
        }
        atomic_write(output_root / 'wandb' / f'{trial_id}.json', sidecar)

        run.summary['terminal_status'] = terminal_status
        run.summary['attempt_count'] = len(attempts)
        if terminal_result is not None:
            metrics = completed_metrics(terminal_result)
            run.log(metrics)
            for key, value in metrics.items():
                run.summary[key] = value
        if terminal_status == 'error':
            raise RuntimeError(f'trial failed without accepted terminal status: {attempts!r}')
    except Exception:
        exit_code = 1
        raise
    finally:
        run.finish(exit_code=exit_code)


if __name__ == '__main__':
    main()
