#!/usr/bin/env python3
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

EXPERIMENT_DIR = Path(__file__).resolve().parent


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


def load_trials(registry_path):
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    trials = []
    for source_key, source in registry['source_commits'].items():
        for batch_mode in registry['batch_modes']:
            for dataset, dataset_entry in registry['datasets'].items():
                batch_size = (
                    dataset_entry['native_batch_size']
                    if batch_mode == 'native'
                    else dataset_entry['num_nodes']
                )
                for repeat in registry['repeats']:
                    trials.append({
                        'id': f'{source_key}-{batch_mode}-{dataset.lower()}-r{repeat}',
                        'source_key': source_key,
                        'source_commit': source['commit'],
                        'source_root_env': source['root_env'],
                        'loader': source['loader'],
                        'batch_mode': batch_mode,
                        'batch_size': batch_size,
                        'dataset': dataset,
                        'num_nodes': dataset_entry['num_nodes'],
                        'dataset_config': dataset_entry['config'],
                        'repeat': repeat,
                        'seed': registry['seed'],
                        'warmup_epochs': registry['warmup_epochs'],
                        'measured_epochs': registry['measured_epochs'],
                    })
    return trials


def child_command(trial, output_path):
    config = trial['dataset_config']
    command = [
        sys.executable,
        str(EXPERIMENT_DIR / 'historical_vecgad_efficiency.py'),
        f"--trial-id={trial['id']}",
        f"--source-root={require_env(trial['source_root_env'])}",
        f"--source-commit={trial['source_commit']}",
        f"--dataset={trial['dataset']}",
        f"--batch-mode={trial['batch_mode']}",
        f"--batch-size={trial['batch_size']}",
        f"--num-nodes={trial['num_nodes']}",
        f"--num-workers={trial['loader']['num_workers']}",
        f"--repeat={trial['repeat']}",
        f"--seed={trial['seed']}",
        f"--warmup-epochs={trial['warmup_epochs']}",
        f"--measured-epochs={trial['measured_epochs']}",
        f"--data-split-seed={config['data_split_seed']}",
        f"--train-rate={config['train_rate']}",
        f"--num-epoch={config['num_epoch']}",
        f"--pp-k={config['pp_k']}",
        f"--progregate-alpha={config['progregate_alpha']}",
        f"--peak-lr={config['peak_lr']}",
        f"--end-lr={config['end_lr']}",
        f"--warmup-updates={config['warmup_updates']}",
        f"--outlier-beta={config.get('outlier_beta', 0.3)}",
        f"--lambda-rec-emb={config['lambda_rec_emb']}",
        f"--rec-loss-weight={config['rec_loss_weight']}",
        f"--ring-R-min={config['ring_R_min']}",
        f"--ring-R-max={config['ring_R_max']}",
        f"--ring-loss-weight={config['ring_loss_weight']}",
        f'--output={output_path}',
    ]
    if trial['loader']['persistent_workers']:
        command.append('--persistent-workers')
    if trial['loader']['pin_memory']:
        command.append('--pin-memory')
    return command


def main():
    import wandb

    registry_path = Path(
        os.environ.get('VECGAD_FORENSIC_REGISTRY', EXPERIMENT_DIR / 'registry.json')
    ).resolve()
    output_root = Path(require_env('VECGAD_FORENSIC_OUTPUT_ROOT')).resolve()
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
        job_type='historical-efficiency-forensic-trial',
        tags=['efficiency', 'historical-forensic', 'vecgad-only', 'hccs90'],
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
            'execution_host': 'HCCS-90',
            'physical_gpu_index': physical_gpu_index,
            'harness_sha': require_env('VECGAD_FORENSIC_HARNESS_SHA'),
            'protocol_sha': require_env('VECGAD_FORENSIC_PROTOCOL_SHA'),
        }, allow_val_change=True)

        result_path = output_root / 'raw' / f'{trial_id}.json'
        log_path = output_root / 'logs' / f'{trial_id}.log'
        result_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment['WANDB_MODE'] = 'disabled'
        environment['PYTHONUNBUFFERED'] = '1'
        with log_path.open('w', encoding='utf-8') as log_stream:
            process = subprocess.run(
                child_command(trial, result_path),
                cwd=EXPERIMENT_DIR,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result = None
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding='utf-8'))
        terminal_status = result.get('status') if result else 'missing_result'
        if process.returncode != 0 or terminal_status not in ('completed', 'gpu_oom'):
            terminal_status = 'error'

        sidecar = {
            'schema_version': 1,
            'trial_id': trial_id,
            'source_key': trial['source_key'],
            'source_commit': trial['source_commit'],
            'batch_mode': trial['batch_mode'],
            'dataset': trial['dataset'],
            'repeat': trial['repeat'],
            'terminal_status': terminal_status,
            'child_returncode': process.returncode,
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
            'harness_sha': require_env('VECGAD_FORENSIC_HARNESS_SHA'),
            'protocol_sha': require_env('VECGAD_FORENSIC_PROTOCOL_SHA'),
        }
        atomic_write(output_root / 'wandb' / f'{trial_id}.json', sidecar)
        run.summary['terminal_status'] = terminal_status
        run.summary['child_returncode'] = process.returncode
        if terminal_status == 'completed':
            metrics = {
                'offline/tokenization_seconds': result['offline']['tokenization_seconds'],
                'offline/token_payload_mib': result['offline']['token_payload_bytes'] / 1024 ** 2,
                'training/tqdm_terminal_rate_it_per_second': (
                    result['training']['tqdm_terminal_rate_it_per_second']
                ),
                'training/synchronized_block_seconds': (
                    result['training']['synchronized_block_seconds']
                ),
                'training/synchronized_epoch_seconds': (
                    result['training']['synchronized_epoch_seconds']
                ),
                'training/synchronized_throughput_it_per_second': (
                    result['training']['synchronized_throughput_it_per_second']
                ),
                'training/optimizer_steps_per_epoch': (
                    result['training']['optimizer_steps_per_epoch']
                ),
                'training/gpu_allocated_peak_gib': (
                    result['training']['gpu_allocated_peak_bytes'] / 1024 ** 3
                ),
                'training/gpu_reserved_peak_gib': (
                    result['training']['gpu_reserved_peak_bytes'] / 1024 ** 3
                ),
            }
            run.log(metrics)
            for key, value in metrics.items():
                run.summary[key] = value
        if terminal_status == 'error':
            raise RuntimeError(
                f'child failed: returncode={process.returncode}, status={terminal_status}'
            )
    except Exception:
        exit_code = 1
        raise
    finally:
        run.finish(exit_code=exit_code)


if __name__ == '__main__':
    main()
