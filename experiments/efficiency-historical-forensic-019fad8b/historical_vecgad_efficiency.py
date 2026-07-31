#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm


SOURCE_CONTRACTS = {
    'e071ae6646451d94fc8e8c9e88305eb76c393089': {
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': True,
        'files': {
            'run.py': '8aedf53ee85c0569212307eb346cb01d6f6b1664c75977a65704f865df470933',
            'utils.py': 'd4f094caac4d99ae745e97e40cf15e2c39f2edc19ea7b5159312ade56767a028',
            'GGADFormer.py': 'c279c43159ad46f135345252472c08e2b1078b66a0d9488e4540e2a7ef4829ce',
        },
    },
    '5bf8205b0d4c54d583b13c547ae62122ffdf2f6a': {
        'num_workers': 0,
        'persistent_workers': False,
        'pin_memory': False,
        'files': {
            'run.py': '17e87737a3dc4e8f209ff8347d68cf168cd2be038c31eb391a8e08d201e3fd1f',
            'utils.py': '2b29692117545eebec068e6543617e2d3c98bf2f626e7c533f1836ca682a7036',
            'GGADFormer.py': 'c279c43159ad46f135345252472c08e2b1078b66a0d9488e4540e2a7ef4829ce',
        },
    },
}


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root, *args):
    return subprocess.check_output(
        ['git', '-C', str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load module from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_identity(device):
    properties = torch.cuda.get_device_properties(device)
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'dgl': dgl.__version__,
        'gpu': {
            'index': device.index,
            'name': properties.name,
            'total_memory_bytes': properties.total_memory,
        },
    }


def build_config(parsed):
    defaults = {
        'weight_decay': 0.0,
        'seed': parsed.seed,
        'data_split_seed': parsed.data_split_seed,
        'train_rate': parsed.train_rate,
        'batch_size': parsed.batch_size,
        'embedding_dim': 256,
        'proj_dim': 64,
        'num_epoch': parsed.num_epoch,
        'drop_prob': 0.0,
        'readout': 'avg',
        'auc_test_rounds': 256,
        'negsamp_ratio': 1,
        'mean': 0.0,
        'var': 0.0,
        'confidence_margin': 2.0,
        'outlier_beta': parsed.outlier_beta,
        'sample_rate': 0.15,
        'model_type': 'GGADFormer',
        'visualize': False,
        'device': 0,
        'pp_k': parsed.pp_k,
        'progregate_alpha': parsed.progregate_alpha,
        'sample_num_p': 7,
        'sample_num_n': 7,
        'sample_size': 10000,
        'GT_ffn_dim': 256,
        'GT_dropout': 0.4,
        'GT_attention_dropout': 0.4,
        'GT_num_heads': 2,
        'GT_num_layers': 3,
        'proj_R_max': 0.5,
        'proj_R_min': 0.1,
        'ring_R_max': parsed.ring_R_max,
        'ring_R_min': parsed.ring_R_min,
        'rec_loss_weight': parsed.rec_loss_weight,
        'bce_loss_weight': 1.0,
        'margin_loss_weight': 0.0,
        'con_loss_weight': 0.1,
        'proj_loss_weight': 0.0,
        'reconstruction_loss_weight': 1.0,
        'ring_loss_weight': parsed.ring_loss_weight,
        'lambda_rec_tok': 1.0,
        'lambda_rec_emb': parsed.lambda_rec_emb,
        'con_loss_temp': 10.0,
        'GNA_temp': 1.0,
        'warmup_updates': parsed.warmup_updates,
        'tot_updates': 1000,
        'peak_lr': parsed.peak_lr,
        'end_lr': parsed.end_lr,
        'warmup_epoch': 20,
        'dataset': parsed.dataset,
    }
    return argparse.Namespace(**defaults)


def prepare(parsed, historical_utils, model_module, device):
    args = build_config(parsed)
    loaded = historical_utils.load_mat(
        args.dataset, args.train_rate, 0.1, args=args
    )
    adj, features, labels = loaded[:3]
    normal_for_train_idx = loaded[10]

    if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features, _ = historical_utils.preprocess_features(features)
    else:
        features = features.todense()

    num_nodes = features.shape[0]
    if num_nodes != parsed.num_nodes:
        raise RuntimeError(f'node-count mismatch: expected {parsed.num_nodes}, got {num_nodes}')
    if parsed.batch_mode == 'fullbatch' and parsed.batch_size != num_nodes:
        raise RuntimeError('fullbatch trial batch size must equal the observed node count')
    if parsed.batch_mode == 'native' and parsed.batch_size >= num_nodes:
        raise RuntimeError('native trial unexpectedly resolves to full batch')

    adj = historical_utils.normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features[np.newaxis])
    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    adj = torch.FloatTensor(adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis]).squeeze(0)

    tokenization_start = time.perf_counter()
    tokens = historical_utils.nagphormer_tokenization(
        features.squeeze(0), adj.squeeze(0), args
    )
    tokenization_seconds = time.perf_counter() - tokenization_start

    model = model_module.GGADFormer(features.shape[-1], args.embedding_dim, 'prelu', args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay
    )
    scheduler = historical_utils.PolynomialDecayLR(
        optimizer,
        warmup_updates=(
            int(0.1 * args.num_epoch)
            if args.warmup_updates == -1
            else args.warmup_updates
        ),
        tot_updates=args.num_epoch,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )
    b_xent = nn.BCEWithLogitsLoss(
        reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device)
    )

    all_node_indices = torch.arange(num_nodes)
    dataset = Data.TensorDataset(tokens, labels, all_node_indices)
    all_indices = set(range(num_nodes))
    known_indices = set(normal_for_train_idx)
    unknown_indices = list(all_indices - known_indices)
    weights = torch.zeros(num_nodes)
    weights[normal_for_train_idx] = 1.0 / len(normal_for_train_idx)
    weights[unknown_indices] = 1.0 / len(unknown_indices)
    sampler = Data.WeightedRandomSampler(weights, num_samples=num_nodes, replacement=True)
    loader = Data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=parsed.num_workers,
        persistent_workers=parsed.persistent_workers,
        pin_memory=parsed.pin_memory,
    )
    normal_for_train_idx = torch.tensor(
        normal_for_train_idx, dtype=torch.long, device=device
    )
    return {
        'args': args,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'b_xent': b_xent,
        'loader': loader,
        'normal_for_train_idx': normal_for_train_idx,
        'tokenization_seconds': tokenization_seconds,
        'token_payload_bytes': tokens.untyped_storage().nbytes(),
    }


def train_epoch(epoch, state, historical_utils, device):
    args = state['args']
    model = state['model']
    optimizer = state['optimizer']
    b_xent = state['b_xent']
    normal_for_train_idx = state['normal_for_train_idx']
    dynamic_weights = historical_utils.get_dynamic_loss_weights(epoch, args)
    model.train()
    batched_bce_loss = torch.tensor(0.0, device=device)
    batched_rec_loss = torch.tensor(0.0, device=device)
    batched_ring_loss = torch.tensor(0.0, device=device)
    for item in state['loader']:
        input_tokens = item[0].to(device)
        batch_global_indices = item[2].to(device)
        optimizer.zero_grad()
        is_known_normal_mask = torch.isin(batch_global_indices, normal_for_train_idx)
        local_normal_for_train_idx = torch.nonzero(
            is_known_normal_mask, as_tuple=False
        ).squeeze(-1)
        _, _, logits, outlier_emb, _, loss_rec, loss_ring = model(
            input_tokens, None, None, local_normal_for_train_idx, True, args
        )
        labels = torch.unsqueeze(
            torch.cat(
                (
                    torch.zeros(len(local_normal_for_train_idx)),
                    torch.ones(len(outlier_emb)),
                )
            ),
            1,
        ).unsqueeze(0).to(device)
        loss_bce = torch.mean(b_xent(logits, labels))
        loss = (
            dynamic_weights['bce_loss_weight'] * loss_bce
            + dynamic_weights['rec_loss_weight'] * loss_rec
            + dynamic_weights['ring_loss_weight'] * loss_ring
        )
        loss.backward()
        optimizer.step()
        batched_bce_loss += loss_bce
        batched_rec_loss += loss_rec
        batched_ring_loss += loss_ring

    scalar_sync = None
    if epoch % 2 == 0:
        scalar_sync = {
            'bce_loss': batched_bce_loss.item(),
            'rec_loss': batched_rec_loss.item(),
            'ring_loss': batched_ring_loss.item(),
        }
    state['scheduler'].step()
    return scalar_sync


def run(parsed):
    output = parsed.output.resolve()
    source_root = parsed.source_root.resolve()
    contract = SOURCE_CONTRACTS.get(parsed.source_commit)
    if contract is None:
        raise RuntimeError(f'undeclared source commit: {parsed.source_commit}')
    observed_contract = {
        'num_workers': parsed.num_workers,
        'persistent_workers': parsed.persistent_workers,
        'pin_memory': parsed.pin_memory,
    }
    expected_loader_contract = {
        key: contract[key]
        for key in ('num_workers', 'persistent_workers', 'pin_memory')
    }
    if observed_contract != expected_loader_contract:
        raise RuntimeError(
            f'loader contract mismatch for {parsed.source_commit}: '
            f'expected {expected_loader_contract!r}, got {observed_contract!r}'
        )
    observed_commit = git_output(source_root, 'rev-parse', 'HEAD')
    if observed_commit != parsed.source_commit:
        raise RuntimeError(
            f'source worktree mismatch: expected {parsed.source_commit}, got {observed_commit}'
        )
    source_status = git_output(source_root, 'status', '--porcelain', '--untracked-files=no')
    if source_status:
        raise RuntimeError(f'historical source worktree has tracked changes: {source_status!r}')
    source_file_hashes = {
        name: sha256(source_root / name) for name in ('run.py', 'utils.py', 'GGADFormer.py')
    }
    if source_file_hashes != contract['files']:
        raise RuntimeError(
            f'historical source file digest mismatch: expected {contract["files"]!r}, '
            f'got {source_file_hashes!r}'
        )

    sys.path.insert(0, str(source_root))
    historical_utils = load_module(source_root / 'utils.py', f'historical_utils_{os.getpid()}')
    model_module = load_module(source_root / 'GGADFormer.py', f'historical_model_{os.getpid()}')
    os.chdir(str(source_root))

    dgl.random.seed(parsed.seed)
    np.random.seed(parsed.seed)
    torch.manual_seed(parsed.seed)
    torch.cuda.manual_seed(parsed.seed)
    torch.cuda.manual_seed_all(parsed.seed)
    random.seed(parsed.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda:0')

    state = prepare(parsed, historical_utils, model_module, device)
    for epoch in range(parsed.warmup_epochs):
        train_epoch(epoch, state, historical_utils, device)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    measured_async_epoch_seconds = []
    scalar_sync_epochs = []
    progress = tqdm(total=parsed.measured_epochs, desc='Training')
    measured_start = time.perf_counter()
    accumulated_async_seconds = 0.0
    for measured_epoch in range(parsed.measured_epochs):
        global_epoch = parsed.warmup_epochs + measured_epoch
        async_start = time.perf_counter()
        scalar_sync = train_epoch(global_epoch, state, historical_utils, device)
        async_seconds = time.perf_counter() - async_start
        accumulated_async_seconds += async_seconds
        measured_async_epoch_seconds.append(async_seconds)
        if scalar_sync is not None:
            scalar_sync_epochs.append(global_epoch)
        progress.set_postfix({
            'Time': f'{accumulated_async_seconds:.1f}s',
            'Epoch': f'{measured_epoch + 1}/{parsed.measured_epochs}',
            'AUC': '0.0000',
            'AP': '0.0000',
        })
        progress.update(1)

    tqdm_state = dict(progress.format_dict)
    tqdm_terminal_rate = tqdm_state.get('rate')
    torch.cuda.synchronize(device)
    synchronized_block_seconds = time.perf_counter() - measured_start
    gpu_allocated_peak = torch.cuda.max_memory_allocated(device)
    gpu_reserved_peak = torch.cuda.max_memory_reserved(device)
    progress.close()
    if not tqdm_terminal_rate or tqdm_terminal_rate <= 0:
        raise RuntimeError(f'invalid terminal tqdm rate: {tqdm_terminal_rate!r}')

    optimizer_steps = len(state['loader'])
    expected_optimizer_steps = math.ceil(parsed.num_nodes / parsed.batch_size)
    if optimizer_steps != expected_optimizer_steps:
        raise RuntimeError(
            f'optimizer-step mismatch: expected {expected_optimizer_steps}, got {optimizer_steps}'
        )
    result = {
        'schema_version': 1,
        'status': 'completed',
        'trial_id': parsed.trial_id,
        'dataset': parsed.dataset,
        'repeat': parsed.repeat,
        'seed': parsed.seed,
        'source': {
            'commit': parsed.source_commit,
            'root': str(source_root),
            'run_py_sha256': sha256(source_root / 'run.py'),
            'utils_py_sha256': sha256(source_root / 'utils.py'),
            'model_py_sha256': sha256(source_root / 'GGADFormer.py'),
            'tracked_status': source_status,
        },
        'runtime': runtime_identity(device),
        'config': {
            **vars(state['args']),
        'batch_mode': parsed.batch_mode,
            'num_nodes': parsed.num_nodes,
            'num_workers': parsed.num_workers,
            'persistent_workers': parsed.persistent_workers,
            'pin_memory': parsed.pin_memory,
        },
        'offline': {
            'tokenization_seconds': state['tokenization_seconds'],
            'token_payload_bytes': state['token_payload_bytes'],
        },
        'training': {
            'warmup_epochs': parsed.warmup_epochs,
            'measured_epochs': parsed.measured_epochs,
            'optimizer_steps_per_epoch': optimizer_steps,
            'measured_async_epoch_seconds': measured_async_epoch_seconds,
            'historical_scalar_sync_epochs': scalar_sync_epochs,
            'tqdm_terminal_rate_it_per_second': tqdm_terminal_rate,
            'tqdm_terminal_n': tqdm_state.get('n'),
            'tqdm_terminal_total': tqdm_state.get('total'),
            'tqdm_elapsed_seconds_before_final_cuda_sync': tqdm_state.get('elapsed'),
            'synchronized_block_seconds': synchronized_block_seconds,
            'synchronized_epoch_seconds': (
                synchronized_block_seconds / parsed.measured_epochs
            ),
            'synchronized_throughput_it_per_second': (
                parsed.measured_epochs / synchronized_block_seconds
            ),
            'gpu_allocated_peak_bytes': gpu_allocated_peak,
            'gpu_reserved_peak_bytes': gpu_reserved_peak,
        },
    }
    atomic_write(output, result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial-id', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--dataset', required=True, choices=['Amazon', 't_finance'])
    parser.add_argument('--batch-mode', required=True, choices=['native', 'fullbatch'])
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--num-nodes', type=int, required=True)
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--warmup-epochs', type=int, required=True)
    parser.add_argument('--measured-epochs', type=int, required=True)
    parser.add_argument('--data-split-seed', type=int, required=True)
    parser.add_argument('--train-rate', type=float, required=True)
    parser.add_argument('--num-epoch', type=int, required=True)
    parser.add_argument('--pp-k', type=int, required=True)
    parser.add_argument('--progregate-alpha', type=float, required=True)
    parser.add_argument('--peak-lr', type=float, required=True)
    parser.add_argument('--end-lr', type=float, required=True)
    parser.add_argument('--warmup-updates', type=int, required=True)
    parser.add_argument('--outlier-beta', type=float, default=0.3)
    parser.add_argument('--lambda-rec-emb', type=float, required=True)
    parser.add_argument('--rec-loss-weight', type=float, required=True)
    parser.add_argument('--ring-R-min', type=float, required=True)
    parser.add_argument('--ring-R-max', type=float, required=True)
    parser.add_argument('--ring-loss-weight', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main():
    parsed = parse_args()
    try:
        run(parsed)
    except torch.cuda.OutOfMemoryError as error:
        atomic_write(parsed.output.resolve(), {
            'schema_version': 1,
            'status': 'gpu_oom',
            'trial_id': parsed.trial_id,
            'dataset': parsed.dataset,
            'repeat': parsed.repeat,
            'source_commit': parsed.source_commit,
            'batch_mode': parsed.batch_mode,
            'error': str(error),
        })
    except Exception as error:
        atomic_write(parsed.output.resolve(), {
            'schema_version': 1,
            'status': 'error',
            'trial_id': parsed.trial_id,
            'dataset': parsed.dataset,
            'repeat': parsed.repeat,
            'source_commit': parsed.source_commit,
            'batch_mode': parsed.batch_mode,
            'error_type': type(error).__name__,
            'error': str(error),
        })
        raise


if __name__ == '__main__':
    main()
