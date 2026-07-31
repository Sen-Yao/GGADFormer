#!/usr/bin/env python3
import argparse
import json
import os
import platform
import random
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def rss_bytes():
    try:
        for line in Path('/proc/self/status').read_text(encoding='utf-8').splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class PeakRSS:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.baseline = rss_bytes()
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        def sample():
            while not self.stop_event.is_set():
                self.peak = max(self.peak, rss_bytes())
                self.stop_event.wait(self.interval)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, rss_bytes())
        return {
            'baseline_bytes': self.baseline,
            'peak_bytes': self.peak,
            'delta_bytes': max(0, self.peak - self.baseline),
        }


def cuda_current(device):
    return {
        'allocated_bytes': torch.cuda.memory_allocated(device),
        'reserved_bytes': torch.cuda.memory_reserved(device),
    }


def cuda_peak(device, baseline):
    peak = {
        'allocated_bytes': torch.cuda.max_memory_allocated(device),
        'reserved_bytes': torch.cuda.max_memory_reserved(device),
    }
    return {
        'baseline': baseline,
        'peak': peak,
        'delta': {key: max(0, peak[key] - baseline[key]) for key in peak},
    }


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def get_split(num_nodes, labels, train_ratio, val_ratio=0.1):
    labels = np.asarray(labels).squeeze()
    indices = list(range(num_nodes))
    random.shuffle(indices)
    train_end = int(num_nodes * train_ratio)
    val_end = train_end + int(num_nodes * val_ratio)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    normal_train = [index for index in train_indices if labels[index] == 0]
    normal_train = normal_train[:int(len(normal_train) * 0.5)]
    return normal_train, val_indices, test_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho-root', type=Path, required=True)
    parser.add_argument('--rho-data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['amazon', 'tfinance'], required=True)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--measured-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--train-ratio', type=float, required=True)
    parser.add_argument('--hidden1', type=int, required=True)
    parser.add_argument('--hidden2', type=int, required=True)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--tau', type=float, default=0.2)
    return parser.parse_args()


def run(args):
    rho_root = args.rho_root.resolve()
    rho_data_root = args.rho_data_root.resolve()
    datasets_dir = rho_data_root / 'datasets'
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f'missing RHO datasets directory: {datasets_dir}')
    os.chdir(rho_data_root)
    sys.path.insert(0, str(rho_root))
    from dataset import Dataset
    from model import RHO

    fix_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the formal RHO benchmark')
    device = torch.device(f'cuda:{args.cuda}')

    # Dataset file reads are deliberately outside the offline phase.
    dataset = Dataset(args.dataset)

    offline_rss = PeakRSS()
    offline_rss.start()
    torch.cuda.synchronize(device)
    offline_gpu_baseline = cuda_current(device)
    torch.cuda.reset_peak_memory_stats(device)
    offline_start = time.perf_counter()

    graph = dataset.graph
    laplacian = dataset.Lap
    labels = graph.ndata['label']
    features = graph.ndata['feature']
    idx_train, _, _ = get_split(features.shape[0], labels, args.train_ratio)
    model = RHO(
        features.shape[1], args.hidden1, args.hidden2, args.nlayers, args.batch_size, args.tau
    ).to(device)
    laplacian = laplacian.coalesce().to(device)
    features = features.to(device)
    labels = labels.to(device)
    model.apply(init_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    torch.cuda.synchronize(device)
    offline_seconds = time.perf_counter() - offline_start
    offline_memory = offline_rss.stop()
    offline_gpu = cuda_peak(device, offline_gpu_baseline)

    training_rss = PeakRSS()
    training_rss.start()
    torch.cuda.synchronize(device)
    training_gpu_baseline = cuda_current(device)
    torch.cuda.reset_peak_memory_stats(device)
    measured = []

    for epoch in range(args.warmup_epochs + args.measured_epochs):
        torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        model.eval()
        with torch.no_grad():
            outputs_global, outputs_local, _ = model(laplacian, features)
            center_global = outputs_global.sum(dim=0) / outputs_global.shape[0]
            center_local = outputs_local.sum(dim=0) / outputs_local.shape[0]
            eps = 0.1
            center_global[(center_global.abs() < eps) & (center_global < 0)] = -eps
            center_global[(center_global.abs() < eps) & (center_global > 0)] = eps
            center_local[(center_local.abs() < eps) & (center_local < 0)] = -eps
            center_local[(center_local.abs() < eps) & (center_local > 0)] = eps

        model.train()
        optimizer.zero_grad()
        outputs_global, outputs_local, nce_loss = model(laplacian, features)
        dist_global = ((outputs_global[idx_train] - center_global) ** 2).sum(dim=1)
        dist_local = ((outputs_local[idx_train] - center_local) ** 2).sum(dim=1)
        loss = (0.5 * dist_global + 0.5 * dist_local).mean() + args.alpha * nce_loss
        loss.backward()
        optimizer.step()
        del outputs_global, outputs_local
        torch.cuda.empty_cache()

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        if epoch >= args.warmup_epochs:
            measured.append(elapsed)

    training_memory = training_rss.stop()
    properties = torch.cuda.get_device_properties(device)
    laplacian_payload = (
        laplacian.indices().untyped_storage().nbytes()
        + laplacian.values().untyped_storage().nbytes()
    )
    config = vars(args).copy()
    config['rho_root'] = str(rho_root)
    config['rho_data_root'] = str(rho_data_root)
    config['output'] = str(args.output)
    return {
        'schema_version': 1,
        'status': 'completed',
        'dataset': args.dataset,
        'method': 'RHO',
        'seed': args.seed,
        'repeat': args.repeat,
        'config': config,
        'runtime': {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'gpu': {
                'index': args.cuda,
                'name': properties.name,
                'total_memory_bytes': properties.total_memory,
            },
        },
        'offline': {
            'seconds': offline_seconds,
            'tokenization_seconds': 0.0,
            'rss': offline_memory,
            'gpu_peak': offline_gpu,
            'token_payload_bytes': 0,
            'feature_payload_bytes': features.untyped_storage().nbytes(),
            'laplacian_payload_bytes': laplacian_payload,
        },
        'training': {
            'warmup_epochs': args.warmup_epochs,
            'measured_epochs': args.measured_epochs,
            'epoch_seconds': measured,
            'rss': training_memory,
            'gpu_peak': cuda_peak(device, training_gpu_baseline),
            'optimizer_steps_per_epoch': 1,
            'batch_size': args.batch_size,
            'center_recomputation_included': True,
        },
        'model': {
            'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
            'trainable_parameter_count': sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
    }


def main():
    args = parse_args()
    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError as error:
        result = {
            'schema_version': 1,
            'status': 'gpu_oom',
            'dataset': args.dataset,
            'method': 'RHO',
            'seed': args.seed,
            'repeat': args.repeat,
            'error_type': type(error).__name__,
            'error': str(error),
        }
        atomic_write(args.output, result)
        raise
    except Exception as error:
        result = {
            'schema_version': 1,
            'status': 'error',
            'dataset': args.dataset,
            'method': 'RHO',
            'seed': args.seed,
            'repeat': args.repeat,
            'error_type': type(error).__name__,
            'error': str(error),
        }
        atomic_write(args.output, result)
        raise
    atomic_write(args.output, result)


if __name__ == '__main__':
    main()
