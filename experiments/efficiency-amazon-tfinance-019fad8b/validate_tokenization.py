#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import scipy.sparse as sp
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils import (
    load_mat,
    nagphormer_tokenization,
    node_neighborhood_feature,
    normalize_adj,
    preprocess_features,
    scipy_sparse_to_torch_sparse,
)


CONFIG = {
    'Amazon': {'pp_k': 5, 'alpha': 0.4},
    't_finance': {'pp_k': 7, 'alpha': 0.3},
}


def legacy_tokens(features, adj, pp_k, alpha):
    tokens = [features]
    for hop in range(pp_k):
        tokens.append(node_neighborhood_feature(adj, features, hop + 1, alpha))
    return torch.stack(tokens, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(CONFIG), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    config = CONFIG[args.dataset]
    load_args = SimpleNamespace(data_split_seed=42, sample_rate=0.15)
    adj, features, *_ = load_mat(args.dataset, 0.05, 0.1, args=load_args)
    if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()
    features = torch.as_tensor(features, dtype=torch.float32)

    operator = normalize_adj(adj) + sp.eye(adj.shape[0], format='coo')
    sparse_operator = scipy_sparse_to_torch_sparse(operator)
    reference_operator = (
        torch.as_tensor(operator.todense(), dtype=torch.float32)
        if args.dataset == 'Amazon'
        else sparse_operator
    )
    token_args = SimpleNamespace(
        pp_k=config['pp_k'],
        progregate_alpha=config['alpha'],
    )
    expected = legacy_tokens(
        features,
        reference_operator,
        config['pp_k'],
        config['alpha'],
    )
    actual = nagphormer_tokenization(features, sparse_operator, token_args)

    per_hop = []
    all_close = True
    for hop in range(config['pp_k'] + 1):
        difference = (actual[:, hop] - expected[:, hop]).abs()
        relative = difference / expected[:, hop].abs().clamp_min(1e-12)
        close = torch.allclose(actual[:, hop], expected[:, hop], rtol=1e-5, atol=1e-6)
        all_close = all_close and close
        per_hop.append({
            'hop': hop,
            'allclose': close,
            'max_absolute_error': difference.max().item(),
            'max_relative_error': relative.max().item(),
        })

    result = {
        'schema_version': 1,
        'dataset': args.dataset,
        'reference': 'legacy_dense' if args.dataset == 'Amazon' else 'legacy_sparse_recomputation',
        'rtol': 1e-5,
        'atol': 1e-6,
        'shape_equal': list(actual.shape) == list(expected.shape),
        'allclose': all_close,
        'per_hop': per_hop,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if not result['shape_equal'] or not result['allclose']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
