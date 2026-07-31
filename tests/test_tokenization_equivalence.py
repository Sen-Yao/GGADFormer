from types import SimpleNamespace

import torch

from utils import nagphormer_tokenization, node_neighborhood_feature


def _args(k=4, alpha=0.3):
    return SimpleNamespace(pp_k=k, progregate_alpha=alpha)


def _legacy_tokens(features, adj, args):
    tokens = [features]
    for hop in range(args.pp_k):
        tokens.append(node_neighborhood_feature(adj, features, hop + 1, args.progregate_alpha))
    return torch.stack(tokens, dim=1)


def test_sequential_dense_matches_legacy_recomputation():
    dense_adj = torch.tensor(
        [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    args = _args()

    expected = _legacy_tokens(features, dense_adj, args)
    actual = nagphormer_tokenization(features, dense_adj, args)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_sequential_sparse_matches_legacy_sparse_recomputation():
    dense_adj = torch.tensor(
        [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    sparse_adj = dense_adj.to_sparse().coalesce()
    features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    args = _args()

    expected = _legacy_tokens(features, sparse_adj, args)
    actual = nagphormer_tokenization(features, sparse_adj, args)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_isolated_nodes_and_self_loops_remain_finite():
    adj = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.2, 0.8]],
        dtype=torch.float32,
    ).to_sparse().coalesce()
    features = torch.ones((3, 2), dtype=torch.float32)

    tokens = nagphormer_tokenization(features, adj, _args(k=3, alpha=0.1))

    assert tokens.shape == (3, 4, 2)
    assert torch.isfinite(tokens).all()
