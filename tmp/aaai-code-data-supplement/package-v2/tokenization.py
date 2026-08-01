"""Incremental sparse graph tokenization."""

import torch


def propagate_once(adjacency, previous, original, alpha):
    """Apply one sparse propagation step with original-feature reinjection."""
    return (1.0 - alpha) * torch.sparse.mm(adjacency, previous) + alpha * original


def incremental_tokenization(features, adjacency, num_hops, alpha):
    """Return [node, hop, feature] tokens using exactly ``num_hops`` sparse steps."""
    if features.dim() != 2:
        raise ValueError("features must have shape [nodes, features]")
    if adjacency.layout != torch.sparse_coo:
        adjacency = adjacency.to_sparse_coo()
    original = features
    current = original
    tokens = [original]
    for _ in range(num_hops):
        current = propagate_once(adjacency, current, original, alpha)
        tokens.append(current)
    return torch.stack(tokens, dim=1)
