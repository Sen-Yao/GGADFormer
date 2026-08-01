"""Dataset loading and preprocessing for the anonymous VecGAD supplement."""

import os
import pickle
import random

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch


STANDARD_DATASETS = ("Amazon", "reddit", "photo", "elliptic", "t_finance", "tolokers")
DATASETS = STANDARD_DATASETS + ("dgraph",)
ROW_NORMALIZED_FEATURES = {"Amazon", "reddit", "elliptic"}


def _split_indices(labels, train_rate, val_rate, split_seed):
    """Match the source split while keeping split RNG isolated from training RNG."""
    n_nodes = len(labels)
    all_idx = list(range(n_nodes))
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(split_seed)
        np.random.seed(split_seed)
        random.shuffle(all_idx)
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)

    n_train = int(n_nodes * train_rate)
    n_val = int(n_nodes * val_rate)
    idx_train = all_idx[:n_train]
    idx_val = all_idx[n_train : n_train + n_val]
    idx_test = all_idx[n_train + n_val :]
    normal_for_train = [i for i in idx_train if labels[i] == 0]
    return idx_train, idx_val, idx_test, normal_for_train


def _row_normalize(features):
    if not sp.issparse(features):
        features = sp.csr_matrix(features)
    rowsum = np.asarray(features.sum(axis=1))
    inv = np.power(rowsum, -1).reshape(-1)
    inv[np.isinf(inv)] = 0.0
    return sp.diags(inv).dot(features)


def _standard_adjacency(network):
    """Build D^-1/2 A D^-1/2 + I, as used by the formal source path."""
    adj = sp.coo_matrix(network)
    rowsum = np.asarray(adj.sum(axis=1))
    inv = np.power(rowsum, -0.5).reshape(-1)
    inv[np.isinf(inv)] = 0.0
    diagonal = sp.diags(inv)
    normalized = diagonal.dot(adj).dot(diagonal)
    return (normalized + sp.eye(adj.shape[0])).tocsr()


def _dgraph_adjacency(adjacency_list, n_nodes):
    rows = []
    cols = []
    for source in range(n_nodes):
        for target in adjacency_list[source]:
            if 0 <= target < n_nodes:
                rows.append(source)
                cols.append(target)
    values = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes))
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj = adj + sp.eye(n_nodes, dtype=np.float32, format="csr")
    degrees = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(degrees, dtype=np.float32)
    nonzero = degrees != 0
    inv[nonzero] = np.power(degrees[nonzero], -0.5)
    return sp.diags(inv).dot(adj).dot(sp.diags(inv)).tocsr()


def scipy_to_torch_sparse(matrix):
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def _mat_field(data, primary, fallback):
    return data[primary] if primary in data else data[fallback]


def load_dataset(dataset, data_dir="dataset", train_rate=0.05, val_rate=0.1,
                 data_split_seed=42):
    """Load one prepared benchmark dataset without downloading or writing files."""
    if dataset not in DATASETS:
        raise ValueError("unsupported dataset: {}".format(dataset))

    if dataset == "dgraph":
        feature_path = os.path.join(data_dir, "dgraphfin.npz")
        adjacency_path = os.path.join(data_dir, "dgraphfin_adj_list")
        loaded = np.load(feature_path)
        labels = (np.asarray(loaded["y"]).reshape(-1) == 1).astype(np.float32)
        features = torch.as_tensor(np.asarray(loaded["x"], dtype=np.float32))
        with open(adjacency_path, "rb") as handle:
            adjacency_list = pickle.load(handle)
        adjacency = _dgraph_adjacency(adjacency_list, features.shape[0])
        torch_adjacency = scipy_to_torch_sparse(adjacency)
    else:
        mat = sio.loadmat(os.path.join(data_dir, "{}.mat".format(dataset)))
        labels = np.asarray(_mat_field(mat, "Label", "gnd")).reshape(-1).astype(np.float32)
        raw_features = _mat_field(mat, "Attributes", "X")
        raw_adjacency = _mat_field(mat, "Network", "A")
        adjacency = _standard_adjacency(raw_adjacency)
        if dataset in ROW_NORMALIZED_FEATURES:
            raw_features = _row_normalize(raw_features)
        if sp.issparse(raw_features):
            raw_features = raw_features.toarray()
        features = torch.as_tensor(np.asarray(raw_features, dtype=np.float32))
        torch_adjacency = scipy_to_torch_sparse(adjacency)

    idx_train, idx_val, idx_test, normal_for_train = _split_indices(
        labels, train_rate, val_rate, data_split_seed
    )
    return {
        "adjacency": torch_adjacency,
        "features": features,
        "labels": torch.as_tensor(labels, dtype=torch.float32),
        "idx_test": idx_test,
        "normal_for_train": normal_for_train,
    }
