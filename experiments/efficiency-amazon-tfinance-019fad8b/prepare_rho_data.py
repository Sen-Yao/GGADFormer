#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import dgl
from dgl.data import FraudAmazonDataset
from dgl.data.utils import save_graphs
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_laplacian(graph):
    graph = dgl.remove_self_loop(graph)
    source, destination = graph.edges()
    node_count = graph.num_nodes()
    values = np.ones(source.numel(), dtype=np.float32)
    adjacency = sp.coo_matrix(
        (values, (source.cpu().numpy(), destination.cpu().numpy())),
        shape=(node_count, node_count),
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.transpose())
    adjacency = adjacency + sp.eye(node_count, dtype=np.float32, format='csr')
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse_sqrt = np.zeros_like(degrees, dtype=np.float32)
    nonzero = degrees > 0
    inverse_sqrt[nonzero] = np.power(degrees[nonzero], -0.5)
    scaling = sp.diags(inverse_sqrt, format='csr')
    return (sp.eye(node_count, dtype=np.float32) - scaling @ adjacency @ scaling).tocoo()


def tfinance_graph(mat_path):
    data = sio.loadmat(mat_path)
    adjacency = sp.coo_matrix(data['Network'])
    graph = dgl.from_scipy(adjacency)
    graph.ndata['feature'] = torch.as_tensor(data['Attributes'], dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(data['Label']).reshape(-1), dtype=torch.long)
    graph.ndata['label'] = torch.nn.functional.one_hot(labels, num_classes=2)
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfinance-mat', type=Path, required=True)
    parser.add_argument('--rho-datasets', type=Path, required=True)
    parser.add_argument('--dgl-raw-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()

    args.rho_datasets.mkdir(parents=True, exist_ok=True)
    args.dgl_raw_dir.mkdir(parents=True, exist_ok=True)

    tfinance = tfinance_graph(args.tfinance_mat)
    tfinance_path = args.rho_datasets / 'tfinance'
    save_graphs(str(tfinance_path), [tfinance])
    tfinance_laplacian_path = args.rho_datasets / 'Lap_matrix_tfinance.npz'
    sp.save_npz(tfinance_laplacian_path, normalized_laplacian(tfinance))

    amazon_dataset = FraudAmazonDataset(raw_dir=str(args.dgl_raw_dir))
    amazon = dgl.to_homogeneous(
        amazon_dataset[0],
        ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'],
    )
    amazon_laplacian_path = args.rho_datasets / 'Lap_matrix_amazon.npz'
    sp.save_npz(amazon_laplacian_path, normalized_laplacian(amazon))

    manifest = {
        'schema_version': 1,
        'transform': 'rho_get_Lap_sparse_equivalent_v1',
        'tfinance_source': {
            'path': str(args.tfinance_mat.resolve()),
            'sha256': sha256(args.tfinance_mat),
            'nodes': tfinance.num_nodes(),
            'edges': tfinance.num_edges(),
        },
        'amazon_source': {
            'dataset': 'dgl.data.FraudAmazonDataset',
            'nodes': amazon.num_nodes(),
            'edges': amazon.num_edges(),
        },
        'outputs': {
            path.name: {'bytes': path.stat().st_size, 'sha256': sha256(path)}
            for path in (tfinance_path, tfinance_laplacian_path, amazon_laplacian_path)
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
