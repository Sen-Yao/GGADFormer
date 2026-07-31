"""Controlled HSC centers and aggregate shell diagnostics."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


HSC_CENTER_Q = {
    "default": None,
    "q0": 0.0,
    "q10": 0.1,
    "q20": 0.2,
    "q30": 0.3,
    "q40": 0.4,
}


@dataclass(frozen=True)
class CenterComponents:
    default: torch.Tensor
    normal: torch.Tensor
    anomaly: torch.Tensor
    selected: torch.Tensor
    anomaly_fraction: torch.Tensor
    q: Optional[float]


def _validate_embeddings(emb):
    if emb.ndim != 3 or emb.size(0) != 1 or emb.size(1) == 0:
        raise ValueError("emb must have shape [1, num_nodes, embedding_dim]")


def _class_components(emb, batch_labels):
    if batch_labels is None:
        raise ValueError("oracle HSC centers require batch ground-truth labels")
    labels = batch_labels.reshape(-1)
    if labels.numel() != emb.size(1):
        raise ValueError("batch label count must match the embedding node dimension")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("HSC center labels must be binary 0/1")

    normal_mask = labels == 0
    anomaly_mask = labels == 1
    if not torch.any(normal_mask):
        raise ValueError("oracle HSC center is undefined without normal nodes")
    if not torch.any(anomaly_mask):
        raise ValueError("oracle HSC center is undefined without anomalous nodes")

    normal = emb[:, normal_mask, :].mean(dim=1, keepdim=True)
    anomaly = emb[:, anomaly_mask, :].mean(dim=1, keepdim=True)
    anomaly_fraction = anomaly_mask.to(dtype=emb.dtype).mean()
    return normal, anomaly, anomaly_fraction


def compute_center_components(emb, batch_labels, condition):
    """Return the default, class-conditional, and selected batch centers."""
    _validate_embeddings(emb)
    if condition not in HSC_CENTER_Q:
        raise ValueError(f"unsupported HSC center condition: {condition}")

    default = emb.mean(dim=1, keepdim=True)
    q = HSC_CENTER_Q[condition]
    normal, anomaly, anomaly_fraction = _class_components(emb, batch_labels)
    selected = default if q is None else (1.0 - q) * normal + q * anomaly
    return CenterComponents(
        default=default,
        normal=normal,
        anomaly=anomaly,
        selected=selected,
        anomaly_fraction=anomaly_fraction,
        q=q,
    )


def compute_hsc_center(emb, batch_labels, condition):
    """Construct the selected center without detaching its autograd graph."""
    if condition == "default":
        _validate_embeddings(emb)
        return emb.mean(dim=1, keepdim=True)
    return compute_center_components(emb, batch_labels, condition).selected


def compute_shell_statistics(outlier_emb, center, radius_min, radius_max):
    """Compute sample-weighted shell counts and hinge-loss sums."""
    if outlier_emb.ndim != 2 or outlier_emb.size(0) == 0:
        raise ValueError("outlier_emb must have shape [num_outliers, embedding_dim]")
    if center.shape != (1, 1, outlier_emb.size(1)):
        raise ValueError("center must have shape [1, 1, embedding_dim]")
    if radius_min > radius_max:
        raise ValueError("radius_min must not exceed radius_max")

    distances = torch.norm(outlier_emb - center.squeeze(0), p=2, dim=1)
    inner = distances < radius_min
    outer = distances > radius_max
    shell = ~(inner | outer)
    penalties = F.relu(radius_min - distances) + F.relu(distances - radius_max)
    return {
        "count": int(distances.numel()),
        "shell_count": int(shell.sum().item()),
        "inner_count": int(inner.sum().item()),
        "outer_count": int(outer.sum().item()),
        "hsc_loss_sum": float(penalties.sum().item()),
    }
