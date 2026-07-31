import math

import torch


def hsc_batch_metrics(
    *,
    batch_index,
    emb,
    outlier_emb,
    loss_bce,
    loss_rec,
    loss_ring,
    loss_total,
    ring_r_min,
    ring_r_max,
    rec_loss_weight,
    ring_loss_weight,
    bce_loss_weight,
):
    """Return detached scalar diagnostics for one existing training batch."""
    with torch.no_grad():
        center = emb.detach().mean(dim=1, keepdim=True).squeeze(0)
        distances = torch.norm(outlier_emb.detach() - center, p=2, dim=1)
        inner = distances < ring_r_min
        outer = distances > ring_r_max
        shell = ~(inner | outer)

        prefix = f"diagnostic/batch_{batch_index}"
        metrics = {
            f"{prefix}/pseudo_count": int(distances.numel()),
            f"{prefix}/hsc/shell_hit_rate": shell.float().mean().item(),
            f"{prefix}/hsc/inner_violation_rate": inner.float().mean().item(),
            f"{prefix}/hsc/outer_violation_rate": outer.float().mean().item(),
            f"{prefix}/hsc/distance_mean": distances.mean().item(),
            f"{prefix}/loss/bce_raw": loss_bce.detach().item(),
            f"{prefix}/loss/rec_combined_raw": loss_rec.detach().item(),
            f"{prefix}/loss/hsc_raw": loss_ring.detach().item(),
            f"{prefix}/loss/bce_weighted": (
                bce_loss_weight * loss_bce.detach()
            ).item(),
            f"{prefix}/loss/rec_combined_weighted": (
                rec_loss_weight * loss_rec.detach()
            ).item(),
            f"{prefix}/loss/hsc_weighted": (
                ring_loss_weight * loss_ring.detach()
            ).item(),
            f"{prefix}/loss/true_weighted_total": loss_total.detach().item(),
        }

    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"non-finite HSC diagnostic {key}: {value}")
    return metrics
