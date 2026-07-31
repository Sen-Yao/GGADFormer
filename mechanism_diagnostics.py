import hashlib
import json
import math
from pathlib import Path

import torch


GRADIENT_EPOCHS = frozenset((0, 1, 2, 5, 10, 20, 50, 100, 150))
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def update_index_trace(digest, trace_name, epoch, batch_index, indices):
    values = indices.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    digest.update(
        "{}:{}:{}:{}\n".format(trace_name, epoch, batch_index, values.size).encode("ascii")
    )
    digest.update(values.tobytes())


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite diagnostic value: {!r}".format(result))
    return result


def distribution_metrics(prefix, values):
    flat = values.detach().reshape(-1).float()
    if flat.numel() == 0:
        raise ValueError("{} has no observations".format(prefix))
    quantile_tensor = torch.tensor(QUANTILES, device=flat.device, dtype=flat.dtype)
    quantile_values = torch.quantile(flat, quantile_tensor)
    metrics = {
        prefix + "/count": int(flat.numel()),
        prefix + "/mean": _finite_float(flat.mean().item()),
        prefix + "/variance": _finite_float(flat.var(unbiased=False).item()),
    }
    for quantile, value in zip(QUANTILES, quantile_values):
        name = str(quantile).replace(".", "p")
        metrics[prefix + "/q" + name] = _finite_float(value.item())
    return metrics


def gradient_metrics(losses, parameters, weights):
    parameters = tuple(parameters)
    names = tuple(losses)
    gradients = {}
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count == 0:
        raise ValueError("no trainable parameters were provided")

    for name in names:
        gradients[name] = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

    metrics = {"gradient/parameter_count": int(parameter_count)}
    squared_norms = {}
    active_counts = {}
    for name in names:
        squared_norm = torch.zeros((), device=losses[name].device)
        active_count = 0
        for parameter, gradient in zip(parameters, gradients[name]):
            if gradient is None:
                continue
            detached = gradient.detach()
            squared_norm = squared_norm + torch.sum(detached * detached)
            active_count += parameter.numel()
        norm = torch.sqrt(squared_norm)
        squared_norms[name] = squared_norm
        active_counts[name] = active_count
        metrics["gradient/{}/raw_norm".format(name)] = _finite_float(norm.item())
        metrics["gradient/{}/weighted_norm".format(name)] = _finite_float(
            (norm * abs(float(weights[name]))).item()
        )
        metrics["gradient/{}/active_parameter_fraction".format(name)] = (
            active_count / parameter_count
        )

    total_squared_norm = torch.zeros((), device=losses[names[0]].device)
    for parameter_index, parameter in enumerate(parameters):
        combined = None
        for name in names:
            gradient = gradients[name][parameter_index]
            if gradient is None:
                continue
            weighted = gradient.detach() * float(weights[name])
            combined = weighted if combined is None else combined + weighted
        if combined is not None:
            total_squared_norm = total_squared_norm + torch.sum(combined * combined)
    total_norm = torch.sqrt(total_squared_norm)
    metrics["gradient/weighted_total_norm"] = _finite_float(total_norm.item())
    for name in names:
        weighted_norm = metrics["gradient/{}/weighted_norm".format(name)]
        metrics["gradient/{}/weighted_norm_to_total".format(name)] = (
            weighted_norm / metrics["gradient/weighted_total_norm"]
            if metrics["gradient/weighted_total_norm"] > 0
            else None
        )

    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            dot = torch.zeros((), device=losses[left].device)
            overlap_count = 0
            for parameter, left_gradient, right_gradient in zip(
                parameters, gradients[left], gradients[right]
            ):
                if left_gradient is None or right_gradient is None:
                    continue
                dot = dot + torch.sum(left_gradient.detach() * right_gradient.detach())
                overlap_count += parameter.numel()
            denominator = torch.sqrt(squared_norms[left] * squared_norms[right])
            cosine = None
            if denominator.item() > 0:
                cosine = _finite_float((dot / denominator).item())
            pair = "{}__{}".format(left, right)
            metrics["gradient_cosine/{}".format(pair)] = cosine
            metrics["gradient_overlap/{}".format(pair)] = overlap_count / parameter_count

    return metrics


def build_update_record(
    epoch,
    batch_index,
    global_update,
    model,
    emb,
    logits,
    outlier_emb,
    local_normal_indices,
    losses,
    weights,
    gradient_record=None,
):
    if model.last_h_mean is None or model.last_reconstruction_displacement is None:
        raise RuntimeError("VecGAD diagnostic state is incomplete")

    center = model.last_h_mean.squeeze(0)
    normal_emb = emb[:, local_normal_indices, :].squeeze(0)
    normal_distances = torch.norm(normal_emb - center, p=2, dim=1)
    pseudo_distances = torch.norm(outlier_emb - center, p=2, dim=1)
    displacement_norms = torch.norm(model.last_reconstruction_displacement, p=2, dim=1)
    normal_norms = torch.norm(normal_emb, p=2, dim=1)
    pseudo_norms = torch.norm(outlier_emb, p=2, dim=1)

    flat_logits = logits.detach().reshape(-1)
    normal_count = int(local_normal_indices.numel())
    pseudo_count = int(outlier_emb.shape[0])
    if flat_logits.numel() != normal_count + pseudo_count:
        raise RuntimeError("training logit partition does not match normal/pseudo counts")

    record = {
        "record_type": "optimizer_update",
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "global_update": int(global_update),
        "normal_count": normal_count,
        "pseudo_count": pseudo_count,
        "loss/bce_raw": _finite_float(losses["bce"].detach().item()),
        "loss/token_rec_raw": _finite_float(losses["token_rec"].detach().item()),
        "loss/emb_rec_raw": _finite_float(losses["emb_rec"].detach().item()),
        "loss/rec_combined_raw": _finite_float(
            losses["rec_combined"].detach().item()
        ),
        "loss/hsc_raw": _finite_float(losses["hsc"].detach().item()),
        "loss/objective_raw": _finite_float(losses["objective"].detach().item()),
    }
    record["loss/token_rec_weighted"] = record["loss/token_rec_raw"] * float(
        weights["token_rec"]
    )
    record["loss/emb_rec_weighted"] = record["loss/emb_rec_raw"] * float(
        weights["emb_rec"]
    )
    record["loss/rec_combined_weighted"] = (
        record["loss/rec_combined_raw"] * float(weights["rec_combined"])
    )
    record["loss/bce_weighted"] = record["loss/bce_raw"] * float(weights["bce"])
    record["loss/hsc_weighted"] = record["loss/hsc_raw"] * float(weights["hsc"])
    record["loss/true_weighted_total"] = (
        record["loss/bce_weighted"]
        + record["loss/rec_combined_weighted"]
        + record["loss/hsc_weighted"]
    )
    record["loss/objective_reconstruction_abs_error"] = abs(
        record["loss/objective_raw"] - record["loss/true_weighted_total"]
    )
    if not math.isclose(
        record["loss/objective_raw"],
        record["loss/true_weighted_total"],
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise RuntimeError("diagnostic loss decomposition does not match objective")

    inner = pseudo_distances < float(model.args.ring_R_min)
    outer = pseudo_distances > float(model.args.ring_R_max)
    record["hsc/inner_violation_rate"] = float(inner.float().mean().item())
    record["hsc/outer_violation_rate"] = float(outer.float().mean().item())
    record["hsc/shell_hit_rate"] = float((~inner & ~outer).float().mean().item())
    record["collapse/normal_embedding_variance_mean"] = _finite_float(
        normal_emb.detach().float().var(dim=0, unbiased=False).mean().item()
    )
    record["collapse/pseudo_embedding_variance_mean"] = _finite_float(
        outlier_emb.detach().float().var(dim=0, unbiased=False).mean().item()
    )
    record["collapse/normal_centered_rms"] = _finite_float(
        torch.sqrt(
            torch.mean(
                (normal_emb.detach().float() - normal_emb.detach().float().mean(dim=0))
                ** 2
            )
        ).item()
    )
    record["collapse/pseudo_centered_rms"] = _finite_float(
        torch.sqrt(
            torch.mean(
                (outlier_emb.detach().float() - outlier_emb.detach().float().mean(dim=0))
                ** 2
            )
        ).item()
    )

    for prefix, values in (
        ("distance/normal_to_center", normal_distances),
        ("distance/pseudo_to_center", pseudo_distances),
        ("norm/normal_embedding", normal_norms),
        ("norm/pseudo_embedding", pseudo_norms),
        ("norm/reconstruction_displacement", displacement_norms),
        ("score/train_all", flat_logits),
        ("score/train_normal", flat_logits[:normal_count]),
        ("score/train_pseudo", flat_logits[normal_count:]),
    ):
        record.update(distribution_metrics(prefix, values))

    if gradient_record is not None:
        record.update(gradient_record)
    return record


def wandb_update_metrics(record):
    batch_prefix = "diagnostic/batch_{}/".format(record["batch_index"])
    excluded = {"record_type", "epoch", "batch_index"}
    return {
        batch_prefix + key: value
        for key, value in record.items()
        if key not in excluded
    }


def append_jsonl(path, record):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
