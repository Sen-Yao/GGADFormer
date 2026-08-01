"""Strict direction--magnitude controls used by the anonymous package."""

import torch


CONTROL_NAMES = ("full", "random_dir", "random_mag", "random_both", "constant_mag")


def normalize_direction(vectors, eps=1e-12):
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    return vectors / norms.clamp_min(eps)


def permute_magnitudes_cyclic(magnitudes, generator=None):
    """Preserve the exact magnitude multiset with a non-zero cyclic shift."""
    if magnitudes.dim() != 2 or magnitudes.shape[1] != 1:
        raise ValueError("magnitudes must have shape [n, 1]")
    count = magnitudes.shape[0]
    if count < 2:
        return magnitudes.clone()
    kwargs = {"device": magnitudes.device}
    if generator is not None:
        kwargs["generator"] = generator
    shift = int(torch.randint(1, count, (1,), **kwargs).item())
    return torch.roll(magnitudes, shifts=shift, dims=0)


def apply_control(vector, control, direction_generator=None, magnitude_generator=None):
    """Apply one strict control while retaining a differentiable projection path."""
    if control not in CONTROL_NAMES:
        raise ValueError("unsupported control: {}".format(control))
    if control == "full":
        return vector

    magnitudes = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    if control in ("random_dir", "random_both"):
        random_kwargs = {"size": vector.shape, "dtype": vector.dtype, "device": vector.device}
        if direction_generator is not None:
            random_kwargs["generator"] = direction_generator
        direction = normalize_direction(torch.randn(**random_kwargs))
    else:
        direction = normalize_direction(vector)

    if control in ("random_mag", "random_both"):
        magnitude = permute_magnitudes_cyclic(magnitudes, magnitude_generator)
    elif control == "constant_mag":
        magnitude = magnitudes.mean().expand_as(magnitudes)
    else:
        magnitude = magnitudes
    return direction * magnitude
