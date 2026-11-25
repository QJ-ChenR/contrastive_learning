"""Margin-MSE updates with optional exact, memory-bounded gradient replay.

Replay caches representations only within an update, then recomputes each
sequence using the same parameters and random state. It is not a persistent
feature cache and does not freeze the LoRA encoder.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch; CUDA kernels need not be bit deterministic."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    """Return a ``torch.load(weights_only=True)`` compatible random-state dict."""
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": np_state[0], "keys": np_state[1].tolist(),
            "position": int(np_state[2]), "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() and torch.cuda.is_initialized() else [],
    }


def restore_rng(state: dict) -> None:
    """Restore all recorded generators; CUDA state requires the same topology."""
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((
        np_state["algorithm"], np.asarray(np_state["keys"], dtype=np.uint32),
        int(np_state["position"]), int(np_state["has_gauss"]),
        float(np_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("Saved CUDA random state requires the same CUDA device count")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def _encode(model, tokens, bgc_id):
    try:
        result = model.encode_tokens(tokens[bgc_id])
        if result.ndim != 1 or not torch.isfinite(result).all():
            raise ValueError("Encoder must return one finite vector")
        return result
    except Exception as exc:
        exc.add_note(f"While encoding BGC {bgc_id}; the sample was not skipped or truncated")
        raise


def _margin_loss(vectors: torch.Tensor, target: float) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape[0] != 3:
        raise ValueError("A margin comparison requires exactly three vectors")
    margin = torch.sum(vectors[0].float() * vectors[1].float()) - torch.sum(
        vectors[0].float() * vectors[2].float()
    )
    loss = (margin - target).square()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite Margin-MSE loss")
    return loss


def _finite_gradients(model) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite gradient in {name}")


def backward_triplet(
    model, tokens: dict[str, list[int]], triplet: tuple[str, str, str, float],
    divisor: float = 1.0, replay: bool = True,
    base_features: dict[str, torch.Tensor] | None = None,
) -> float:
    """Accumulate one triplet's gradient and return its unscaled squared error.

    ``divisor`` is the actual number of triplets in this optimizer update,
    including a shorter final update. This function never steps an optimizer.
    The caller sets model.train() before invoking it.
    """
    if isinstance(divisor, bool) or not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("divisor must be finite and positive")
    if len(triplet) != 4 or len(set(triplet[:3])) != 3:
        raise ValueError("Triplets require three distinct BGC IDs and a continuous gap")
    anchor, first, second, target = triplet
    target = float(target)
    if not math.isfinite(target) or not -1 <= target <= 1:
        raise ValueError("The chemical similarity gap must be finite and in [-1, 1]")
    ids = (anchor, first, second)
    if base_features is not None:
        if getattr(model, "mode", None) != "head":
            raise ValueError("Persistent frozen base features are allowed only in head mode")
        base = torch.stack([base_features[bgc_id].detach() for bgc_id in ids]).to(
            device=model.device, dtype=torch.float32,
        )
        vectors = model.project(base)
        loss = _margin_loss(vectors, target)
        (loss / divisor).backward()
        _finite_gradients(model)
        return float(loss.detach().cpu())

    if not replay:
        vectors = torch.stack([_encode(model, tokens, bgc_id) for bgc_id in ids])
        loss = _margin_loss(vectors, target)
        (loss / divisor).backward()
        _finite_gradients(model)
        return float(loss.detach().cpu())

    states, representations = [], []
    with torch.no_grad():
        for bgc_id in ids:
            states.append(capture_rng())
            representations.append(_encode(model, tokens, bgc_id).detach())
    after_forward = capture_rng()
    vectors = torch.stack(representations).detach().requires_grad_(True)
    loss = _margin_loss(vectors, target)
    (loss / divisor).backward()
    gradients = vectors.grad.detach()
    if not torch.isfinite(gradients).all():
        raise FloatingPointError("Non-finite representation gradient during replay")
    # Restore the post-first-pass state even if the replay/backward fails. No
    # optimizer update can occur between these recomputations.
    try:
        for bgc_id, state, reference, gradient in zip(ids, states, representations, gradients):
            restore_rng(state)
            encoded = _encode(model, tokens, bgc_id)
            # With identical parameters/RNG, recomputation must preserve the
            # representation. Allow mixed-precision/backend rounding tolerance.
            if not torch.allclose(encoded.detach(), reference, rtol=2e-3, atol=2e-4):
                raise RuntimeError(f"Gradient replay changed representation for {bgc_id}")
            torch.autograd.backward(encoded, gradient)
    finally:
        restore_rng(after_forward)
    _finite_gradients(model)
    return float(loss.detach().cpu())


def _config_value(config, key):
    return config[key] if isinstance(config, dict) else getattr(config, key)


def make_optimizer(model, config) -> torch.optim.AdamW:
    """Optimize trainable projection/adapter tensors, never frozen weights."""
    head_ids = {id(parameter) for parameter in model.head.parameters()}
    head, adapters = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (head if id(parameter) in head_ids else adapters).append(parameter)
    if getattr(model, "mode", None) == "head" and adapters:
        raise ValueError("Head-only mode unexpectedly contains trainable backbone parameters")
    groups = []
    for name, parameters, rate_name in (("head", head, "head_lr"), ("lora", adapters, "lora_lr")):
        if parameters:
            rate = float(_config_value(config, rate_name))
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError(f"{rate_name} must be finite and positive")
            groups.append({"params": parameters, "lr": rate, "initial_lr": rate, "name": name})
    if not groups:
        raise ValueError("No trainable parameters; frozen mode does not use an optimizer")
    weight_decay = float(_config_value(config, "weight_decay"))
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def set_learning_rate(
    optimizer, step: int, total_steps: int, warmup_ratio: float,
) -> float:
    """Set LR for zero-based upcoming update ``step``; return its multiplier.

    Warmup uses at least one update when requested. Decay reaches zero at
    ``step == total_steps`` (after the planned last update). All scheduler
    information is either explicit here or stored in optimizer.initial_lr.
    """
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= total_steps:
        raise ValueError("step must be an integer between 0 and total_steps")
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup = min(total_steps, max(1, round(total_steps * warmup_ratio))) if warmup_ratio else 0
    if step == total_steps:
        factor = 0.0
    elif step < warmup:
        factor = (step + 1) / warmup
    else:
        factor = (total_steps - step) / max(1, total_steps - warmup)
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            raise ValueError("Optimizer group has no saved initial_lr")
        group["lr"] = group["initial_lr"] * factor
    return factor


def finish_step(model, optimizer, max_grad_norm: float) -> float:
    """Validate and clip accumulated gradients, update, clear, return old norm."""
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    if not parameters:
        raise ValueError("No accumulated gradients to update")
    _finite_gradients(model)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.detach().cpu())
