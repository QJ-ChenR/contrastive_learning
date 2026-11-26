"""Reproducible training, validation, resource probes, and explicit final evaluation."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .common import package_versions, sha256, source_hashes, write_json, write_tsv
from .config import TrainingConfig, load_config
from .data import TripletSampler, load_cohort
from .metrics import retrieval_metrics


def cohort_for(config):
    return load_cohort(config.project_root, config.path("split_dir"), config.path("matrix_path"))


def sampler_for(cohort, seed):
    ids = cohort.ids_by_split["train"]
    return TripletSampler(ids, cohort.scores(ids, ids), seed)


def audit(config):
    """Validate the fixed split and training supervision without loading any model."""
    cohort = cohort_for(config)
    sampler = sampler_for(cohort, config.seed)
    triplets = sampler.epoch(0)
    lengths = [len(cohort.sequence(key)) for key in cohort.ids_by_split["train"]]
    gaps = np.asarray([row[3] for row in triplets])
    return {
        "status": "passed", "schema": "dna_training_audit_v1",
        "split_counts": {name: len(ids) for name, ids in cohort.ids_by_split.items()},
        "train_sequences_verified": len(lengths),
        "train_length_bp": {"min": min(lengths), "median": float(np.median(lengths)), "max": max(lengths)},
        "sampler": sampler.metadata,
        "epoch_0": {"triplets": len(triplets), "unique_anchors": len({row[0] for row in triplets}),
                    "mean_absolute_target_gap": float(np.abs(gaps).mean()),
                    "zero_gap_triplets": int(np.count_nonzero(gaps == 0))},
        "input_fingerprint": cohort.fingerprint,
        "training_loss": "MSE((cos(anchor,b)-cos(anchor,c))-(S(anchor,b)-S(anchor,c)))",
        "heldout_relations_used_for_sampling": False,
    }


def tokenize_ids(model, cohort, ids, output=None):
    tokens, rows = {}, []
    for index, key in enumerate(ids):
        try:
            tokens[key], metadata = model.tokenize(cohort.sequence(key))
        except Exception as error:
            error.add_note(f"Tokenization failed for BGC {key}; no input was truncated or removed")
            raise
        rows.append({"bgc_id": key, "split": cohort.records[key]["split_status"], **metadata})
        if (index + 1) % 100 == 0 or index + 1 == len(ids):
            print(f"Tokenized {index + 1}/{len(ids)} complete BGCs", flush=True)
    if output is not None:
        write_tsv(output, rows)
    return tokens, rows


def prepare_token_inputs(model, tokens):
    """Cache only validated IDs/masks, never parameter-dependent representations."""
    prepared = {key: model.prepare_tokens(ids) for key, ids in tokens.items()}
    byte_count = sum(value.input_ids.numel() * value.input_ids.element_size()
                     + value.attention_mask.numel() * value.attention_mask.element_size()
                     for value in prepared.values())
    print(f"Prepared {len(prepared)} DNA inputs on {model.device}: {byte_count / 2**20:.1f} MiB", flush=True)
    return prepared


def _signature(config, cohort, model):
    # Paths are operational, not model/data identity. Hashes permit moving the
    # project and original checkpoint together without changing the experiment.
    settings = config.to_dict()
    for key in ("project_root", "split_dir", "matrix_path", "output_dir", "checkpoint",
                "tokenizer_path", "config_path"):
        settings.pop(key)
    metadata = model.metadata
    for key in ("checkpoint_path", "config_path", "tokenizer_path"):
        metadata.pop(key, None)
    return json.loads(json.dumps({
        "settings": settings, "input": cohort.fingerprint, "model": metadata,
        "code": source_hashes(), "versions": package_versions(),
    }, sort_keys=True))


def _atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema") != "dna_training_v1":
        raise ValueError("Unsupported training checkpoint")
    return state


@contextmanager
def run_lock(directory):
    """OS lock is released on interruption; no stale-lock deletion is necessary."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is using {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _base_cache(model, tokens, ids, directory, signature):
    """Detached caches are allowed exclusively when the backbone stays frozen."""
    import torch
    if model.mode == "lora":
        raise ValueError("LoRA cannot train from detached frozen feature caches")
    path = Path(directory) / "base_features.npz"
    metadata_path = Path(directory) / "base_features.json"
    tag = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("signature") != tag or metadata.get("sha256") != sha256(path):
            raise ValueError("Frozen feature cache identity/hash mismatch")
        with np.load(path, allow_pickle=False) as saved:
            if saved["ids"].tolist() != ids:
                raise ValueError("Frozen feature cache ID/order mismatch")
            vectors = saved["embeddings"].copy()
        if vectors.shape != (len(ids), model.hidden_size) or not np.isfinite(vectors).all():
            raise ValueError("Invalid frozen feature cache")
    else:
        model.eval()
        vectors = []
        with torch.no_grad():
            for index, key in enumerate(ids):
                try:
                    vectors.append(model.base_tokens(tokens[key]).detach().cpu().numpy())
                except Exception as error:
                    error.add_note(f"Frozen encoding failed for BGC {key}; no sample was skipped")
                    raise
                if (index + 1) % 50 == 0 or index + 1 == len(ids):
                    print(f"Frozen base features {index + 1}/{len(ids)}", flush=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        np.savez(path, ids=np.asarray(ids), embeddings=vectors)
        write_json(metadata_path, {"signature": tag, "sha256": sha256(path), "kind": "normalized_frozen_base"})
    return {key: torch.from_numpy(vector.copy()) for key, vector in zip(ids, vectors)}


def encode_ids(model, tokens, ids, base_features=None):
    import torch
    model.eval()
    vectors = []
    with torch.no_grad():
        for index, key in enumerate(ids):
            try:
                vector = model.project(base_features[key]) if base_features is not None else model.encode_tokens(tokens[key])
                vectors.append(vector.detach().cpu().numpy())
            except Exception as error:
                error.add_note(f"Encoding failed for BGC {key}; no sample was skipped")
                raise
            if base_features is None and ((index + 1) % 100 == 0 or index + 1 == len(ids)):
                print(f"Encoded {index + 1}/{len(ids)} BGCs with current model", flush=True)
    return np.asarray(vectors, dtype=np.float32)


def _export_evaluation(output, metrics, embeddings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "embeddings.npz", **embeddings)
    write_json(output / "metrics.json", metrics)
    write_tsv(output / "per_query.tsv", metrics["per_query"])


def evaluate_model(model, cohort, tokens, split, k, base_features=None, export_dir=None, *, return_embeddings=False):
    query_ids, gallery_ids = cohort.ids_by_split[split], cohort.ids_by_split["train"]
    # Every validation pass re-encodes the gallery with the current LoRA model.
    # A frozen backbone may reuse ONLY its pre-projection vectors.
    gallery = encode_ids(model, tokens, gallery_ids, base_features)
    queries = encode_ids(model, tokens, query_ids, base_features)
    metrics = retrieval_metrics(query_ids, gallery_ids, queries, gallery,
                                cohort.scores(query_ids, gallery_ids), k=k)
    embeddings = {"query_ids": np.asarray(query_ids), "gallery_ids": np.asarray(gallery_ids),
                  "query_embeddings": queries, "gallery_embeddings": gallery}
    if export_dir is not None:
        _export_evaluation(export_dir, metrics, embeddings)
    return (metrics, embeddings) if return_embeddings else metrics


def _state_hash(state):
    """Hash small CPU adapter/head states, without serializing frozen weights."""
    import torch
    digest = hashlib.sha256(json.dumps(state["identity"], sort_keys=True).encode())
    for kind in ("head", "adapters"):
        for name, value in sorted(state[kind].items()):
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([kind, name, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validation_snapshot(embeddings, epoch, signature, model_state):
    """Keep an exact selected-checkpoint evaluation in the atomic checkpoint.

    CPU tensors stay compatible with torch.load(weights_only=True). They are
    only reused for exporting that same checkpoint's validation, never training.
    """
    import torch
    return {
        "schema": "dna_validation_snapshot_v1", "epoch": epoch,
        "signature_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
        "model_sha256": _state_hash(model_state),
        "query_ids": embeddings["query_ids"].tolist(), "gallery_ids": embeddings["gallery_ids"].tolist(),
        "query_embeddings": torch.from_numpy(embeddings["query_embeddings"].copy()),
        "gallery_embeddings": torch.from_numpy(embeddings["gallery_embeddings"].copy()),
        "embeddings_sha256": {
            key: hashlib.sha256(embeddings[key].tobytes()).hexdigest()
            for key in ("query_embeddings", "gallery_embeddings")
        },
    }


def _export_selected_validation(selected, cohort, k, output):
    """Return None for old checkpoints; reject a present but mismatched cache."""
    snapshot = selected.get("best_validation")
    if snapshot is None:
        return None
    expected_signature = hashlib.sha256(json.dumps(selected["signature"], sort_keys=True).encode()).hexdigest()
    if (snapshot.get("schema") != "dna_validation_snapshot_v1"
            or cohort.fingerprint != selected["signature"]["input"]
            or k != selected["signature"]["settings"]["eval_k"]
            or snapshot["epoch"] != selected["epoch"]
            or snapshot["signature_sha256"] != expected_signature
            or snapshot["model_sha256"] != _state_hash(selected["model"])
            or snapshot["query_ids"] != cohort.ids_by_split["validation"]
            or snapshot["gallery_ids"] != cohort.ids_by_split["train"]):
        raise ValueError("Selected validation cache identity mismatch")
    embeddings = {key: np.asarray(snapshot[key]) for key in ("query_ids", "gallery_ids")}
    width = selected["model"]["identity"]["embedding_dim"]
    for kind in ("query", "gallery"):
        key = f"{kind}_embeddings"
        value = snapshot[key].detach().cpu().numpy()
        if (value.dtype != np.float32 or value.shape != (len(embeddings[f"{kind}_ids"]), width)
                or not np.isfinite(value).all()
                or hashlib.sha256(value.tobytes()).hexdigest() != snapshot["embeddings_sha256"][key]):
            raise ValueError("Selected validation cache shape/content mismatch")
        embeddings[key] = value
    metrics = retrieval_metrics(snapshot["query_ids"], snapshot["gallery_ids"],
                                embeddings["query_embeddings"], embeddings["gallery_embeddings"],
                                cohort.scores(snapshot["query_ids"], snapshot["gallery_ids"]), k=k)
    if metrics["aggregate"]["ndcg_at_k"] != selected["best_score"]:
        raise ValueError("Selected validation cache score differs from checkpoint")
    _export_evaluation(output, metrics, embeddings)
    return metrics


def _checkpoint(model, optimizer, signature, config, epoch, step, best_score, best_epoch, stale, history, best_model,
                best_validation=None):
    from .engine import capture_rng
    return {
        "schema": "dna_training_v1", "config": config.to_dict(), "signature": signature,
        "model": model.trainable_state(), "optimizer": None if optimizer is None else optimizer.state_dict(),
        "rng": capture_rng(), "epoch": epoch, "optimizer_step": step,
        "best_score": best_score, "best_epoch": best_epoch, "stale_epochs": stale, "history": history,
        "best_model": best_model,
        "best_validation": best_validation,
    }


def _write_best(output, state):
    # last.pt is the transaction boundary. Keeping the best small adapter/head
    # state inside it permits repairing best.pt after interruption between the
    # two atomic writes. No frozen backbone weights are duplicated.
    selected = {**state, "model": state["best_model"], "epoch": state["best_epoch"], "optimizer": None}
    _atomic_torch_save(Path(output) / "best.pt", selected)


def train(config: TrainingConfig, resume=False, stop_after_epochs=None):
    """Train/validate only; final test evaluation requires a separate CLI command.

    ``stop_after_epochs`` permits an intentional, reproducible epoch-boundary
    interruption without changing the total planned LR schedule.
    """
    import torch
    from .engine import backward_triplet, finish_step, make_optimizer, restore_rng, seed_everything, set_learning_rate
    from .model import load_encoder

    config.validate()
    if stop_after_epochs is not None and (
        isinstance(stop_after_epochs, bool) or not isinstance(stop_after_epochs, int) or stop_after_epochs <= 0
    ):
        raise ValueError("stop_after_epochs must be a positive integer")
    output = config.path("output_dir")
    with run_lock(output):
        if not resume and any(path.name != ".lock" for path in output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}; use --resume or a new --output-dir")
        try:
            if resume:
                state = load_state(output / "last.pt")
            else:
                state = None
            seed_everything(config.seed)
            cohort = cohort_for(config)
            sampler = sampler_for(cohort, config.seed)
            model = load_encoder(config.model_settings())
            signature = _signature(config, cohort, model)
            if state is not None and state["signature"] != signature:
                raise ValueError("Resume identity differs: config, data, model, code, or dependencies changed")
            if state is not None:
                model.load_trainable_state(state["model"])
                _write_best(output, state)
            if state is None:
                write_json(output / "config.json", config.to_dict())
                write_json(output / "run.json", {
                    "schema": "dna_training_run_v1", "signature": signature, "model": model.metadata,
                    "sampler": sampler.metadata, "loss": "margin_mse", "target_gap_scale": 1.0,
                    "selection": f"validation_vs_train_gallery_ndcg@{config.eval_k}_gain_S",
                    "test_used_for_training_or_selection": False,
                    "gradient_replay": config.gradient_replay,
                    "input_cache": "validated_token_ids_and_masks_on_device",
                    "validation_cache": "selected_checkpoint_only_in_last_and_best",
                })
            ids = cohort.ids_by_split["train"] + cohort.ids_by_split["validation"]
            tokens, _ = tokenize_ids(model, cohort, ids, output / "token_lengths.tsv")
            tokens = prepare_token_inputs(model, tokens)
            features = _base_cache(model, tokens, ids, output, signature) if config.mode != "lora" else None
            optimizer = make_optimizer(model, config) if config.mode != "frozen" else None
            if state is not None:
                if optimizer is not None:
                    optimizer.load_state_dict(state["optimizer"])
                restore_rng(state["rng"])
                history = state["history"]
                start_epoch, step = state["epoch"] + 1, state["optimizer_step"]
                best_score, best_epoch, stale = state["best_score"], state["best_epoch"], state["stale_epochs"]
                best_model = state["best_model"]
                best_validation = state.get("best_validation")
            else:
                print("Initial validation: encoding current gallery and queries", flush=True)
                validation_started = time.monotonic()
                initial, embeddings = evaluate_model(model, cohort, tokens, "validation", config.eval_k, features,
                                                     return_embeddings=True)
                best_score = initial["aggregate"]["ndcg_at_k"]
                if best_score is None or not math.isfinite(best_score):
                    raise ValueError("Validation has no defined nDCG; cannot select a checkpoint")
                history = [{"epoch": 0, "train_loss": None, "optimizer_step": 0,
                            "validation_seconds": time.monotonic() - validation_started,
                            "validation": initial["aggregate"]}]
                best_epoch, stale, start_epoch, step = 0, 0, 1, 0
                best_model = model.trainable_state()
                best_validation = _validation_snapshot(embeddings, 0, signature, best_model)
                initial_state = _checkpoint(model, optimizer, signature, config, 0, 0, best_score, 0, 0, history,
                                            best_model, best_validation)
                _atomic_torch_save(output / "last.pt", initial_state)
                _write_best(output, initial_state)
                write_json(output / "history.json", history)
                print(f"Epoch 0 validation nDCG@{config.eval_k}={best_score:.6f}", flush=True)

            total_steps = math.ceil(len(cohort.ids_by_split["train"]) / config.triplets_per_update) * config.max_epochs
            completed_this_call = 0
            interrupted = False
            if optimizer is not None and stale < config.patience:
                for epoch in range(start_epoch, config.max_epochs + 1):
                    started = time.monotonic()
                    print(f"Epoch {epoch}: starting training updates", flush=True)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    triplets = sampler.epoch(epoch - 1)
                    losses, gradient_norms, step_seconds = [], [], []
                    checkpointed_triplets = 0
                    if model.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(model.device)
                    for offset in range(0, len(triplets), config.triplets_per_update):
                        step_started = time.monotonic()
                        batch = triplets[offset:offset + config.triplets_per_update]
                        set_learning_rate(optimizer, step, total_steps, config.warmup_ratio)
                        for triplet in batch:
                            losses.append(backward_triplet(model, tokens, triplet, divisor=len(batch),
                                                           replay=config.gradient_replay, base_features=features))
                            checkpointed_triplets += int(getattr(model, "active_gradient_checkpointing", False))
                        gradient_norms.append(finish_step(model, optimizer, config.max_grad_norm))
                        step_seconds.append(time.monotonic() - step_started)
                        step += 1
                        if offset == 0 or step % 10 == 0 or offset + len(batch) == len(triplets):
                            print(f"Epoch {epoch}: {min(offset + len(batch), len(triplets))}/{len(triplets)} "
                                  f"triplets, step={step}, loss={np.mean(losses):.6f}, "
                                  f"last_update_seconds={step_seconds[-1]:.2f}, "
                                  f"training_elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
                    training_seconds = time.monotonic() - started
                    peak = torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None
                    reserved = torch.cuda.max_memory_reserved(model.device) if model.device.type == "cuda" else None
                    print(f"Epoch {epoch}: training finished in {training_seconds:.1f}s; starting validation", flush=True)
                    validation_started = time.monotonic()
                    metrics, embeddings = evaluate_model(model, cohort, tokens, "validation", config.eval_k, features,
                                                         return_embeddings=True)
                    validation_seconds = time.monotonic() - validation_started
                    score = metrics["aggregate"]["ndcg_at_k"]
                    if score is None or not math.isfinite(score):
                        raise ValueError("Non-finite/undefined validation selection metric")
                    improved = score > best_score + config.min_delta
                    if improved:
                        best_score, best_epoch, stale = score, epoch, 0
                        best_model = model.trainable_state()
                        best_validation = _validation_snapshot(embeddings, epoch, signature, best_model)
                    else:
                        stale += 1
                    row = {
                        "epoch": epoch, "train_loss": float(np.mean(losses)), "optimizer_step": step,
                        "triplets": len(triplets), "unique_anchors": len({triplet[0] for triplet in triplets}),
                        "mean_absolute_target_gap": float(np.mean([abs(triplet[3]) for triplet in triplets])),
                        "mean_gradient_norm": float(np.mean(gradient_norms)),
                        "training_peak_cuda_allocated_bytes": peak,
                        "training_peak_cuda_reserved_bytes": reserved,
                        "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                        "mean_update_seconds": float(np.mean(step_seconds)),
                        "training_triplets_per_second": len(triplets) / max(training_seconds, 1e-9),
                        "checkpointed_triplets": checkpointed_triplets,
                        "epoch_seconds_including_validation": time.monotonic() - started,
                        "validation": metrics["aggregate"], "selected": improved,
                    }
                    history.append(row)
                    saved = _checkpoint(model, optimizer, signature, config, epoch, step, best_score, best_epoch, stale,
                                        history, best_model, best_validation)
                    _atomic_torch_save(output / "last.pt", saved)
                    if improved:
                        _write_best(output, saved)
                    write_json(output / "history.json", history)
                    print(f"Epoch {epoch}: loss={row['train_loss']:.6f}, validation nDCG@{config.eval_k}={score:.6f}, "
                          f"best_epoch={best_epoch}, stale={stale}/{config.patience}", flush=True)
                    completed_this_call += 1
                    if stale >= config.patience:
                        break
                    if stop_after_epochs is not None and completed_this_call >= stop_after_epochs and epoch < config.max_epochs:
                        interrupted = True
                        break
            selected = load_state(output / "best.pt")
            best_metrics = _export_selected_validation(selected, cohort, config.eval_k, output / "validation")
            reused_validation = best_metrics is not None
            if best_metrics is None:
                model.load_trainable_state(selected["model"])
                best_metrics = evaluate_model(model, cohort, tokens, "validation", config.eval_k, features,
                                              output / "validation")
            else:
                print(f"Exported cached validation for best epoch {selected['epoch']}; no encoder recomputation", flush=True)
            summary = {
                "status": "interrupted_at_epoch_boundary" if interrupted else "completed",
                "mode": config.mode, "seed": config.seed, "best_epoch": selected["epoch"],
                "completed_epochs": history[-1]["epoch"], "optimizer_steps": step,
                "best_validation": best_metrics["aggregate"], "test_evaluated": False,
                "best_checkpoint": "best.pt", "resume_checkpoint": "last.pt",
                "reused_best_validation": reused_validation,
            }
            write_json(output / "summary.json", summary)
            (output / "failure.json").unlink(missing_ok=True)
            return summary
        except (Exception, KeyboardInterrupt) as error:
            # Preserve the last completed epoch. Do not mask the exception or
            # silently remove a long sequence from the training cohort.
            write_json(output / "failure.json", {
                "exception": type(error).__name__, "message": str(error),
                "notes": getattr(error, "__notes__", []),
                "resume_policy": "last.pt contains the last completed epoch, if present",
            })
            raise


def evaluate_run(run_dir, split="validation", overrides=None):
    """Explicit evaluation of best.pt; test is never called from train()."""
    allowed = {"project_root", "checkpoint", "tokenizer_path", "config_path", "device", "dtype", "attention_backend"}
    if set(overrides or {}) - allowed:
        raise ValueError("Evaluation overrides may change paths/device/backend, not the saved training or metric protocol")
    from .engine import seed_everything
    from .model import load_encoder
    run_dir = Path(run_dir).resolve()
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    with run_lock(run_dir):
        # Repair an interrupted best.pt write from the authoritative last.pt.
        latest = load_state(run_dir / "last.pt")
        best_path = run_dir / "best.pt"
        saved = load_state(best_path) if best_path.exists() else None
        if saved is None or saved["epoch"] != latest["best_epoch"] or saved["signature"] != latest["signature"]:
            _write_best(run_dir, latest)
            saved = load_state(best_path)
        config = load_config(run_dir / "config.json", overrides)
        seed_everything(config.seed)
        cohort = cohort_for(config)
        if cohort.fingerprint != saved["signature"]["input"]:
            raise ValueError("Evaluation data differs from training split/matrix")
        model = load_encoder(config.model_settings())
        model.load_trainable_state(saved["model"])
        ids = cohort.ids_by_split["train"] + cohort.ids_by_split[split]
        tokens, _ = tokenize_ids(model, cohort, ids)
        tokens = prepare_token_inputs(model, tokens)
        destination = run_dir / f"evaluation_{split}"
        metrics = evaluate_model(model, cohort, tokens, split, config.eval_k, export_dir=destination)
        write_json(destination / "provenance.json", {
            "checkpoint_sha256": sha256(run_dir / "best.pt"), "checkpoint_epoch": saved["epoch"],
            "config": config.to_dict(), "model": model.metadata, "input_fingerprint": cohort.fingerprint,
            "code_sha256": source_hashes(), "versions": package_versions(), "split": split,
        })
        return metrics["aggregate"]


def probe(config, destination):
    """Measure full-length forward/backward/optimizer/save-reload on train only.

    Probe median/P90/longest anchors and the three longest distinct inputs
    together. No probe weights enter a training run.
    """
    import torch
    from .engine import backward_triplet, finish_step, make_optimizer, seed_everything
    from .model import load_encoder
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {"status": "running", "config": config.to_dict(), "probes": rows,
              "input_policy": "full DNA; failure is fatal; no length exclusion or truncation",
              "resource_settings": {
                  "gradient_replay": config.gradient_replay,
                  "gradient_checkpointing": config.gradient_checkpointing,
                  "checkpoint_token_budget": config.checkpoint_token_budget,
                  "dtype": config.dtype, "attention_backend": config.attention_backend,
                  "encoder_forward_count_scope": "encode_tokens calls; excludes internal layer checkpoint recomputation",
              }}
    write_json(destination / "probe.json", report)
    try:
        config.validate()
        seed_everything(config.seed)
        cohort = cohort_for(config)
        model = load_encoder(config.model_settings())
        report.update(model=model.metadata, input_fingerprint=cohort.fingerprint)
        report["device"] = {
            "type": model.device.type, "name": torch.cuda.get_device_name(model.device)
            if model.device.type == "cuda" else str(model.device),
            "total_memory_bytes": torch.cuda.get_device_properties(model.device).total_memory
            if model.device.type == "cuda" else None,
        }
        train_ids = cohort.ids_by_split["train"]
        tokens, _ = tokenize_ids(model, cohort, train_ids, destination / "train_token_lengths.tsv")
        tokens = {key: model.prepare_tokens(value) for key, value in tokens.items()}
        report["input_tensor_cache"] = {
            "count": len(tokens), "device": str(model.device),
            "bytes": sum(value.input_ids.numel() * value.input_ids.element_size()
                         + value.attention_mask.numel() * value.attention_mask.element_size()
                         for value in tokens.values()),
            "contents": "fixed token IDs and attention masks only; no cached LoRA representations",
        }
        order = sorted(train_ids, key=lambda key: (len(tokens[key]), key))
        if len(order) < 3:
            raise ValueError("A resource probe requires at least three distinct training BGCs")
        cases = []
        for label, quantile in (("median", 0.5), ("p90", 0.9), ("longest", 1.0)):
            anchor = order[round((len(order) - 1) * quantile)]
            others = [key for key in order if key != anchor][:2]
            cases.append((label, (anchor, *others)))
        if config.mode == "lora" and config.gradient_checkpointing and config.checkpoint_token_budget > 0:
            eligible = [triplet for triplet in sampler_for(cohort, config.seed).epoch(0)
                        if sum(len(tokens[key]) + 2 for key in triplet[:3]) <= config.checkpoint_token_budget]
            report["activation_budget_boundary"] = {
                "source": "epoch_0_training_sampler", "eligible_triplets": len(eligible),
                "status": "included" if eligible else "no_sample_within_budget",
            }
            if eligible:
                boundary = max(eligible, key=lambda triplet: (
                    sum(len(tokens[key]) + 2 for key in triplet[:3]), triplet[:3],
                ))
                cases.append(("activation_budget_boundary", tuple(boundary[:3])))
        cases.append(("longest_triplet", tuple(reversed(order[-3:]))))
        optimizer = make_optimizer(model, config) if config.mode != "frozen" else None
        for label, members in cases:
            anchor, first, second = members
            scores = cohort.scores([anchor], [first, second])[0]
            triplet = (*members, float(scores[0] - scores[1]))
            lengths = [len(tokens[key]) + 2 for key in members]
            case = {"case": label, "anchor": anchor,
                    "base_count": int(cohort.records[anchor]["dna_length_bp"]),
                    "input_tokens": lengths[0], "triplet": list(members),
                    "triplet_input_tokens": lengths, "triplet_total_input_tokens": sum(lengths)}
            report["active_probe"] = case
            write_json(destination / "probe.json", report)
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
                torch.cuda.reset_peak_memory_stats(model.device)
            started = time.monotonic()
            if optimizer is None:
                model.eval()
                forward_ids = members if label == "longest_triplet" else (anchor,)
                with torch.no_grad():
                    for key in forward_ids:
                        model.encode_tokens(tokens[key])
                loss, norm = None, None
                forward_calls = len(forward_ids)
            else:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = backward_triplet(model, tokens, triplet, replay=config.gradient_replay)
                norm = finish_step(model, optimizer, config.max_grad_norm)
                forward_calls = 6 if config.gradient_replay else 3
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            rows.append({**case, "encoder_forward_calls": forward_calls,
                         "active_gradient_checkpointing": model.active_gradient_checkpointing,
                         "loss": loss, "gradient_norm": norm, "seconds": time.monotonic() - started,
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(model.device)
                         if model.device.type == "cuda" else None,
                         "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(model.device)
                         if model.device.type == "cuda" else None})
            report.pop("active_probe")
            write_json(destination / "probe.json", report)
            print(f"Probed {label} {anchor}: {lengths} full input tokens", flush=True)
        model.eval()
        check_ids = tokens[cases[0][1][0]]
        with torch.no_grad():
            expected = model.encode_tokens(check_ids).detach().cpu()
        saved = model.trainable_state()
        with tempfile.TemporaryDirectory(prefix="dna-reload-probe-", dir=destination) as temporary:
            path = Path(temporary) / "state.pt"
            torch.save(saved, path)
            # Loading a second encoder after releasing the first also checks
            # that only adapter/head state plus the original checkpoint suffice.
            del optimizer, model
            if config.device.startswith("cuda"):
                torch.cuda.empty_cache()
            model = load_encoder(config.model_settings())
            model.load_trainable_state(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            with torch.no_grad():
                actual = model.encode_tokens(check_ids).detach().cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        report.update(status="passed", save_reload_max_abs_error=float((actual - expected).abs().max()))
        write_json(destination / "probe.json", report)
        return report
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                              "notes": getattr(error, "__notes__", [])})
        write_json(destination / "probe.json", report)
        raise
