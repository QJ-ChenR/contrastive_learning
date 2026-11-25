"""Gradient-enabled full-DNA ModernBERT representations.

The tokenization, content-token mean and pre-projection normalization mirror
``BGC_annotation/data_preparation/frozen_dna.py``.  The frozen extractor is not
imported: its inference-only execution cannot train encoder adapters.
"""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_LORA_TARGETS = ["attn.Wqkv", "attn.Wo"]


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_files(directory: Path) -> list[Path]:
    """Require a complete local HF checkpoint, including indexed shards."""
    directory = directory.resolve()
    for name in (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    ):
        candidate = directory / name
        if not candidate.is_file():
            continue
        if not name.endswith(".index.json"):
            return [candidate]
        index = json.loads(candidate.read_text())
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Invalid checkpoint shard index: {candidate}")
        files = [candidate]
        for shard in sorted(set(mapping.values())):
            path = (directory / shard).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Checkpoint shard escapes the checkpoint directory")
            if not path.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {path}")
            files.append(path)
        return files
    raise FileNotFoundError(
        f"No trained model weights in {directory}; config-only loading is forbidden"
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unit(vector: torch.Tensor, description: str) -> torch.Tensor:
    vector = vector.float()
    norms = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if not torch.isfinite(vector).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError(f"{description} is zero or non-finite")
    return vector / norms


class DNAEncoder(nn.Module):
    """One complete BGC per backbone forward; independent unit BGC vectors.

    ``base_tokens`` returns the normalized, pre-projection representation.
    Only the frozen/head arms may cache it between optimization steps.
    ``project`` accepts one vector or a batch of these normalized vectors.
    """

    def __init__(self, backbone, tokenizer, settings: dict, provenance: dict | None = None):
        super().__init__()
        self.settings = dict(settings)
        self.mode = settings.get("mode", "lora")
        if self.mode not in {"frozen", "head", "lora"}:
            raise ValueError("mode must be frozen, head or lora")
        self.device = torch.device(settings.get("device", "cpu"))
        self.dtype_name = settings.get("dtype", "float32")
        if self.dtype_name not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        self.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype_name]
        self.attention_backend = settings.get("attention_backend", "sdpa")
        if self.attention_backend not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("Unsupported attention backend")
        if self.attention_backend == "flash_attention_2" and (
            self.dtype_name != "bfloat16" or self.device.type != "cuda"
        ):
            raise ValueError("flash_attention_2 requires CUDA and bfloat16")
        self.max_tokens = settings.get("max_tokens")
        if self.max_tokens is not None:
            _positive_int(self.max_tokens, "max_tokens")
            if self.max_tokens < 3:
                raise ValueError("max_tokens includes CLS/SEP and must be at least 3")
        self.seed = settings.get("seed", 42)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        config = backbone.config
        if getattr(config, "model_type", None) != "modernbert":
            raise ValueError("Expected a ModernBERT backbone")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer with exact DNA offsets is required")
        self.hidden_size = int(config.hidden_size)
        self.config_max_position_embeddings = int(config.max_position_embeddings)
        if len(tokenizer) != int(config.vocab_size):
            raise ValueError("Tokenizer vocabulary size differs from checkpoint")
        if backbone.get_input_embeddings().weight.shape[0] != len(tokenizer):
            raise ValueError("Checkpoint input embedding rows differ from tokenizer")
        self.special_ids = {}
        for name in ("cls", "sep", "pad", "unk"):
            token_id = getattr(tokenizer, f"{name}_token_id", None)
            if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < len(tokenizer):
                raise ValueError(f"Invalid tokenizer {name}_token_id: {token_id}")
            self.special_ids[name] = token_id
        if len(set(self.special_ids.values())) != 4:
            raise ValueError("CLS, SEP, PAD and UNK must have distinct IDs")
        if config.pad_token_id != self.special_ids["pad"]:
            raise ValueError("Tokenizer PAD ID differs from checkpoint")
        padding_idx = getattr(backbone.get_input_embeddings(), "padding_idx", None)
        if padding_idx is not None and padding_idx != self.special_ids["pad"]:
            raise ValueError("Tokenizer PAD ID differs from embedding padding_idx")
        corrections = {}
        for name, token_id in {
            "cls_token_id": self.special_ids["cls"], "bos_token_id": self.special_ids["cls"],
            "sep_token_id": self.special_ids["sep"], "eos_token_id": self.special_ids["sep"],
        }.items():
            old = getattr(config, name, None)
            if old != token_id:
                corrections[name] = {"config_value": old, "tokenizer_value": token_id}
                setattr(config, name, token_id)
        self.tokenizer = tokenizer
        backbone.requires_grad_(False)
        backbone.eval()
        self.gradient_checkpointing = bool(settings.get("gradient_checkpointing", True)) and self.mode == "lora"
        self.lora_modules: list[str] = []
        self.lora_settings: dict = {}
        if self.mode == "lora":
            from peft import LoraConfig, get_peft_model

            rank = _positive_int(settings.get("lora_rank", 8), "lora_rank")
            alpha = _positive_int(settings.get("lora_alpha", 16), "lora_alpha")
            dropout = float(settings.get("lora_dropout", 0.05))
            if not 0 <= dropout < 1:
                raise ValueError("lora_dropout must be in [0, 1)")
            targets = settings.get("lora_targets", DEFAULT_LORA_TARGETS)
            if not isinstance(targets, (list, tuple)) or not targets or any(
                not isinstance(target, str) or not target for target in targets
            ) or len(set(targets)) != len(targets):
                raise ValueError("lora_targets must be a nonempty list of unique module suffixes")
            linear_names = [name for name, module in backbone.named_modules() if isinstance(module, nn.Linear)]
            for target in targets:
                matches = [name for name in linear_names if name == target or name.endswith("." + target)]
                if not matches:
                    raise ValueError(f"LoRA target {target!r} matches no actual Linear module")
                self.lora_modules.extend(matches)
            self.lora_modules = sorted(set(self.lora_modules))
            self.lora_settings = {"rank": rank, "alpha": alpha, "dropout": dropout, "targets": list(targets)}
            # The pretrained model is still on CPU here; adapter initialization
            # cannot consume the projection head's random stream.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed)
                backbone = get_peft_model(
                    backbone,
                    LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                               target_modules=self.lora_modules, bias="none"),
                )
            if self.gradient_checkpointing:
                backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            if not any(parameter.requires_grad for parameter in backbone.parameters()):
                raise ValueError("LoRA failed to create trainable adapter parameters")
        self.backbone = backbone.to(device=self.device)
        self.projection_dim = self.hidden_size if self.mode == "frozen" else _positive_int(
            settings.get("projection_dim", 256), "projection_dim"
        )
        if self.mode == "frozen":
            self.head = nn.Identity()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed)
                self.head = nn.Linear(self.hidden_size, self.projection_dim, bias=False, dtype=torch.float32)
            self.head.to(device=self.device)
        self._metadata = {
            **dict(provenance or {}),
            "encoder": "ModernBERT_DNA", "mode": self.mode,
            "hidden_size": self.hidden_size, "embedding_dim": self.projection_dim,
            "sequence_mode": "full", "max_tokens": self.max_tokens,
            "automatic_truncation": False, "automatic_window_fallback": False,
            "config_max_position_embeddings": self.config_max_position_embeddings,
            "config_length_enforced": False,
            "pooling": "content_token_mean_including_UNK_excluding_CLS_SEP_PAD",
            "normalization": "L2_before_projection_and_after_projection",
            "projection": "identity" if self.mode == "frozen" else "linear_without_bias",
            "projection_seed": self.seed, "dtype": self.dtype_name,
            "attention_backend": self.attention_backend, "device": str(self.device),
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_use_reentrant": False,
            "special_token_ids": self.special_ids,
            "special_id_corrections": corrections,
            "lora": self.lora_settings, "lora_target_modules": self.lora_modules,
            "torch_version": str(torch.__version__),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
        self.train(self.mode != "frozen")

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode in {"frozen", "head"}:
            self.backbone.eval()
        return self

    def _check_length(self, count: int) -> None:
        if count < 1:
            raise ValueError("DNA has no content tokens")
        if self.max_tokens is not None and count + 2 > self.max_tokens:
            raise ValueError(
                f"Full DNA requires {count + 2} tokens, above max_tokens={self.max_tokens}; "
                "no truncation or window fallback was performed"
            )

    def tokenize(self, sequence: str) -> tuple[list[int], dict]:
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("DNA must be a nonempty uppercase sequence")
        if set(sequence) - set("ACGTRYSWKMBDHVN"):
            raise ValueError("DNA contains unsupported characters")
        encoded = self.tokenizer(
            sequence, add_special_tokens=False, truncation=False, padding=False,
            return_attention_mask=False, return_token_type_ids=False,
            # The generic tokenizer length warning assumes positional indexing;
            # ModernBERT uses RoPE. Record the exceeded config length below and
            # enforce only an explicitly requested resource cap.
            return_offsets_mapping=True, verbose=False,
        )
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if not ids or len(ids) != len(offsets):
            raise ValueError("Tokenizer returned empty or inconsistent content")
        prohibited = {self.special_ids[name] for name in ("cls", "sep", "pad")}
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is not None:
            prohibited.add(mask_id)
        end = 0
        for token_id, (start, stop) in zip(ids, offsets):
            if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < len(self.tokenizer):
                raise ValueError("Tokenizer returned an invalid vocabulary ID")
            if token_id in prohibited:
                raise ValueError("Tokenizer inserted a structural special token into DNA content")
            if start != end or not start < stop <= len(sequence):
                raise ValueError("Token offsets do not cover DNA exactly once")
            end = stop
        if end != len(sequence):
            raise ValueError("Tokenizer dropped the end of DNA")
        self._check_length(len(ids))
        unknown = [i for i, token_id in enumerate(ids) if token_id == self.special_ids["unk"]]
        return list(ids), {
            "sequence_length": len(sequence),
            "sequence_sha256": sha256(sequence.encode("ascii")).hexdigest(),
            "content_token_count": len(ids), "token_count": len(ids),
            "total_input_tokens": len(ids) + 2,
            "unknown_token_count": len(unknown),
            "unknown_base_count": sum(offsets[i][1] - offsets[i][0] for i in unknown),
            "ambiguous_base_count": sum(base not in "ACGT" for base in sequence),
            "exceeds_config_max_position_embeddings": len(ids) + 2 > self.config_max_position_embeddings,
            "sequence_mode": "full", "window_count": 1,
        }

    def base_tokens(self, ids: list[int]) -> torch.Tensor:
        self._check_length(len(ids))
        prohibited = {self.special_ids[name] for name in ("cls", "sep", "pad")}
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is not None:
            prohibited.add(mask_id)
        if any(isinstance(token_id, bool) or not isinstance(token_id, int)
               or not 0 <= token_id < len(self.tokenizer) or token_id in prohibited for token_id in ids):
            raise ValueError("Content IDs contain an invalid or structural token")
        inputs = torch.tensor(
            [[self.special_ids["cls"], *ids, self.special_ids["sep"]]],
            dtype=torch.long, device=self.device,
        )
        context = nullcontext() if self.mode == "lora" else torch.no_grad()
        with context:
            output = self.backbone(input_ids=inputs, attention_mask=torch.ones_like(inputs), return_dict=True)
            hidden = output.last_hidden_state
            if hidden.shape != (1, len(ids) + 2, self.hidden_size):
                raise ValueError("Unexpected ModernBERT hidden-state shape")
            content = hidden[0, 1:-1].float()
            if not torch.isfinite(content).all():
                raise ValueError("Non-finite ModernBERT content hidden states")
            return _unit(content.mean(dim=0), "DNA content-token mean")

    def project(self, base: torch.Tensor) -> torch.Tensor:
        if base.ndim not in {1, 2} or base.shape[-1] != self.hidden_size:
            raise ValueError("Projection expects one or a batch of normalized base vectors")
        return _unit(self.head(base.to(device=self.device, dtype=torch.float32)), "Projected DNA vector")

    def encode_tokens(self, ids: list[int]) -> torch.Tensor:
        return self.project(self.base_tokens(ids))

    def _state_identity(self) -> dict:
        keys = (
            "checkpoint_files", "config_sha256", "tokenizer_files", "tokenizer_definition_sha256", "mode", "hidden_size",
            "embedding_dim", "projection", "special_token_ids", "lora", "lora_target_modules",
        )
        return {key: self._metadata[key] for key in keys if key in self._metadata}

    def trainable_state(self) -> dict:
        adapters = {}
        if self.mode == "lora":
            from peft import get_peft_model_state_dict
            adapters = get_peft_model_state_dict(self.backbone, save_embedding_layers=False)
        return {
            "schema": "dna_encoder_trainable_v1", "identity": self._state_identity(),
            "head": {name: value.detach().cpu().clone() for name, value in self.head.state_dict().items()},
            "adapters": {name: value.detach().cpu().clone() for name, value in adapters.items()},
        }

    def load_trainable_state(self, state: dict) -> None:
        if state.get("schema") != "dna_encoder_trainable_v1" or state.get("identity") != self._state_identity():
            raise ValueError("Saved trainable state belongs to a different encoder/projection configuration")
        expected = self.trainable_state()
        for kind in ("head", "adapters"):
            received = state.get(kind)
            if not isinstance(received, dict) or received.keys() != expected[kind].keys():
                raise ValueError(f"Saved {kind} keys do not match model")
            for name, value in received.items():
                if not isinstance(value, torch.Tensor) or value.shape != expected[kind][name].shape:
                    raise ValueError(f"Saved {kind} tensor has incorrect shape: {name}")
                if not torch.isfinite(value).all():
                    raise ValueError(f"Saved {kind} tensor is non-finite: {name}")
        self.head.load_state_dict(state["head"], strict=True)
        if self.mode == "lora":
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.backbone, state["adapters"])


def load_encoder(settings: dict) -> DNAEncoder:
    """Load the complete MLM checkpoint strictly, then retain its DNA backbone."""
    from transformers import AutoConfig, AutoTokenizer, ModernBertForMaskedLM
    import transformers

    mode = settings.get("mode", "lora")
    if mode not in {"frozen", "head", "lora"}:
        raise ValueError("mode must be frozen, head or lora")
    dtype_name = settings.get("dtype", "float32")
    if dtype_name not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be float32 or bfloat16")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    backend = settings.get("attention_backend", "sdpa")
    device = torch.device(settings.get("device", "cpu"))
    if backend not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError("Unsupported attention backend")
    if backend == "flash_attention_2" and (dtype != torch.bfloat16 or device.type != "cuda"):
        raise ValueError("flash_attention_2 requires CUDA and bfloat16")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support bfloat16")
    path = Path(settings["checkpoint"]).expanduser().resolve()
    files = weight_files(path)
    tokenizer_path = Path(settings.get("tokenizer_path") or path).expanduser().resolve()
    config_path = Path(settings.get("config_path") or path).expanduser().resolve()
    config = AutoConfig.from_pretrained(str(config_path), local_files_only=True, trust_remote_code=False)
    if config.model_type != "modernbert":
        raise ValueError("Checkpoint model_type must be modernbert")
    # The pinned Transformers 5.1 implementation has no implicit torch.compile
    # path. Its legacy reference_compile option is deprecated and unused.
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with exact DNA offsets is required")
    if config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("Tokenizer PAD ID differs from checkpoint")
    loaded, info = ModernBertForMaskedLM.from_pretrained(
        str(path), config=config, local_files_only=True,
        dtype=dtype, attn_implementation=backend,
        output_loading_info=True, ignore_mismatched_sizes=False,
    )
    problems = {key: info[key] for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
    ) if info.get(key)}
    if problems:
        raise ValueError(f"Checkpoint weights do not strictly match model: {problems}")
    for name, parameter in loaded.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"Non-finite checkpoint parameter: {name}")
    config_file = config_path / "config.json" if config_path.is_dir() else config_path
    tokenizer_files = [file for file in (
        tokenizer_path / "tokenizer.json", tokenizer_path / "tokenizer_config.json",
        tokenizer_path / "special_tokens_map.json", tokenizer_path / "added_tokens.json",
        tokenizer_path / "vocab.json", tokenizer_path / "vocab.txt", tokenizer_path / "merges.txt",
    ) if file.is_file()]
    provenance = {
        "checkpoint_path": str(path),
        "checkpoint_files": [
            {"name": file.name, "size_bytes": file.stat().st_size, "sha256": file_sha256(file)}
            for file in files
        ],
        "checkpoint_loading": "strict_local_ModernBertForMaskedLM",
        "config_path": str(config_file), "config_sha256": file_sha256(config_file),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_files": [{"name": file.name, "sha256": file_sha256(file)} for file in tokenizer_files],
        "tokenizer_definition_sha256": sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest(),
        "transformers_version": transformers.__version__,
    }
    if mode == "lora":
        import peft
        provenance["peft_version"] = peft.__version__
    return DNAEncoder(loaded.model, tokenizer, settings, provenance)
