"""Strict, relocatable configuration for the three DNA experimental arms."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingConfig:
    project_root: str = str(PROJECT_ROOT)
    split_dir: str = "data/processed/dna_homology_split_v1"
    matrix_path: str = "data/raw/BGC-clustering-benchmark/tanimoto_results/NPAtlas_bm_v1.tsv"
    checkpoint: str = ""
    tokenizer_path: str | None = None
    config_path: str | None = None
    output_dir: str | None = None
    mode: str = "head"
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention_backend: str = "flash_attention_2"
    projection_dim: int = 256
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("attn.Wqkv", "attn.Wo")
    gradient_checkpointing: bool = True
    gradient_replay: bool = True
    max_tokens: int | None = None
    max_epochs: int = 20
    patience: int = 4
    min_delta: float = 0.0001
    triplets_per_update: int = 8
    head_lr: float = 0.0001
    lora_lr: float = 0.00003
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    eval_k: int = 10

    def validate(self):
        if self.mode not in {"frozen", "head", "lora"}:
            raise ValueError("mode must be frozen, head, or lora")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16; fp16 scaling is not implemented")
        if self.attention_backend not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("Unsupported attention_backend")
        if not (self.device == "cpu" or self.device == "cuda" or self.device.startswith("cuda:")):
            raise ValueError("device must be cpu, cuda, or cuda:N")
        if self.attention_backend == "flash_attention_2" and (
            self.device == "cpu" or self.dtype == "float32"
        ):
            raise ValueError("flash_attention_2 requires CUDA and bfloat16")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU execution requires float32")
        for name in ("projection_dim", "lora_rank", "lora_alpha", "max_epochs",
                     "patience", "triplets_per_update", "eval_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("head_lr", "lora_lr", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_delta", "weight_decay", "warmup_ratio", "lora_dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.warmup_ratio >= 1 or self.lora_dropout >= 1:
            raise ValueError("warmup_ratio and lora_dropout must be less than 1")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 3
        ):
            raise ValueError("max_tokens must be null or an integer >= 3; exceeded inputs fail, never truncate")
        if not self.lora_targets or not isinstance(self.lora_targets, (tuple, list)) or any(
            not isinstance(name, str) or not name for name in self.lora_targets
        ):
            raise ValueError("lora_targets must be a nonempty list of module suffixes")
        if len(set(self.lora_targets)) != len(self.lora_targets):
            raise ValueError("lora_targets must be unique")
        for name in ("gradient_checkpointing", "gradient_replay"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("project_root", "split_dir", "matrix_path", "checkpoint"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a path string")
        for name in ("output_dir", "tokenizer_path", "config_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be null or a nonempty path string")
        return self

    def path(self, name: str) -> Path:
        value = getattr(self, name)
        if name == "output_dir" and value is None:
            value = f"runs/dna_{self.mode}_s{self.seed}"
        if not value:
            raise ValueError(f"A local {name} path is required")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self.project_root).expanduser().resolve() / path
        return path.resolve()

    def model_settings(self) -> dict:
        result = asdict(self)
        for name in ("checkpoint", "tokenizer_path", "config_path"):
            result[name] = str(self.path(name)) if getattr(self, name) else None
        if result["checkpoint"] is None:
            raise ValueError("Set checkpoint to the trained local ModernBERT checkpoint directory")
        return result

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path=None, overrides=None) -> TrainingConfig:
    values = {} if path is None else json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("Training config must be a JSON object")
    unknown = set(values) - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    if "lora_targets" in values:
        if not isinstance(values["lora_targets"], (list, tuple)):
            raise ValueError("lora_targets must be a list")
        values["lora_targets"] = tuple(values["lora_targets"])
    config = TrainingConfig(**values)
    config.project_root = str(Path(config.project_root).expanduser().resolve())
    return config.validate()
