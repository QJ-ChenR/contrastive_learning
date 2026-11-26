"""Command-line entry points. Test evaluation is always an explicit command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainingConfig, load_config


def _common(parser, *, evaluation=False):
    if not evaluation:
        parser.add_argument("--config", help="JSON config; paths inside it resolve from project root")
        parser.add_argument("--mode", choices=("frozen", "head", "lora"))
        parser.add_argument("--seed", type=int)
        parser.add_argument("--output-dir", help="New run directory; existing runs require --resume")
    parser.add_argument("--project-root", help="Root used for relative dataset/checkpoint paths")
    parser.add_argument("--checkpoint", help="Local trained ModernBERT MLM checkpoint directory")
    parser.add_argument("--tokenizer-path", help="Local original tokenizer directory")
    parser.add_argument("--config-path", help="Local original ModernBERT config directory")
    parser.add_argument("--device", help="cpu, cuda or cuda:N")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--attention-backend", choices=("eager", "sdpa", "flash_attention_2"))


def _training_resources(parser):
    parser.add_argument(
        "--gradient-replay", action=argparse.BooleanOptionalAction, default=None,
        help="Recompute each DNA forward to save memory; --no-gradient-replay retains the triplet graphs",
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None,
        help="Recompute transformer layers during backward; disabling needs a separate memory probe",
    )
    parser.add_argument(
        "--checkpoint-token-budget", type=int,
        help="Keep activations for triplets within this total-token budget; 0 always uses requested checkpointing",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Full-DNA encoder training with continuous chemical similarity gaps")
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="Verify split, DNA hashes and train-only sampling; no checkpoint required")
    _common(audit_parser)
    audit_parser.add_argument("--report", help="Optional JSON report destination")
    probe_parser = commands.add_parser("probe", help="Probe train median/P90/max full-token forward/backward and save/reload")
    _common(probe_parser)
    _training_resources(probe_parser)
    probe_parser.add_argument("--report-dir", required=True)
    train_parser = commands.add_parser("train", help="Run frozen baseline, head-only, or LoRA; validation only")
    _common(train_parser)
    _training_resources(train_parser)
    train_parser.add_argument("--resume", action="store_true", help="Restore last.pt from the run directory")
    train_parser.add_argument("--stop-after-epochs", type=int, help="Stop at an epoch boundary, retaining the planned LR schedule")
    evaluate_parser = commands.add_parser("evaluate", help="Evaluate best.pt and export ID-aligned query/gallery vectors")
    _common(evaluate_parser, evaluation=True)
    evaluate_parser.add_argument("--run-dir", required=True)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    keys = ("project_root", "checkpoint", "tokenizer_path", "config_path", "device", "dtype", "attention_backend")
    overrides = {key: getattr(arguments, key) for key in keys if getattr(arguments, key) is not None}
    from .runtime import audit, evaluate_run, probe, train
    if arguments.command == "evaluate":
        result = evaluate_run(arguments.run_dir, arguments.split, overrides)
    else:
        overrides.update({key: getattr(arguments, key) for key in ("mode", "seed", "output_dir")
                          if getattr(arguments, key) is not None})
        if arguments.command in {"train", "probe"}:
            overrides.update({key: getattr(arguments, key) for key in (
                "gradient_replay", "gradient_checkpointing", "checkpoint_token_budget",
            ) if getattr(arguments, key) is not None})
        config_path = arguments.config
        if arguments.command == "train" and arguments.resume and config_path is None:
            # Recover the original protocol instead of accidentally resuming
            # with today's defaults. Explicit overrides are identity-checked.
            # Locate the saved config before validating execution options;
            # e.g. --device cpu must inherit its saved float32/eager settings.
            candidate = TrainingConfig(**overrides)
            config_path = candidate.path("output_dir") / "config.json"
        config = load_config(config_path, overrides)
        if arguments.command == "audit":
            result = audit(config)
            if arguments.report:
                from .common import write_json
                write_json(Path(arguments.report), result)
        elif arguments.command == "probe":
            result = probe(config, arguments.report_dir)
        else:
            if arguments.stop_after_epochs is not None and arguments.stop_after_epochs <= 0:
                parser.error("--stop-after-epochs must be positive")
            result = train(config, arguments.resume, arguments.stop_after_epochs)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0
