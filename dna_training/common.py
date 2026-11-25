"""Small artifact helpers; outputs are atomic and strict JSON."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_tsv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "dna_training").glob("*.py"))
    paths += [root / "scripts/train_dna_encoder.py", root / "scripts/manifest_matrix.py"]
    return {str(path.relative_to(root)): sha256(path) for path in paths if path.is_file()}


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "numpy", "biopython", "flash-attn",
                    "tokenizers", "safetensors", "accelerate", "huggingface-hub"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
