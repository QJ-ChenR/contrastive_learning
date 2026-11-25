"""Verified BGC partitions and deterministic, continuous-score triplet sampling.

This module never creates a new split or assigns a missing score to zero.  The
chemical matrix is the fixed, complete source matrix used by the split audit.
Only the caller's explicit training submatrix is given to ``TripletSampler``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.manifest_matrix import load_matrix


SPLITS = ("train", "validation", "test")
_DNA_ALPHABET = frozenset("ACGTRYSWKMBDHVN")
_REQUIRED_ARTIFACTS = (
    "bgc_split_manifest.tsv", "train_ids.txt", "validation_ids.txt", "test_ids.txt",
    "input_files.tsv", "homology_groups.tsv",
)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _unique_ids(values: Sequence[str], context: str) -> list[str]:
    ids = list(values)
    if any(not isinstance(key, str) or not key.strip() or key != key.strip() for key in ids):
        raise ValueError(f"{context}: empty or whitespace-padded BGC ID")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context}: duplicate BGC ID")
    return ids


@dataclass
class Cohort:
    project_root: Path
    ids_by_split: dict[str, list[str]]
    records: dict[str, dict[str, str]]
    fingerprint: dict
    _matrix_ids: list[str] = field(repr=False)
    _matrix: np.ndarray = field(repr=False)
    _matrix_index: dict[str, int] = field(init=False, repr=False)
    _eligible: frozenset[str] = field(init=False, repr=False)
    _sequences: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._matrix_index = {key: index for index, key in enumerate(self._matrix_ids)}
        self._eligible = frozenset(self.ids)

    @property
    def ids(self) -> list[str]:
        """Eligible IDs, in train / validation / test order."""
        return [key for split in SPLITS for key in self.ids_by_split[split]]

    def scores(self, query_ids: Sequence[str], gallery_ids: Sequence[str]) -> np.ndarray:
        """Return an ID-aligned copy, using the audit's upper-triangle convention."""
        query_ids = _unique_ids(query_ids, "Query IDs")
        gallery_ids = _unique_ids(gallery_ids, "Gallery IDs")
        if not set(query_ids + gallery_ids) <= self._eligible:
            raise ValueError("Scores requested for an excluded or unknown BGC")
        queries = np.asarray([self._matrix_index[key] for key in query_ids], dtype=int)[:, None]
        gallery = np.asarray([self._matrix_index[key] for key in gallery_ids], dtype=int)[None, :]
        # load_matrix permits tiny numerical asymmetry.  Preserve the exact
        # convention used by the existing split/positive-neighbor diagnostics.
        return self._matrix[np.minimum(queries, gallery), np.maximum(queries, gallery)].copy()

    def sequence(self, bgc_id: str) -> str:
        """Read a full original-orientation BGC; verify bytes and parsed DNA once."""
        if bgc_id not in self._eligible:
            raise ValueError(f"Sequence requested for an excluded or unknown BGC: {bgc_id}")
        if bgc_id in self._sequences:
            return self._sequences[bgc_id]
        row = self.records[bgc_id]
        path = _resolve(self.project_root, row["gbk_path"])
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != row["gbk_sha256"]:
            raise ValueError(f"GenBank file hash changed for {bgc_id}: {path}")
        from io import StringIO
        from Bio import SeqIO

        parsed = list(SeqIO.parse(StringIO(raw.decode("utf-8")), "genbank"))
        if len(parsed) != 1:
            raise ValueError(f"Expected exactly one GenBank record for {bgc_id}")
        record = parsed[0]
        if record.id != row["record_id"]:
            raise ValueError(f"GenBank record identity changed for {bgc_id}")
        sequence = str(record.seq).upper()
        if not sequence or set(sequence) - _DNA_ALPHABET:
            raise ValueError(f"Empty or unsupported DNA for {bgc_id}")
        if len(sequence) != int(row["dna_length_bp"]):
            raise ValueError(f"DNA length changed for {bgc_id}")
        if sha256(sequence.encode("ascii")).hexdigest() != row["dna_sha256"]:
            raise ValueError(f"DNA sequence hash changed for {bgc_id}")
        self._sequences[bgc_id] = sequence
        return sequence


def load_cohort(
    project_root: str | Path, split_dir: str | Path, matrix_path: str | Path,
) -> Cohort:
    """Verify the existing snapshot without accessing external search inputs.

    Relative paths resolve against ``project_root``, so copying the project tree
    to the server preserves the manifest.  The fingerprint contains content
    hashes, not absolute paths, and remains equal after such relocation.
    """
    root = Path(project_root).resolve()
    split_dir = _resolve(root, split_dir)
    matrix_path = _resolve(root, matrix_path)
    provenance_path = split_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema") != "dna_homology_split_v1":
        raise ValueError("Expected dna_homology_split_v1 provenance")
    expected_hashes = provenance.get("artifacts_sha256", {})
    checked = {}
    for name in _REQUIRED_ARTIFACTS:
        expected = expected_hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Split provenance lacks a valid hash for {name}")
        actual = file_sha256(split_dir / name)
        if actual != expected:
            raise ValueError(f"Split artifact hash changed: {name}")
        checked[name] = actual

    rows = _read_tsv(split_dir / "bgc_split_manifest.tsv")
    _unique_ids([row["bgc_id"] for row in rows], "Split manifest")
    records = {row["bgc_id"]: row for row in rows}
    ids_by_split = {}
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    groups = _read_tsv(split_dir / "homology_groups.tsv")
    group_ids = [row["group_id"] for row in groups]
    if any(not key for key in group_ids) or len(set(group_ids)) != len(group_ids):
        raise ValueError("Empty or duplicate homology group ID")
    groups_by_id = {row["group_id"]: row for row in groups}
    for row in rows:
        assignment = row["split_status"]
        if assignment not in {*SPLITS, "excluded", "excluded_missing_dna"}:
            raise ValueError(f"Unknown split assignment for {row['bgc_id']}: {assignment}")
        group = row.get("homology_group_id", "")
        if group:
            if group in group_splits and group_splits[group] != assignment:
                raise ValueError(f"Homology group crosses partitions: {group}")
            group_splits[group] = assignment
            if group not in groups_by_id or groups_by_id[group]["split"] != assignment:
                raise ValueError(f"Homology group assignment disagrees with manifest: {group}")

    for split in SPLITS:
        ids = _unique_ids((split_dir / f"{split}_ids.txt").read_text().splitlines(), f"{split} IDs")
        if not ids:
            raise ValueError(f"Empty {split} partition")
        if seen_ids & set(ids):
            raise ValueError("BGC IDs cross partitions")
        seen_ids.update(ids)
        expected = {row["bgc_id"] for row in rows if row["split_status"] == split}
        if set(ids) != expected:
            raise ValueError(f"{split} ID list disagrees with split manifest")
        for key in ids:
            row = records[key]
            if row["dna_status"] != "available":
                raise ValueError(f"Unavailable DNA assigned to {split}: {key}")
            expected_eligibility = "eligible_train" if split == "train" else "not_training"
            if row["training_eligibility"] != expected_eligibility:
                raise ValueError(f"Invalid training eligibility for {key}")
            if int(row["product_count"]) < 1 or not json.loads(row["product_ids_json"]):
                raise ValueError(f"No valid product mapping for {key}")
            if json.loads(row["exclusion_reasons_json"]):
                raise ValueError(f"Excluded BGC entered {split}: {key}")
            group = row["homology_group_id"]
            if not group:
                raise ValueError(f"Missing homology group for {key}")
            group_row = groups_by_id[group]
            if (int(group_row["external_count"]) > 0
                    or json.loads(group_row["quarantine_ids_json"])
                    or json.loads(group_row["exclusion_reasons_json"])):
                raise ValueError(f"External or quarantined group entered {split}: {group}")
            if key not in json.loads(group_row["candidate_ids_json"]):
                raise ValueError(f"BGC missing from its homology group: {key}")
            for hash_name in ("gbk_sha256", "dna_sha256"):
                digest = row[hash_name]
                if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
                    raise ValueError(f"Invalid {hash_name} for {key}")
            if int(row["dna_length_bp"]) <= 0 or not row["gbk_path"] or not row["record_id"]:
                raise ValueError(f"Incomplete sequence metadata for {key}")
        ids_by_split[split] = ids

    inputs = _read_tsv(split_dir / "input_files.tsv")
    # Matching the parent directory as well as the filename distinguishes the
    # product mapping source_data/NPAtlas_bm_v1.tsv from the similarity matrix.
    candidates = [row for row in inputs if (
        _resolve(root, row["path"]) == matrix_path
        or Path(row["path"]).parts[-2:] == matrix_path.parts[-2:]
    )]
    if len(candidates) != 1:
        raise ValueError("Cannot identify one matrix source in split input_files.tsv")
    matrix_hash = file_sha256(matrix_path)
    if matrix_hash != candidates[0]["sha256"]:
        raise ValueError("Chemical matrix hash differs from the split's source matrix")
    matrix = load_matrix(matrix_path)
    missing = seen_ids - set(matrix["ids"])
    if missing:
        raise ValueError(f"Eligible BGC IDs missing from chemical matrix: {sorted(missing)}")
    fingerprint = {
        "schema": "dna_training_cohort_v1",
        "split_schema": provenance["schema"],
        "split_provenance_sha256": file_sha256(provenance_path),
        "split_artifacts_sha256": checked,
        "matrix_sha256": matrix_hash,
        "score_convention": "source_matrix_upper_triangle",
        "split_counts": {split: len(ids) for split, ids in ids_by_split.items()},
    }
    return Cohort(root, ids_by_split, records, fingerprint, matrix["ids"], matrix["values"])


class TripletSampler:
    """One triplet per training anchor per epoch, with continuous signed gaps.

    Shuffle anchors once per epoch. Alternating anchors use (a) one top-k
    candidate plus a uniformly chosen different candidate, or (b) candidates
    from two distinct rank-quartile buckets. With an odd number of anchors, (a)
    receives one extra anchor. Candidate orientation is randomized. Ties are
    retained as zero-gap targets; there is no positive/negative cutoff.
    """

    def __init__(self, ids: Sequence[str], similarities: np.ndarray, seed: int, top_k: int = 10):
        self.ids = _unique_ids(ids, "Sampler training IDs")
        if len(self.ids) < 3:
            raise ValueError("Triplet sampling requires at least three training BGCs")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Sampler seed must be a nonnegative integer")
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError("Sampler top_k must be a positive integer")
        self.seed = int(seed)
        self.top_k = int(top_k)
        scores = np.asarray(similarities, dtype=np.float64)
        n = len(self.ids)
        if scores.shape != (n, n):
            raise ValueError("Training similarity matrix must align exactly to sampler IDs")
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Training scores must be known, finite values in [0, 1]")
        if not np.allclose(scores, scores.T, rtol=0, atol=1e-12):
            raise ValueError("Training similarity matrix must be symmetric")
        if not np.allclose(np.diag(scores), 1, rtol=0, atol=1e-12):
            raise ValueError("Training similarity matrix must have unit diagonal")
        self.similarities = scores.copy()
        self._candidate_indices = [np.delete(np.arange(n, dtype=int), anchor) for anchor in range(n)]

    @property
    def metadata(self) -> dict:
        return {
            "schema": "continuous_triplet_sampler_v2", "seed": self.seed,
            "anchor_count": len(self.ids), "triplets_per_epoch": len(self.ids),
            "anchor_policy": "each_training_anchor_once_in_shuffled_order",
            "candidate_policy": "alternating_top_k_plus_uniform_and_distinct_rank_quartiles",
            "top_k": self.top_k, "effective_top_k": min(self.top_k, len(self.ids) - 1),
            "rank_tie_break": "seeded_per_epoch_permutation_then_stable_score_sort",
            "candidate_orientation": "randomized",
            "target": "S(anchor,candidate1)-S(anchor,candidate2)",
            "target_scale": 1.0, "zero_gap_policy": "retain", "binary_cutoff": None,
        }

    def epoch(self, epoch: int) -> list[tuple[str, str, str, float]]:
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch)]))
        result = []
        for position, anchor in enumerate(rng.permutation(len(self.ids))):
            # Randomize exact-score ties reproducibly each epoch, so top-k and
            # rank buckets do not repeatedly favor a fixed lexical ID subset.
            candidates = rng.permutation(self._candidate_indices[int(anchor)])
            candidates = candidates[np.argsort(self.similarities[anchor, candidates], kind="stable")]
            if position % 2 == 0:
                top_count = min(self.top_k, len(candidates))
                first = int(rng.choice(candidates[-top_count:]))
                second = int(rng.choice(candidates[candidates != first]))
            else:
                quartiles = [bucket for bucket in np.array_split(candidates, 4) if len(bucket)]
                bins = rng.choice(len(quartiles), size=2, replace=False)
                first = int(rng.choice(quartiles[int(bins[0])]))
                second = int(rng.choice(quartiles[int(bins[1])]))
            if rng.random() < 0.5:
                first, second = second, first
            target = float(self.similarities[anchor, first] - self.similarities[anchor, second])
            result.append((self.ids[int(anchor)], self.ids[first], self.ids[second], target))
        return result
