#!/usr/bin/env python3
"""Summarize product-pair thresholds after the verified B/E/G/H overlap screen."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from build_manifest import PROJECT, display_path, read_tsv, sha256, write_json, write_tsv
from manifest_matrix import load_matrix, summarize_pairs


def summarize(overlap_root, output):
    overlap_root, output = overlap_root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    for protected in [overlap_root, PROJECT / "data/raw", PROJECT / "data/processed/benchmark_manifest_v1"]:
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Output must be separate from existing input snapshots")
    overlap_provenance = json.loads((overlap_root / "provenance.json").read_text())
    for filename, expected in overlap_provenance["artifact_sha256"].items():
        if sha256(overlap_root / filename) != expected:
            raise ValueError(f"Overlap audit artifact changed: {filename}")
    manifest = Path(overlap_provenance["candidate_manifest"])
    manifest = manifest if manifest.is_absolute() else PROJECT / manifest
    if sha256(manifest / "provenance.json") != overlap_provenance["candidate_provenance_sha256"]:
        raise ValueError("Candidate manifest provenance changed")
    manifest_provenance = json.loads((manifest / "provenance.json").read_text())
    if sha256(manifest / "input_files.tsv") != manifest_provenance["artifacts_sha256"]["input_files.tsv"]:
        raise ValueError("Candidate input inventory changed")
    matrix_source = next(row for row in read_tsv(manifest / "input_files.tsv") if row["role"] == "matrix")
    matrix_path = Path(matrix_source["path"])
    matrix_path = matrix_path if matrix_path.is_absolute() else PROJECT / matrix_path
    if sha256(matrix_path) != matrix_source["sha256"]:
        raise ValueError("Product similarity matrix changed")
    ids_path = overlap_root / "remaining_after_exact_screen.txt"
    ids = ids_path.read_text().splitlines()
    if len(ids) != len(set(ids)) or any(not key for key in ids):
        raise ValueError("Duplicate or empty candidate IDs")
    matrix = load_matrix(matrix_path)
    if set(ids) - set(matrix["ids"]):
        raise ValueError("Candidate IDs missing from the product matrix")
    summary = json.loads((overlap_root / "summary.json").read_text())
    if len(ids) != summary["remaining_after_exact_screen"]:
        raise ValueError("Candidate count disagrees with overlap audit")
    review_ids = set(summary["partial_protein_review_ids"]) | set(summary["id_only_review_ids"])
    thresholds = [i / 20 for i in range(21)]
    tables, degree_rows, equality_rows = [], [], []
    cohorts = {"after_confirmed_overlap": set(ids), "also_quarantine_review": set(ids) - review_ids}
    for cohort, members in cohorts.items():
        indices = [i for i, key in enumerate(matrix["ids"]) if key in members]
        result = summarize_pairs(matrix, indices, thresholds)
        submatrix = matrix["values"][np.ix_(indices, indices)]
        scores = submatrix[np.triu_indices(len(indices), k=1)]
        for row in result["summary"]:
            tables.append({"cohort": cohort, **row,
                           "positive_pair_fraction": row["positive_pair_count"] / row["unordered_pair_count"] if row["unordered_pair_count"] else 0,
                           "positive_bgc_fraction": row["bgc_with_positive_neighbor"] / row["bgc_count"] if row["bgc_count"] else 0,
                           "pairs_equal_to_threshold": int(np.count_nonzero(scores == row["threshold"]))})
        for bgc_id, degrees in result["degrees"].items():
            degree_rows.append({"cohort": cohort, "bgc_id": bgc_id,
                                **{f"positive_neighbors_gt_{threshold:.2f}": degrees[str(threshold)] for threshold in thresholds}})
        equality_rows.append({"cohort": cohort, "bgc_count": len(members),
                              "score_exactly_one_pair_count": int(np.count_nonzero(scores == 1))})
        for row in result["summary"]:
            assert sum(degrees[str(row["threshold"])] for degrees in result["degrees"].values()) == 2 * row["positive_pair_count"]
        assert all(a["positive_pair_count"] >= b["positive_pair_count"]
                   for a, b in zip(result["summary"], result["summary"][1:]))
    output.mkdir(parents=True)
    for name, rows in [("threshold_statistics.tsv", tables), ("positive_neighbor_counts.tsv", degree_rows),
                       ("score_one_statistics.tsv", equality_rows)]:
        write_tsv(output / name, rows, list(rows[0]))
    metadata = {"schema": "product_threshold_statistics_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "positive_rule": "S > threshold; unordered nonself BGC pairs; equality is not positive",
                "thresholds": thresholds, "cohort_sizes": {name: len(members) for name, members in cohorts.items()},
                "review_ids_only_quarantined_in_sensitivity_cohort": sorted(review_ids),
                "train_validation_test_split_created": False, "near_homology_clearance_claimed": False,
                "matrix_path": display_path(matrix_path), "matrix_sha256": sha256(matrix_path),
                "candidate_ids_path": display_path(ids_path), "candidate_ids_sha256": sha256(ids_path),
                "overlap_provenance_sha256": sha256(overlap_root / "provenance.json"),
                "code_sha256": {display_path(path): sha256(path) for path in [Path(__file__), PROJECT / "scripts/manifest_matrix.py"]},
                "numpy_version": np.__version__,
                "artifacts_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())}}
    write_json(output / "provenance.json", metadata)
    print(json.dumps({"output": display_path(output), "cohort_sizes": metadata["cohort_sizes"],
                      "threshold_rows": len(tables), "exact_score_one": equality_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-root", type=Path, default=PROJECT / "data/processed/begh_content_overlap_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed/product_thresholds_v1")
    args = parser.parse_args()
    summarize(args.overlap_root, args.output)
