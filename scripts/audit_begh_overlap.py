#!/usr/bin/env python3
"""Audit current candidate DNA against the actual B/E/G/H evaluation inputs."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import re

from Bio import SeqIO
from Bio.Seq import Seq

from build_manifest import PROJECT, display_path, read_tsv, sha256, validate, write_json, write_tsv
from manifest_matrix import load_matrix, summarize_pairs


DATASETS = {"Dataset_B_Divergent": 39, "Dataset_E_MIBiG_BS1": 281,
            "Dataset_G_Planomonospora": 11, "Dataset_H_Reclassification": 90}


def content(path):
    records = list(SeqIO.parse(StringIO(path.read_text(encoding="utf-8")), "genbank"))
    if len(records) != 1:
        raise ValueError(f"Expected one complete DNA record: {path}")
    record = records[0]
    sequence = str(record.seq).upper()
    if not sequence or set(sequence) - set("ACGTRYSWKMBDHVN"):
        raise ValueError(f"Invalid DNA content: {path}")
    proteins, missing = Counter(), 0
    for feature in record.features:
        if feature.type != "CDS":
            continue
        translations = feature.qualifiers.get("translation", [])
        if len(translations) != 1:
            missing += 1
            continue
        protein = "".join(translations[0].split()).rstrip("*")
        if not protein:
            missing += 1
        else:
            proteins[protein] += 1
    return {"sequence": sequence, "rc": str(Seq(sequence).reverse_complement()),
            "proteins": proteins, "missing_translations": missing}


def dna_match(candidate, evaluation, evaluation_rc):
    """Exact full equality/containment; coordinates use the original strands."""
    for strand, oriented in [("+", evaluation), ("-", evaluation_rc)]:
        if len(candidate) <= len(oriented):
            pos = oriented.find(candidate)
            if pos < 0:
                continue
            start = pos if strand == "+" else len(evaluation) - pos - len(candidate)
            return {"relationship": "exact_full_sequence" if len(candidate) == len(oriented) else "candidate_fully_contained_in_evaluation",
                    "evaluation_strand": strand, "candidate_start": 0, "candidate_end": len(candidate),
                    "evaluation_start": start, "evaluation_end": start + len(candidate)}
        pos = candidate.find(oriented)
        if pos >= 0:
            return {"relationship": "evaluation_fully_contained_in_candidate", "evaluation_strand": strand,
                    "candidate_start": pos, "candidate_end": pos + len(oriented),
                    "evaluation_start": 0, "evaluation_end": len(evaluation)}
    return None


def audit(manifest_root, evaluation_root, output):
    manifest_root, evaluation_root, output = [path.resolve() for path in (manifest_root, evaluation_root, output)]
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    for protected in [manifest_root, evaluation_root, PROJECT / "data/raw"]:
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(f"Output overlaps protected inputs: {protected}")
    inventory_validation = validate(manifest_root)
    candidate_rows = [row for row in read_tsv(manifest_root / "bgc_manifest.tsv") if row["dna_status"] == "available"]
    candidates = {}
    for row in candidate_rows:
        path = Path(row["gbk_path"])
        path = path if path.is_absolute() else PROJECT / path
        if sha256(path) != row["gbk_sha256"]:
            raise ValueError(f"Candidate file changed: {path}")
        candidates[row["bgc_id"]] = {**content(path), "row": row, "path": path}
    evaluations, evaluation_rows, input_rows = {}, [], []
    for name in ["pilot_be_v1", "pilot_gh_reconstructed_v1"]:
        root = evaluation_root / "data/processed" / name
        cohort = root / "cohort.tsv"
        input_rows.append({"path": str(cohort), "sha256": sha256(cohort), "role": "evaluation_cohort"})
        for row in read_tsv(cohort):
            if row["dataset"] not in DATASETS:
                raise ValueError(f"Unexpected evaluation dataset: {row['dataset']}")
            key = row["sample_id"]
            if key in evaluations:
                raise ValueError(f"Repeated evaluation sample ID: {key}")
            path = root / row["dataset"] / "inputs" / (key + ".gbk")
            if sha256(path) != row["source_sha256"]:
                raise ValueError(f"Prepared evaluation hash mismatch: {path}")
            raw_path = evaluation_root / "data/raw/curated_datasets" / row["archive_relpath"]
            if sha256(raw_path) != row.get("raw_source_sha256", row["source_sha256"]):
                raise ValueError(f"Raw evaluation hash mismatch: {raw_path}")
            prepared = content(path)
            if prepared["sequence"] != content(raw_path)["sequence"]:
                raise ValueError(f"Raw/prepared DNA differ: {key}")
            evaluations[key] = {**prepared, "row": row}
            evaluation_rows.append({"dataset": row["dataset"], "sample_id": key,
                                    "original_bgc_id": row["original_bgc_id"], "prepared_path": str(path),
                                    "prepared_sha256": sha256(path), "raw_path": str(raw_path),
                                    "raw_sha256": sha256(raw_path), "dna_length_bp": len(prepared["sequence"]),
                                    "raw_prepared_dna_identical": True})
    observed = dict(Counter(row["dataset"] for row in evaluation_rows))
    if observed != DATASETS:
        raise ValueError(f"Evaluation cohort coverage changed: {observed}")

    matches, protein_rows, id_rows = [], [], []
    for number, (bgc_id, candidate) in enumerate(candidates.items(), 1):
        for sample_id, evaluation in evaluations.items():
            test_row = evaluation["row"]
            base = {"bgc_id": bgc_id, "dataset": test_row["dataset"], "sample_id": sample_id,
                    "original_bgc_id": test_row["original_bgc_id"]}
            match = dna_match(candidate["sequence"], evaluation["sequence"], evaluation["rc"])
            if match:
                # Independent coordinate assertion for every reported match.
                left = candidate["sequence"][match["candidate_start"]:match["candidate_end"]]
                right = evaluation["sequence"][match["evaluation_start"]:match["evaluation_end"]]
                if match["evaluation_strand"] == "-":
                    right = str(Seq(right).reverse_complement())
                if left != right:
                    raise ValueError("Internal DNA coordinate verification failed")
                matches.append({**base, **match, "candidate_bp": len(candidate["sequence"]),
                                "evaluation_bp": len(evaluation["sequence"]),
                                "candidate_file_sha256": candidate["row"]["gbk_sha256"],
                                "evaluation_file_sha256": test_row["source_sha256"]})
            if bgc_id in re.findall(r"BGC\d{7}", test_row["original_bgc_id"]):
                id_rows.append({**base, "dna_content_match": bool(match)})
            a, b = candidate["proteins"], evaluation["proteins"]
            common = sum((a & b).values())
            ca, cb = sum(a.values()), sum(b.values())
            full = bool(a) and a == b and candidate["missing_translations"] == 0 and evaluation["missing_translations"] == 0
            if full or (common >= 3 and (common / ca >= 0.8 or common / cb >= 0.8)):
                protein_rows.append({**base, "shared_exact_cds_translation_count": common,
                                     "candidate_translated_cds": ca, "evaluation_translated_cds": cb,
                                     "candidate_exact_cds_fraction": common / ca,
                                     "evaluation_exact_cds_fraction": common / cb,
                                     "candidate_untranslated_cds": candidate["missing_translations"],
                                     "evaluation_untranslated_cds": evaluation["missing_translations"],
                                     "complete_protein_multiset_equal": full, "dna_content_match": bool(match)})
        if number % 200 == 0:
            print(f"Compared {number}/{len(candidates)} current candidates with all 421 evaluation records", flush=True)

    matched_ids = {row["bgc_id"] for row in matches}
    full_ids = {row["bgc_id"] for row in matches if row["relationship"] == "exact_full_sequence"}
    protein_only = {row["bgc_id"] for row in protein_rows if row["complete_protein_multiset_equal"]} - matched_ids
    review_ids = {row["bgc_id"] for row in protein_rows} - matched_ids - protein_only
    id_only = {row["bgc_id"] for row in id_rows} - matched_ids - protein_only
    old_ids = {row["bgc_id"] for row in candidate_rows if "historical_confirmed_id_match" in row["historical_overlap_status"]}
    # Complete protein equality is a separate evidence category; never count
    # partial CDS sharing as proof of DNA duplication.
    excluded = matched_ids | protein_only
    remaining = sorted(set(candidates) - excluded)
    by_candidate = defaultdict(list)
    for row in matches:
        by_candidate[row["bgc_id"]].append(row)
    candidate_status = []
    for bgc_id in sorted(candidates):
        candidate_status.append({"bgc_id": bgc_id, "dna_overlap_confirmed": bgc_id in matched_ids,
                                 "complete_protein_only_overlap": bgc_id in protein_only,
                                 "partial_protein_review": bgc_id in review_ids,
                                 "id_only_review": bgc_id in id_only,
                                 "old_confirmed_id_match": bgc_id in old_ids,
                                 "exact_screen_status": "exclude_content_overlap" if bgc_id in excluded else "remaining_after_exact_screen",
                                 "matched_evaluation_ids_json": json.dumps(sorted({r["sample_id"] for r in by_candidate[bgc_id]})),
                                 "matched_datasets_json": json.dumps(sorted({r["dataset"] for r in by_candidate[bgc_id]})),
                                 "near_homology_status": "not_checked"})
    summary = {"available_candidate_count": len(candidates), "evaluation_sample_count": len(evaluations),
               "evaluation_counts": observed, "dna_match_pair_count": len(matches),
               "full_dna_equal_candidate_count": len(full_ids),
               "strict_containment_additional_candidate_count": len(matched_ids - full_ids),
               "dna_overlap_candidate_count": len(matched_ids), "complete_protein_only_ids": sorted(protein_only),
               "excluded_content_candidate_count": len(excluded), "remaining_after_exact_screen": len(remaining),
               "remaining_if_review_ids_quarantined": len(set(remaining) - review_ids - id_only),
               "partial_protein_review_ids": sorted(review_ids), "id_only_review_ids": sorted(id_only),
               "old_confirmed_id_count": len(old_ids), "new_content_ids_beyond_old_list": sorted(excluded - old_ids),
               "old_ids_without_current_content_match": sorted(old_ids - excluded),
               "per_dataset": {dataset: {"evaluation_samples": DATASETS[dataset],
                                         "matched_evaluation_samples": len({r["sample_id"] for r in matches if r["dataset"] == dataset}),
                                         "matched_candidate_bgcs": len({r["bgc_id"] for r in matches if r["dataset"] == dataset})}
                               for dataset in DATASETS},
               "dna_policy": "Literal uppercase equality or full containment, both orientations, N is not a wildcard; all reported coordinates verified",
               "protein_policy": "Existing CDS translation multiset including multiplicity; no missing translations for complete equality. Partial: >=3 shared exact translations and >=80% coverage on either side, review only.",
               "near_homology_screen_completed": False, "files_deleted": False, "training_split_created": False}
    matrix = load_matrix(PROJECT / "data/raw/BGC-clustering-benchmark/tanimoto_results/NPAtlas_bm_v1.tsv")
    pair_stats = summarize_pairs(matrix, [i for i, key in enumerate(matrix["ids"]) if key in remaining])["summary"]
    # Recheck every file used by this audit before writing derived evidence.
    for row in evaluation_rows:
        for kind in ["prepared", "raw"]:
            if sha256(Path(row[kind + "_path"])) != row[kind + "_sha256"]:
                raise ValueError(f"Evaluation input changed during audit: {row[kind + '_path']}")
    for value in candidates.values():
        if sha256(value["path"]) != value["row"]["gbk_sha256"]:
            raise ValueError("Candidate changed during audit")
    output.mkdir(parents=True)
    tables = {"evaluation_inputs.tsv": evaluation_rows, "candidate_status.tsv": candidate_status,
              "dna_matches.tsv": matches, "protein_review.tsv": protein_rows, "id_matches.tsv": id_rows,
              "remaining_pair_statistics.tsv": pair_stats, "cohort_sources.tsv": input_rows}
    for name, rows in tables.items():
        if rows:
            write_tsv(output / name, rows, list(rows[0]))
    (output / "remaining_after_exact_screen.txt").write_text("".join(key + "\n" for key in remaining))
    (output / "excluded_content_ids.txt").write_text("".join(key + "\n" for key in sorted(excluded)))
    write_json(output / "summary.json", summary)
    metadata = {"schema": "begh_content_overlap_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_manifest": display_path(manifest_root),
                "candidate_provenance_sha256": sha256(manifest_root / "provenance.json"),
                "candidate_validation": inventory_validation, "evaluation_root": str(evaluation_root),
                "script_sha256": sha256(Path(__file__)),
                "artifact_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())}}
    write_json(output / "provenance.json", metadata)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data/processed/benchmark_manifest_v1")
    parser.add_argument("--evaluation-project", type=Path, default=PROJECT.parent / "BGC_retrieve")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed/begh_content_overlap_v1")
    args = parser.parse_args()
    audit(args.manifest, args.evaluation_project, args.output)
