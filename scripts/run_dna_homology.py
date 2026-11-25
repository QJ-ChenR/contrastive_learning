#!/usr/bin/env python3
"""Run a versioned local BLASTN search on candidates plus reserved evaluations.

No chemical labels are used. All DNA is retained at full length and in its
original orientation. The output is complete only when provenance.json exists.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from Bio.Seq import Seq

from audit_begh_overlap import content, dna_match
from build_manifest import PROJECT, display_path, read_tsv, sha256, validate, write_json, write_tsv


BLAST_FIELDS = "qseqid sseqid qlen slen nident length qstart qend sstart send evalue bitscore gaps btop"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def verified_snapshot(root):
    metadata = json.loads((root / "provenance.json").read_text())
    hashes = metadata.get("artifacts_sha256", metadata.get("artifact_sha256", {}))
    if not hashes:
        raise ValueError(f"Missing output hashes: {root}")
    for name, expected in hashes.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Changed snapshot artifact: {root / name}")
    return metadata


def run(manifest, overlap, output, blast_bin, threads, task="megablast", word_size=16, reuse_exact=None):
    manifest, overlap, output = [resolve(p).resolve() for p in (manifest, overlap, output)]
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    for protected in (manifest, overlap, PROJECT / "data/raw"):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(f"Output overlaps an input directory: {protected}")
    if threads < 1:
        raise ValueError("threads must be positive")
    blastn, makeblastdb = [Path(blast_bin).resolve() / name for name in ("blastn", "makeblastdb")]
    versions = {p.name: subprocess.check_output([str(p), "-version"], text=True).strip()
                for p in (blastn, makeblastdb)}
    validate(manifest)
    previous = verified_snapshot(overlap)
    if sha256(manifest / "provenance.json") != previous["candidate_provenance_sha256"]:
        raise ValueError("Manifest differs from the overlap audit's source")
    rows, sequences = [], {}
    inputs = []
    for path in (manifest / "provenance.json", manifest / "bgc_manifest.tsv", overlap / "provenance.json",
                 overlap / "evaluation_inputs.tsv", overlap / "summary.json", overlap / "dna_matches.tsv"):
        inputs.append({"path": display_path(path), "sha256": sha256(path), "role": "source_snapshot"})

    def add(node, role, path, expected, bgc_id="", dataset="", sample_id=""):
        if node in sequences:
            raise ValueError(f"Duplicate node: {node}")
        path = resolve(path)
        if sha256(path) != expected:
            raise ValueError(f"Changed GBK: {path}")
        sequence = content(path)["sequence"]
        sequences[node] = sequence
        rows.append({"node_id": node, "role": role, "bgc_id": bgc_id, "dataset": dataset,
                     "sample_id": sample_id, "dna_length_bp": len(sequence),
                     "dna_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                     "gbk_path": display_path(path), "gbk_sha256": expected})

    for row in read_tsv(manifest / "bgc_manifest.tsv"):
        if row["dna_status"] == "available":
            add(row["bgc_id"], "candidate", row["gbk_path"], row["gbk_sha256"], bgc_id=row["bgc_id"])
    for row in read_tsv(overlap / "evaluation_inputs.tsv"):
        add("EVAL_" + row["sample_id"], "external_evaluation", row["prepared_path"], row["prepared_sha256"],
            dataset=row["dataset"], sample_id=row["sample_id"])
    rows.sort(key=lambda r: r["node_id"])
    output.mkdir(parents=True)
    write_tsv(output / "sequence_manifest.tsv", rows, list(rows[0]))
    write_tsv(output / "input_files.tsv", inputs, list(inputs[0]))
    fasta = output / "sequences.fasta"
    with fasta.open("w") as handle:
        for row in rows:
            handle.write(f">{row['node_id']}\n{sequences[row['node_id']]}\n")
    protocol = {
        "scope": "Operational DNA near-homology grouping; not all remote homology or GCF truth",
        "candidate_count": sum(r["role"] == "candidate" for r in rows),
        "external_count": sum(r["role"] == "external_evaluation" for r in rows),
        "identity_threshold": 0.80, "min_hsp_alignment_length": 100, "max_evalue": 1e-10,
        "shorter_sequence_coverage_threshold": 0.80, "diagnostic_coverage_threshold": 0.50,
        "coverage": "union of positions in BTOP columns with a base on both sides; both query directions combined",
        "strand": "both", "dust": "yes", "soft_masking": True, "task": task, "word_size": word_size,
        "search_sensitivity_limit": "BLAST heuristic hits under the recorded task/seed; no equivalence to exhaustive alignment claimed",
        "max_target_seqs": len(rows), "max_hsps": "unlimited (not set)",
        "exact_literal_equality_and_containment": "all unordered pairs, both orientations; N is literal",
        "split_ratios": [0.8, 0.1, 0.1], "split_seed": 42,
        "review_policy": "quarantine complete groups containing existing unresolved review IDs",
        "chemical_labels_used_to_select_split": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "protocol.json", protocol)
    print(f"Prepared {len(rows)} full-length sequences; checking literal equality/containment", flush=True)
    exact = []
    keys = sorted(sequences)
    rc = {key: str(Seq(sequence).reverse_complement()) for key, sequence in sequences.items()}
    if reuse_exact is not None:
        reuse_exact = resolve(reuse_exact)
        old = verified_snapshot(reuse_exact)
        if not (old.get("exact_scan_complete") or old.get("schema") == "dna_homology_search_v1"):
            raise ValueError("Exact scan source is not complete")
        if (sha256(fasta) != sha256(reuse_exact / "sequences.fasta")
                or sha256(output / "sequence_manifest.tsv") != sha256(reuse_exact / "sequence_manifest.tsv")):
            raise ValueError("Prepared DNA differs from the reusable exact scan")
        for row in read_tsv(reuse_exact / "exact_edges.tsv"):
            a, b = row["node_a"], row["node_b"]
            match = dna_match(sequences[a], sequences[b], rc[b])
            if match is None or any(str(value) != row[key] for key, value in match.items()):
                raise ValueError("Reused exact evidence failed sequence/coordinate validation")
            exact.append({"node_a": a, "node_b": b, **match})
        inputs.append({"path": display_path(reuse_exact / "provenance.json"),
                       "sha256": sha256(reuse_exact / "provenance.json"), "role": "reused_complete_exact_scan"})
        inputs.append({"path": display_path(reuse_exact / "exact_edges.tsv"),
                       "sha256": sha256(reuse_exact / "exact_edges.tsv"), "role": "reused_exact_edges"})
        write_tsv(output / "input_files.tsv", inputs, list(inputs[0]))
        print(f"Revalidated {len(exact)} exact edges from identical complete DNA inputs", flush=True)
    else:
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                match = dna_match(sequences[a], sequences[b], rc[b])
                if match:
                    exact.append({"node_a": a, "node_b": b, **match})
            if (i + 1) % 300 == 0:
                print(f"Exact scan: {i + 1}/{len(keys)}", flush=True)
    fields = ["node_a", "node_b", "relationship", "evaluation_strand", "candidate_start", "candidate_end",
              "evaluation_start", "evaluation_end"]
    write_tsv(output / "exact_edges.tsv", exact, fields)
    # Previously confirmed overlaps must survive the fresh all-sequence scan.
    exact_pairs = {(r["node_a"], r["node_b"]) for r in exact}
    for row in read_tsv(overlap / "dna_matches.tsv"):
        if tuple(sorted((row["bgc_id"], "EVAL_" + row["sample_id"]))) not in exact_pairs:
            raise ValueError("Prior exact overlap was not reproduced")
    del sequences, rc
    commands = []
    environment = {**os.environ, "NCBI_DONT_USE_NCBIRC": "1", "BLAST_USAGE_REPORT": "false"}
    with tempfile.TemporaryDirectory(prefix="bgc-blast-db-") as tmp:
        database = str(Path(tmp) / "sequences")
        command = [str(makeblastdb), "-in", str(fasta), "-dbtype", "nucl", "-parse_seqids",
                   "-blastdb_version", "4", "-out", database]
        commands.append(command)
        with (output / "makeblastdb.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
        raw = output / "blast_hsps.tsv"
        command = [str(blastn), "-query", str(fasta), "-db", database, "-task", task,
                   "-word_size", str(word_size), "-evalue", "1e-10", "-perc_identity", "80", "-strand", "both",
                   "-dust", "yes", "-soft_masking", "true", "-max_target_seqs", str(len(rows)),
                   "-num_threads", str(threads), "-outfmt", "6 " + BLAST_FIELDS, "-out", str(raw)]
        commands.append(command)
        write_json(output / "commands.json", commands)
        print(f"Starting local BLASTN all-v-all (task={task}, word_size={word_size}) with {threads} threads", flush=True)
        with (output / "blastn.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
    with raw.open("rb") as source, (output / "blast_hsps.tsv.gz").open("wb") as destination:
        with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0) as compressed:
            shutil.copyfileobj(source, compressed)
    raw.unlink()  # Only the newly generated, losslessly compressed scratch file.
    for row in rows:
        if sha256(resolve(row["gbk_path"])) != row["gbk_sha256"]:
            raise ValueError("GBK changed during search")
    for row in inputs:
        if sha256(resolve(row["path"])) != row["sha256"]:
            raise ValueError("Snapshot changed during search")
    metadata = {
        "schema": "dna_homology_search_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": display_path(manifest), "overlap": display_path(overlap), "versions": versions,
        "executable_sha256": {str(p): sha256(p) for p in (blastn, makeblastdb)},
        "code_sha256": {display_path(Path(__file__)): sha256(Path(__file__)),
                        display_path(PROJECT / "scripts/audit_begh_overlap.py"): sha256(PROJECT / "scripts/audit_begh_overlap.py")},
        "blast_fields": BLAST_FIELDS.split(), "threads": threads, "exact_edge_count": len(exact),
        "node_role_counts": dict(Counter(r["role"] for r in rows)),
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "provenance.json", metadata)
    print(json.dumps({"output": display_path(output), "status": "complete", "exact_edge_count": len(exact)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data/processed/benchmark_manifest_v1")
    parser.add_argument("--overlap", type=Path, default=PROJECT / "data/processed/begh_content_overlap_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed/dna_homology_search_v1")
    parser.add_argument("--blast-bin", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--task", choices=("blastn", "megablast", "dc-megablast"), default="megablast")
    parser.add_argument("--word-size", type=int, default=16)
    parser.add_argument("--reuse-exact", type=Path, help="Reuse a hash-verified complete exact scan on identical DNA")
    args = parser.parse_args()
    run(args.manifest, args.overlap, args.output, args.blast_bin, args.threads, args.task, args.word_size, args.reuse_exact)
