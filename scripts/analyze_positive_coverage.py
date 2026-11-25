#!/usr/bin/env python3
"""Audit product-positive coverage using existing class and cluster annotations.

Retain original author labels, including ambiguity and missingness. This script
does not refit clusters, resolve biological annotations, or split training data.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from build_manifest import PROJECT, display_path, json_text, read_tsv, sha256, write_json, write_tsv
from manifest_matrix import load_matrix


def check_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Input changed: {path}")


def read_headerless(path, columns, encoding="utf-8"):
    with path.open(encoding=encoding, newline="") as stream:
        rows = list(csv.reader(stream))
    if any(len(row) != columns or not row[0] for row in rows):
        raise ValueError(f"Unexpected CSV shape or empty ID: {path}")
    return rows


def examples(members, product_names, limit=3):
    """One name per distinct BGC first; examples are not group names."""
    selected = []
    for key in sorted(members):
        name = next((v for v in product_names[key] if v not in selected), None)
        if name:
            selected.append(name)
        if len(selected) == limit:
            break
    return selected


def subset_stats(members, adjacency):
    internal_degree = {key: len(adjacency[key] & members) for key in members}
    active = sum(bool(adjacency[key]) for key in members)
    degree_sum = sum(len(adjacency[key]) for key in members)
    internal_edges = sum(internal_degree.values()) // 2
    return {"bgc_count": len(members), "bgc_with_positive_neighbor": active,
            "positive_bgc_fraction": active / len(members) if members else 0,
            "bgc_without_positive_neighbor": len(members) - active,
            "bgc_with_internal_positive_neighbor": sum(v > 0 for v in internal_degree.values()),
            "internal_positive_pairs": internal_edges,
            "incident_positive_pairs": degree_sum - internal_edges,
            "positive_degree_sum": degree_sum}


def assignment(labels, key):
    values = labels.get(key, set())
    if not values:
        return "missing", None
    if len(values) > 1:
        return "multiple_labels", None
    label = next(iter(values))
    return ("noise", None) if label == "-1" else ("single", label)


def analyze(threshold_root, output, threshold):
    threshold_root, output = threshold_root.resolve(), output.resolve()
    manifest = PROJECT / "data/processed/benchmark_manifest_v1"
    source = PROJECT / "data/raw/BGC-clustering-benchmark"
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    for root in [PROJECT / "data/raw", manifest, threshold_root,
                 PROJECT / "data/processed/begh_content_overlap_v1"]:
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output must be separate from existing input snapshots")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")

    provenance_path = threshold_root / "provenance.json"
    previous = json.loads(provenance_path.read_text())
    for name, digest in previous["artifacts_sha256"].items():
        check_hash(threshold_root / name, digest)
    ids_path = PROJECT / previous["candidate_ids_path"]
    matrix_path = PROJECT / previous["matrix_path"]
    check_hash(ids_path, previous["candidate_ids_sha256"])
    check_hash(matrix_path, previous["matrix_sha256"])
    overlap_provenance = ids_path.parent / "provenance.json"
    check_hash(overlap_provenance, previous["overlap_provenance_sha256"])
    overlap = json.loads(overlap_provenance.read_text())
    check_hash(manifest / "provenance.json", overlap["candidate_provenance_sha256"])
    original = json.loads((manifest / "provenance.json").read_text())
    for name in ["products.tsv", "bgc_products.tsv", "bgc_manifest.tsv"]:
        check_hash(manifest / name, original["artifacts_sha256"][name])
    ids = sorted(ids_path.read_text().splitlines())
    if len(ids) != len(set(ids)) or any(not v for v in ids):
        raise ValueError("Duplicate or empty cohort ID")
    cohort = set(ids)
    matrix = load_matrix(matrix_path)
    positions = {key: i for i, key in enumerate(matrix["ids"])}
    indices = [positions[key] for key in ids]
    scores = matrix["values"][np.ix_(indices, indices)]
    left, right = np.where(np.triu(scores > threshold, k=1))
    edges = [(ids[i], ids[j], float(scores[i, j])) for i, j in zip(left, right)]
    adjacency = {key: set() for key in ids}
    for a, b, _ in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    degrees = {key: len(v) for key, v in adjacency.items()}
    assert sum(degrees.values()) == 2 * len(edges)
    previous_row = next((r for r in read_tsv(threshold_root / "threshold_statistics.tsv")
                         if r["cohort"] == "after_confirmed_overlap" and float(r["threshold"]) == threshold), None)
    if previous_row:
        assert int(previous_row["positive_pair_count"]) == len(edges)
        assert int(previous_row["bgc_count"]) == len(ids)
        for row in read_tsv(threshold_root / "positive_neighbor_counts.tsv"):
            if row["cohort"] == "after_confirmed_overlap":
                assert degrees[row["bgc_id"]] == int(row[f"positive_neighbors_gt_{threshold:.2f}"])

    products = {r["product_id"]: r for r in read_tsv(manifest / "products.tsv")}
    product_names = defaultdict(list)
    for row in read_tsv(manifest / "bgc_products.tsv"):
        name = products[row["product_id"]]["compound_name"]
        if name not in product_names[row["bgc_id"]]:
            product_names[row["bgc_id"]].append(name)

    class_path = source / "bgc_class.csv"
    class_source = read_headerless(class_path, 4, "cp1252")
    class_variants, classes = defaultdict(set), defaultdict(set)
    source_class_rows = []
    for line, (key, name, product, label) in enumerate(class_source, 1):
        labels = set(label.split(":"))
        if "" in labels:
            raise ValueError(f"Empty class label at {line}")
        classes[key].update(labels)
        class_variants[key].add(":".join(sorted(labels)))
        source_class_rows.append({"source_line": line, "bgc_id": key, "compound_name": name,
                                  "product_id": product, "raw_class": label, "in_cohort": key in cohort})
    class_rows, signature_rows = [], []
    for label in sorted(set().union(*(classes[key] for key in ids))):
        members = {key for key in ids if label in classes[key]}
        class_rows.append({"class": label, **subset_stats(members, adjacency)})
    signatures = defaultdict(set)
    for key in ids:
        signatures[":".join(sorted(classes[key])) or "__missing__"].add(key)
    for label, members in sorted(signatures.items()):
        signature_rows.append({"class_signature": label, **subset_stats(members, adjacency)})

    cluster_paths = sorted((source / "product_clusters").glob("*.csv")) + sorted((source / "bgc_clusters").glob("*.csv"))
    cluster_labels, group_rows, method_rows, source_cluster_rows = {}, [], [], []
    concentration_rows, pair_group_rows, cluster_issues = [], [], {}
    for path in cluster_paths:
        method = path.stem
        kind = "chemical_butina" if path.parent.name == "product_clusters" else "predicted_gcf"
        raw_rows = read_headerless(path, 2)
        labels = defaultdict(set)
        for line, (key, label) in enumerate(raw_rows, 1):
            if not label:
                raise ValueError(f"Empty group label: {path}:{line}")
            labels[key].add(label)
            source_cluster_rows.append({"method": method, "source_line": line,
                                        "bgc_id": key, "group_id": label, "in_cohort": key in cohort})
        cluster_labels[method] = labels
        states = {key: assignment(labels, key) for key in ids}
        valid = {key: label for key, (status, label) in states.items() if status == "single"}
        groups = defaultdict(set)
        for key, label in valid.items():
            groups[label].add(key)
        local_rows = []
        for label, members in groups.items():
            local_rows.append({"method": method, "kind": kind, "group_id": label,
                               **subset_stats(members, adjacency),
                               "product_examples_json": json_text(examples(members, product_names)),
                               "class_labels_json": json_text(sorted(set().union(*(classes[key] for key in members)))),
                               "member_ids_json": json_text(sorted(members))})
        local_rows.sort(key=lambda r: (-r["positive_degree_sum"], -r["internal_positive_pairs"], r["group_id"]))
        group_rows.extend(local_rows)
        pair_counts, raw_pair_counts, group_pairs = Counter(), Counter(), Counter()
        for a, b, _ in edges:
            relation = "unresolved" if a not in valid or b not in valid else "within" if valid[a] == valid[b] else "cross"
            pair_counts[relation] += 1
            if relation != "unresolved":
                group_pairs[tuple(sorted((valid[a], valid[b])))] += 1
            a_labels, b_labels = labels.get(a, set()) - {"-1"}, labels.get(b, set()) - {"-1"}
            raw_relation = "unmapped" if not a_labels or not b_labels else "shared_membership" if a_labels & b_labels else "disjoint_membership"
            raw_pair_counts[raw_relation] += 1
        state_counts = Counter(v[0] for v in states.values())
        method_rows.append({"method": method, "kind": kind, "source_rows": len(raw_rows),
                           "source_unique_bgcs": len(labels), "mapped_single_label_bgcs": len(valid),
                           "missing_bgcs": state_counts["missing"], "multiple_label_bgcs": state_counts["multiple_labels"],
                           "noise_bgcs": state_counts["noise"],
                           "mapped_active_bgcs": sum(degrees[key] > 0 for key in valid),
                           "current_group_count": len(groups),
                           "groups_with_multiple_bgcs": sum(len(v) >= 2 for v in groups.values()),
                           "groups_with_active_bgcs": sum(r["bgc_with_positive_neighbor"] > 0 for r in local_rows),
                           "groups_with_internal_pairs": sum(r["internal_positive_pairs"] > 0 for r in local_rows),
                           "within_positive_pairs": pair_counts["within"], "cross_positive_pairs": pair_counts["cross"],
                           "unresolved_positive_pairs": pair_counts["unresolved"],
                           "multi_membership_shared_pairs": raw_pair_counts["shared_membership"],
                           "multi_membership_disjoint_pairs": raw_pair_counts["disjoint_membership"],
                           "multi_membership_unmapped_pairs": raw_pair_counts["unmapped"]})
        cluster_issues[method] = {"multiple_labels": {k: sorted(v) for k, v in labels.items() if len(v) > 1},
                                  "source_ids_outside_matrix": sorted(set(labels) - set(matrix["ids"]))}
        for (a, b), count in sorted(group_pairs.items()):
            pair_group_rows.append({"method": method, "group_a": a, "group_b": b,
                                    "relation": "within" if a == b else "cross", "positive_pairs": count})
        for k in [1, 5, 10, 20]:
            weight = sum(r["positive_degree_sum"] for r in local_rows[:k])
            concentration_rows.append({"method": method, "top_k_by_degree_sum": k,
                                       "positive_degree_sum": weight,
                                       "fraction_of_all_positive_endpoints": weight / (2 * len(edges)) if edges else 0,
                                       "group_ids_json": json_text([r["group_id"] for r in local_rows[:k]])})
        assert sum(pair_counts.values()) == len(edges)
        assert sum(r["internal_positive_pairs"] for r in local_rows) == pair_counts["within"]
        assert sum(r["positive_degree_sum"] for r in local_rows) == sum(degrees[key] for key in valid)

    component_rows, component_for = [], {}
    unseen = set(ids)
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        members, pending = {seed}, [seed]
        while pending:
            for neighbor in adjacency[pending.pop()] & unseen:
                unseen.remove(neighbor)
                members.add(neighbor)
                pending.append(neighbor)
        component_rows.append({"component_id": seed, **subset_stats(members, adjacency),
                               "product_examples_json": json_text(examples(members, product_names)),
                               "member_ids_json": json_text(sorted(members))})
        component_for.update({key: seed for key in members})
    component_rows.sort(key=lambda r: (-r["internal_positive_pairs"], -r["bgc_count"], r["component_id"]))
    assert sum(r["internal_positive_pairs"] for r in component_rows) == len(edges)

    bgc_rows, pair_rows = [], []
    for key in ids:
        row = {"bgc_id": key, "positive_neighbors": degrees[key], "component_id": component_for[key],
               "class_labels_json": json_text(sorted(classes[key])),
               "class_source_conflict": len(class_variants[key]) > 1,
               "class_variants_json": json_text(sorted(class_variants[key])),
               "product_names_json": json_text(product_names[key])}
        for method, labels in cluster_labels.items():
            row[f"{method}_labels_json"] = json_text(sorted(labels.get(key, set())))
            row[f"{method}_status"] = assignment(labels, key)[0]
        bgc_rows.append(row)
    for a, b, score in edges:
        row = {"bgc_a": a, "bgc_b": b, "product_similarity": score,
               "class_a_json": json_text(sorted(classes[a])), "class_b_json": json_text(sorted(classes[b])),
               "class_relation": "missing" if not classes[a] or not classes[b] else "shared" if classes[a] & classes[b] else "disjoint"}
        for method, labels in cluster_labels.items():
            a_state, a_label = assignment(labels, a)
            b_state, b_label = assignment(labels, b)
            row[f"{method}_relation"] = "unresolved" if a_state != "single" or b_state != "single" else "within" if a_label == b_label else "cross"
        pair_rows.append(row)
    class_relations = Counter(r["class_relation"] for r in pair_rows)
    summary = {"bgc_count": len(ids), "threshold": threshold, "comparison": ">",
               "positive_pairs": len(edges), "bgc_with_positive_neighbor": sum(v > 0 for v in degrees.values()),
               "bgc_without_positive_neighbor": sum(v == 0 for v in degrees.values()),
               "class_positive_pair_relations": dict(class_relations),
               "class_missing_ids": [key for key in ids if not classes[key]],
               "class_conflicts_all": {key: sorted(v) for key, v in class_variants.items() if len(v) > 1},
               "class_rows": len(class_source), "class_unique_source_ids": len({r[0] for r in class_source}),
               "class_atomic_counts_overlap_for_hybrids": True,
               "class_duplicate_policy": "union; preserve raw rows and conflicting variants; no spelling or biology correction",
               "group_policy": "retain original author groups; no refit; single non-noise labels for primary statistics; preserve multiple memberships for sensitivity",
               "cluster_issues": cluster_issues,
               "component_count": len(component_rows),
               "nontrivial_component_count": sum(r["bgc_count"] > 1 for r in component_rows),
               "largest_component_bgcs": component_rows[0]["bgc_count"],
               "largest_component_positive_pairs": component_rows[0]["internal_positive_pairs"],
               "top10_component_positive_pairs": sum(r["internal_positive_pairs"] for r in component_rows[:10]),
               "group_product_examples_are_associations_not_group_names_or_maximizing_product_pairs": True,
               "curated_gcf_truth_available": False, "training_split_created": False}
    output.mkdir(parents=True)
    tables = {"class_statistics.tsv": class_rows, "class_signature_statistics.tsv": signature_rows,
              "group_statistics.tsv": group_rows, "method_coverage.tsv": method_rows,
              "group_concentration.tsv": concentration_rows, "positive_group_pairs.tsv": pair_group_rows,
              "positive_components.tsv": component_rows, "bgc_annotations.tsv": bgc_rows,
              "positive_pairs.tsv": pair_rows, "source_classes.tsv": source_class_rows,
              "source_cluster_memberships.tsv": source_cluster_rows}
    for name, rows in tables.items():
        if rows:
            write_tsv(output / name, rows, list(rows[0]))
    write_json(output / "summary.json", summary)
    input_paths = [provenance_path, ids_path, matrix_path, overlap_provenance, class_path,
                   manifest / "provenance.json", manifest / "products.tsv", manifest / "bgc_products.tsv",
                   manifest / "bgc_manifest.tsv", threshold_root / "threshold_statistics.tsv",
                   threshold_root / "positive_neighbor_counts.tsv", *cluster_paths]
    write_json(output / "provenance.json", {
        "schema": "positive_coverage_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "positive_rule": f"S > {threshold}; unordered distinct BGC pairs",
        "source_sha256": {display_path(p): sha256(p) for p in input_paths},
        "code_sha256": {display_path(p): sha256(p) for p in [Path(__file__), PROJECT / "scripts/build_manifest.py", PROJECT / "scripts/manifest_matrix.py"]},
        "numpy_version": np.__version__,
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir())}})
    print(json.dumps({"output": display_path(output), **{k: summary[k] for k in ["bgc_count", "positive_pairs", "bgc_with_positive_neighbor", "largest_component_bgcs", "largest_component_positive_pairs"]}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-root", type=Path, default=PROJECT / "data/processed/product_thresholds_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed/positive_coverage_065_v1")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    analyze(args.threshold_root, args.output, args.threshold)
