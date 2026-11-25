#!/usr/bin/env python3
"""Build fixed BGC holdouts from a completed DNA homology search snapshot."""

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
import gzip
import hashlib
from pathlib import Path
import platform
import tempfile

import numpy as np

from build_manifest import PROJECT, display_path, read_tsv, sha256, write_json, write_tsv
from dna_homology import aggregate_blast, BLAST_FIELDS, PAIR_FIELDS
from group_split import build_group_split, SPLITS
from manifest_matrix import load_matrix, summarize_pairs
from run_dna_homology import resolve, verified_snapshot


def exclusion_paths(nodes, edges, external, review):
    """Return shortest reproducible paths to an external or unresolved-review seed."""
    adjacency = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    sources = sorted(external) + sorted(review)
    queue = deque(sources)
    parent = {node: None for node in sources}
    source_for = {node: node for node in sources}
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in parent:
                parent[neighbor] = node
                source_for[neighbor] = source_for[node]
                queue.append(neighbor)
    rows = []
    for node in sorted(nodes & parent.keys()):
        path = [node]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        seed = source_for[node]
        rows.append({"bgc_id": node, "seed_node": seed,
                     "seed_type": "external_evaluation" if seed in external else "unresolved_review",
                     "edge_count": len(path) - 1, "path_json": json.dumps(path)})
    return rows


def chemical_diagnostics(matrix, split_by_id, group_by_id, thresholds=(0.6, 0.65, 0.7, 0.75)):
    """Read chemical scores only AFTER sequence-only assignments are finalized."""
    thresholds = tuple(float(t) for t in thresholds)
    if 0.65 not in thresholds:
        raise ValueError("Diagnostic thresholds must include the fixed primary threshold 0.65")
    index = {key: i for i, key in enumerate(matrix["ids"])}

    def score(a, b):
        # Match summarize_pairs' original-matrix upper-triangle convention,
        # including any accepted numerical asymmetry around a strict threshold.
        i, j = sorted((index[a], index[b]))
        return float(matrix["values"][i, j])

    selected = {name: sorted(key for key, label in split_by_id.items() if label == name) for name in SPLITS}
    statistics, positives, neighbors, retrieval = [], [], [], []
    for name, ids in selected.items():
        result = summarize_pairs(matrix, [index[key] for key in ids], thresholds)
        for row in result["summary"]:
            statistics.append({"split": name, **row})
        for key in ids:
            neighbors.append({"bgc_id": key, "split": name,
                              "positive_neighbors_065_within_split": result["degrees"][key]["0.65"]})
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                similarity = score(a, b)
                if similarity > 0.65:
                    positives.append({"split": name, "bgc_a": a, "bgc_b": b, "product_similarity": similarity,
                                      "group_a": group_by_id[a], "group_b": group_by_id[b],
                                      "same_homology_group": group_by_id[a] == group_by_id[b]})
    train = selected["train"]
    train_degrees = {}
    for name in SPLITS:
        counts = []
        for key in selected[name]:
            count = sum(score(key, other) > 0.65 for other in train if other != key)
            train_degrees[key] = count
            counts.append(count)
        retrieval.append({"query_split": name, "gallery_split": "train", "query_count": len(counts),
                          "gallery_count": len(train), "positive_rule": "S > 0.65; self excluded",
                          "queries_with_positive": sum(c > 0 for c in counts),
                          "queries_without_positive": sum(c == 0 for c in counts),
                          "directed_positive_query_gallery_pairs": sum(counts)})
    for row in neighbors:
        row["positive_neighbors_065_training_gallery"] = train_degrees[row["bgc_id"]]
    all_ids = sorted(key for ids in selected.values() for key in ids)
    pair_locations = Counter()
    for i, a in enumerate(all_ids):
        for b in all_ids[i + 1:]:
            if score(a, b) > 0.65:
                pair_locations[tuple(sorted((split_by_id[a], split_by_id[b])))] += 1
    cross = [{"split_a": a, "split_b": b, "positive_pair_count": count,
              "usable_for_training": a == b == "train"}
             for (a, b), count in sorted(pair_locations.items())]
    return {"statistics": statistics, "positive_pairs": positives, "neighbors": neighbors,
            "retrieval": retrieval, "positive_pair_locations": cross}


def validate(output, check_inputs=True):
    output = resolve(output)
    meta = verified_snapshot(output)
    if meta["schema"] != "dna_homology_split_v1":
        raise ValueError("Unsupported split schema")
    inputs = read_tsv(output / "input_files.tsv")
    if check_inputs:
        for row in inputs:
            if sha256(resolve(row["path"])) != row["sha256"]:
                raise ValueError(f"Changed split input: {row['path']}")
    rows = read_tsv(output / "bgc_split_manifest.tsv")
    records = {r["bgc_id"]: r for r in rows}
    if len(records) != len(rows):
        raise ValueError("Duplicate BGC assignment")
    members = read_tsv(output / "homology_members.tsv")
    node_group = {r["node_id"]: r["group_id"] for r in members}
    if len(node_group) != len(members):
        raise ValueError("Duplicate homology node")
    candidate_nodes = {r["node_id"]: r for r in members if r["role"] == "candidate"}
    if set(candidate_nodes) != {r["bgc_id"] for r in rows if r["dna_status"] == "available"}:
        raise ValueError("Available BGCs differ from grouped candidate nodes")
    for key, node in candidate_nodes.items():
        if (records[key]["homology_group_id"] != node["group_id"]
                or records[key]["split_status"] != node["split"]):
            raise ValueError("BGC assignment disagrees with its homology node")
    group_splits = defaultdict(set)
    for row in members:
        group_splits[row["group_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A homology group crosses assignments")
    for edge in read_tsv(output / "homology_edges.tsv"):
        if node_group[edge["node_a"]] != node_group[edge["node_b"]]:
            raise ValueError("A homology edge crosses groups")
    groups = read_tsv(output / "homology_groups.tsv")
    for group in groups:
        if int(group["external_count"]) and group["split"] not in ("excluded", "external_only"):
            raise ValueError("External-connected group assigned to a model partition")
        if json.loads(group["quarantine_ids_json"]) and group["split"] != "excluded":
            raise ValueError("Review-connected group assigned to a model partition")
    for name in SPLITS:
        ids = (output / (name + "_ids.txt")).read_text().splitlines()
        if ids != sorted(key for key, row in records.items() if row["split_status"] == name):
            raise ValueError("ID list disagrees with manifest")
    source_manifest = {r["bgc_id"]: r for r in read_tsv(resolve(meta["manifest"]) / "bgc_manifest.tsv")} if check_inputs else None
    if source_manifest is not None:
        if set(source_manifest) != set(records):
            raise ValueError("Candidate inventory lost records")
        for key, row in records.items():
            if row["product_ids_json"] != source_manifest[key]["product_ids_json"]:
                raise ValueError("Product list changed")
    for row in rows:
        if row["dna_status"] != "available" and row["split_status"] != "excluded_missing_dna":
            raise ValueError("Unavailable DNA assigned for training")
    positive_counts, degrees, observed_pairs = Counter(), Counter(), set()
    for pair in read_tsv(output / "positive_pairs.tsv"):
        if not (records[pair["bgc_a"]]["split_status"] == records[pair["bgc_b"]]["split_status"] == pair["split"]):
            raise ValueError("Cross-partition pair found in within-partition positive list")
        if float(pair["product_similarity"]) <= 0.65:
            raise ValueError("Non-positive pair found")
        key = tuple(sorted((pair["bgc_a"], pair["bgc_b"])))
        if key[0] == key[1] or key in observed_pairs:
            raise ValueError("Self or repeated positive pair")
        observed_pairs.add(key)
        positive_counts[pair["split"]] += 1
        degrees.update(key)
    for row in read_tsv(output / "positive_neighbors.tsv"):
        if int(row["positive_neighbors_065_within_split"]) != degrees[row["bgc_id"]]:
            raise ValueError("Positive degrees disagree with pair list")
    for row in read_tsv(output / "pair_statistics.tsv"):
        if float(row["threshold"]) == 0.65:
            name = row["split"]
            ids = [key for key, r in records.items() if r["split_status"] == name]
            if (int(row["bgc_count"]) != len(ids)
                    or int(row["positive_pair_count"]) != positive_counts[name]
                    or int(row["bgc_with_positive_neighbor"]) != sum(degrees[key] > 0 for key in ids)
                    or int(row["negative_pair_count"]) != len(ids) * (len(ids) - 1) // 2 - positive_counts[name]):
                raise ValueError("Split pair statistics are inconsistent")
    for row in read_tsv(output / "bgc_products_split.tsv"):
        if row["split_status"] != records[row["bgc_id"]]["split_status"]:
            raise ValueError("Product association crosses BGC assignment")
    return {"status": "passed", "artifact_hashes_checked": len(meta["artifacts_sha256"]),
            "input_hashes_checked": len(inputs) if check_inputs else 0,
            "bgc_count": len(rows), "node_count": len(members),
            "cross_group_homology_edges": 0, "groups_crossing_splits": 0}


def merge_hsp_sources(primary, supplement, destination, expected_ids):
    """Combine sequence evidence, documenting the limited partial-search scope.

    Only an unterminated final line of the interrupted supplement is withheld.
    Every retained HSP is subsequently checked by aggregate_blast; malformed
    complete lines are never silently discarded.
    """
    reports = []
    with destination.open("w") as writer:
        for label, path in (("complete_primary", primary), ("partial_supplement", supplement)):
            opener = gzip.open if path.suffix == ".gz" else open
            query_ids, rows, dropped = set(), 0, b""
            with opener(path, "rt") as reader:
                for line in reader:
                    if label == "partial_supplement" and not line.endswith("\n"):
                        dropped = line.encode()
                        continue
                    if not line.strip() or line.startswith("#"):
                        continue
                    writer.write(line if line.endswith("\n") else line + "\n")
                    rows += 1
                    query_ids.add(line.split("\t", 1)[0])
            reports.append({"source": label, "path": display_path(path), "sha256": sha256(path),
                            "retained_hsp_rows": rows, "reported_query_count": len(query_ids),
                            "reported_query_ids": sorted(query_ids),
                            "queries_without_reported_hits": sorted(set(expected_ids) - query_ids),
                            "withheld_unterminated_final_line_bytes": len(dropped),
                            "withheld_final_line_sha256": hashlib.sha256(dropped).hexdigest() if dropped else None})
    return reports


def build(search, output, supplement_search=None):
    search, output = resolve(search).resolve(), resolve(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    previous = verified_snapshot(search)
    if previous["schema"] != "dna_homology_search_v1" or previous["blast_fields"] != list(BLAST_FIELDS):
        raise ValueError("Unexpected search schema/BLAST format")
    manifest, overlap = resolve(previous["manifest"]), resolve(previous["overlap"])
    for protected in (search, manifest, overlap, PROJECT / "data/raw"):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Output overlaps inputs")
    verified_snapshot(manifest)
    verified_snapshot(overlap)
    for row in read_tsv(search / "input_files.tsv"):
        if sha256(resolve(row["path"])) != row["sha256"]:
            raise ValueError("Search source changed")
    nodes = read_tsv(search / "sequence_manifest.tsv")
    for row in nodes:
        if sha256(resolve(row["gbk_path"])) != row["gbk_sha256"]:
            raise ValueError(f"DNA source changed: {row['node_id']}")
    lengths = {r["node_id"]: int(r["dna_length_bp"]) for r in nodes}
    candidates = {r["node_id"] for r in nodes if r["role"] == "candidate"}
    external = {r["node_id"] for r in nodes if r["role"] == "external_evaluation"}
    protocol = json.loads((search / "protocol.json").read_text())
    supplement_paths = []
    if supplement_search is not None:
        supplement_search = resolve(supplement_search).resolve()
        if (output == supplement_search or output.is_relative_to(supplement_search)
                or supplement_search.is_relative_to(output)):
            raise ValueError("Output overlaps the supplemental search")
        supplement_meta = verified_snapshot(supplement_search)
        if (supplement_meta.get("schema") != "dna_homology_search_incomplete_v1"
                or supplement_meta.get("full_blast_complete") is not False):
            raise ValueError("Supplement must be explicitly recorded as an incomplete search")
        if sha256(search / "sequence_manifest.tsv") != sha256(supplement_search / "sequence_manifest.tsv"):
            raise ValueError("Supplement uses different sequence identities")
        if sha256(search / "sequences.fasta") != sha256(supplement_search / "sequences.fasta"):
            raise ValueError("Supplement uses different DNA")
        supplement_protocol = json.loads((supplement_search / "protocol.json").read_text())
        supplement_paths = [supplement_search / name for name in ("provenance.json", "blast_hsps.tsv", "protocol.json")]
        protocol["partial_supplement"] = {"path": display_path(supplement_search),
                                          "task": supplement_protocol["task"], "word_size": supplement_protocol["word_size"],
                                          "policy": "all complete valid HSPs merged before interval union; uneven query scope; not a second complete search"}
    old_summary = json.loads((overlap / "summary.json").read_text())
    review = set(old_summary["partial_protein_review_ids"] + old_summary["id_only_review_ids"])
    print("Aggregating BLAST alignments and building sequence-only groups", flush=True)
    aggregation_options = dict(identity_threshold=protocol["identity_threshold"],
                               min_alignment_length=protocol["min_hsp_alignment_length"], max_evalue=protocol["max_evalue"],
                               edge_coverage=protocol["shorter_sequence_coverage_threshold"], report_coverage=protocol["diagnostic_coverage_threshold"])
    if supplement_search is not None:
        with tempfile.TemporaryDirectory(prefix="bgc-combined-hsps-") as temporary:
            combined = Path(temporary) / "combined.tsv"
            source_reports = merge_hsp_sources(search / "blast_hsps.tsv.gz", supplement_search / "blast_hsps.tsv", combined, lengths)
            evidence = aggregate_blast(combined, lengths, **aggregation_options)
    else:
        evidence = aggregate_blast(search / "blast_hsps.tsv.gz", lengths, **aggregation_options)
        source_reports = [{"source": "complete_primary", "reported_query_count": evidence["statistics"]["reported_query_count"],
                           "queries_without_reported_hits": evidence["statistics"]["queries_without_reported_hits"]}]
    blast_edges = {(r["node_a"], r["node_b"]): r for r in evidence["edges"]}
    exact_edges = {(r["node_a"], r["node_b"]): r for r in read_tsv(search / "exact_edges.tsv")}
    edges = sorted(set(blast_edges) | set(exact_edges))
    result = build_group_split(candidates, external, edges, quarantine_ids=review,
                               seed=protocol["split_seed"], ratios=protocol["split_ratios"])
    assignments = result["assignments"]
    if not result["summary"]["all_splits_nonempty"]:
        raise ValueError("Not enough eligible groups for three nonempty partitions")
    # Assignments are now fixed. Product scores cannot influence the grouping or seed.
    matrix_path = PROJECT / "data/raw/BGC-clustering-benchmark/tanimoto_results/NPAtlas_bm_v1.tsv"
    matrix = load_matrix(matrix_path)
    chemistry = chemical_diagnostics(matrix, {k: r["split"] for k, r in assignments.items()}, result["node_to_group"])
    original = read_tsv(manifest / "bgc_manifest.tsv")
    group_map = {g["group_id"]: g for g in result["groups"]}
    neighbor_map = {r["bgc_id"]: r for r in chemistry["neighbors"]}
    output_rows = []
    for row in original:
        row = dict(row)
        key = row["bgc_id"]
        entry = assignments.get(key)
        row["homology_group_id"] = entry["group_id"] if entry else ""
        row["homology_status"] = "dna_80_identity_80_shorter_coverage_screened" if entry else "not_assessed_missing_dna"
        row["split_status"] = entry["split"] if entry else "excluded_missing_dna"
        row["training_eligibility"] = "eligible_train" if entry and entry["split"] == "train" else "not_training"
        row["exclusion_reasons_json"] = json.dumps(entry["exclusion_reasons"] if entry else ["missing_dna"])
        row["current_evaluation_overlap_status"] = (
            "connected_to_external_by_sequence_graph" if entry and group_map[entry["group_id"]]["external_count"]
            else "no_connection_under_fixed_dna_rule" if entry else "not_checked_missing_dna")
        row["positive_neighbors_065_within_split"] = neighbor_map.get(key, {}).get("positive_neighbors_065_within_split", "")
        row["positive_neighbors_065_training_gallery"] = neighbor_map.get(key, {}).get("positive_neighbors_065_training_gallery", "")
        output_rows.append(row)
    output_map = {r["bgc_id"]: r for r in output_rows}
    edge_rows = []
    for pair in edges:
        blast, exact = blast_edges.get(pair), exact_edges.get(pair)
        edge_rows.append({"node_a": pair[0], "node_b": pair[1],
                          "evidence": "exact_and_blastn" if exact and blast else "exact_literal" if exact else "blastn",
                          "exact_relationship": exact["relationship"] if exact else "",
                          "shorter_coverage": blast["shorter_coverage"] if blast else 1.0,
                          "longer_coverage": blast["longer_coverage"] if blast else min(lengths[pair[0]], lengths[pair[1]]) / max(lengths[pair[0]], lengths[pair[1]]),
                          "min_hsp_identity": blast["min_hsp_identity"] if blast else 1.0,
                          "group_id": result["node_to_group"][pair[0]]})
    group_rows = []
    for group in result["groups"]:
        group_rows.append({**{k: v for k, v in group.items() if not isinstance(v, list)},
                           **{k + "_json": json.dumps(v) for k, v in group.items() if isinstance(v, list)}})
    member_rows = [{**r, "group_id": result["node_to_group"][r["node_id"]],
                    "split": group_map[result["node_to_group"][r["node_id"]]]["split"]} for r in nodes]
    paths = exclusion_paths(candidates, edges, external, review)
    if {r["bgc_id"] for r in paths} != {key for key, r in assignments.items() if r["split"] == "excluded"}:
        raise ValueError("Excluded assignments disagree with traceable exclusion paths")
    product_rows = [{**r, "group_id": output_map[r["bgc_id"]]["homology_group_id"],
                     "split_status": output_map[r["bgc_id"]]["split_status"]}
                    for r in read_tsv(manifest / "bgc_products.tsv")]
    old_excluded = set((overlap / "excluded_content_ids.txt").read_text().splitlines())
    now_excluded = {key for key, r in assignments.items() if r["split"] == "excluded"}
    if not old_excluded <= now_excluded:
        raise ValueError("A previously confirmed overlap was assigned to a model partition")
    summary = {**result["summary"], "missing_dna_count": len(original) - len(candidates),
               "original_candidate_count": len(original), "old_exact_excluded_count": len(old_excluded),
               "additional_excluded_after_exact_screen": len(now_excluded - old_excluded),
               "review_seed_ids": sorted(review), "exact_edge_count": len(exact_edges),
               "blast_edge_count": len(blast_edges), "largest_eligible_group": max(g["candidate_count"] for g in result["groups"] if g["split"] in SPLITS),
               "within_split_statistics": [r for r in chemistry["statistics"] if r["threshold"] == 0.65],
               "retrieval_coverage": chemistry["retrieval"], "training_started": False,
               "no_remote_protein_homology_or_scaffold_isolation_claimed": True,
               "chemical_scores_used_only_after_assignment": True}
    output.mkdir(parents=True)
    write_tsv(output / "bgc_split_manifest.tsv", output_rows, list(output_rows[0]))
    tables = {
        "homology_groups.tsv": (group_rows, list(group_rows[0])),
        "homology_members.tsv": (member_rows, list(member_rows[0])),
        "homology_edges.tsv": (edge_rows, ["node_a", "node_b", "evidence", "exact_relationship", "shorter_coverage", "longer_coverage", "min_hsp_identity", "group_id"]),
        "homology_pair_diagnostics.tsv": (evidence["pairs"], PAIR_FIELDS),
        "exclusion_paths.tsv": (paths, ["bgc_id", "seed_node", "seed_type", "edge_count", "path_json"]),
        "bgc_products_split.tsv": (product_rows, list(product_rows[0])),
        "pair_statistics.tsv": (chemistry["statistics"], list(chemistry["statistics"][0])),
        "positive_pairs.tsv": (chemistry["positive_pairs"], ["split", "bgc_a", "bgc_b", "product_similarity", "group_a", "group_b", "same_homology_group"]),
        "positive_neighbors.tsv": (chemistry["neighbors"], list(chemistry["neighbors"][0])),
        "retrieval_coverage.tsv": (chemistry["retrieval"], list(chemistry["retrieval"][0])),
        "positive_pair_locations.tsv": (chemistry["positive_pair_locations"], ["split_a", "split_b", "positive_pair_count", "usable_for_training"]),
    }
    for name, (rows, fields) in tables.items():
        write_tsv(output / name, rows, fields)
    for name in (*SPLITS, "excluded", "excluded_missing_dna"):
        ids = sorted(key for key, row in output_map.items() if row["split_status"] == name)
        (output / (name + "_ids.txt")).write_text("".join(key + "\n" for key in ids))
    write_json(output / "summary.json", summary)
    write_json(output / "protocol.json", protocol)
    write_json(output / "alignment_statistics.json", evidence["statistics"])
    write_json(output / "alignment_source_coverage.json", source_reports)
    # Record partial coverage across partitions as a diagnostic, without moving samples.
    cross_partial = []
    for row in evidence["pairs"]:
        a, b = row["node_a"], row["node_b"]
        sa = group_map[result["node_to_group"][a]]["split"]
        sb = group_map[result["node_to_group"][b]]["split"]
        if sa != sb and (sa in SPLITS or sb in SPLITS):
            cross_partial.append({**row, "split_a": sa, "split_b": sb})
    write_tsv(output / "cross_split_partial_hits.tsv", cross_partial, [*PAIR_FIELDS, "split_a", "split_b"])
    stat_rows = summary["within_split_statistics"]
    report = ["# DNA 同源分组与 BGC 划分 v1", "", "状态：数据划分完成；未启动训练。", "",
              "## 固定协议", "",
              f"完整 DNA，BLASTN 双向搜索（task={protocol['task']}，word_size={protocol['word_size']}）；HSP 一致性 ≥80%、长度 ≥100、E-value ≤1e-10。BTOP 配对碱基区间分别取并集（不重复累加，不计 gap 列），覆盖较短序列 ≥80% 时连边；同长取双侧覆盖较小值。另纳入全部正反向字面相等／完整包含。",
              "", ("另合并先前中止的 blastn word11 搜索中全部完整有效 HSP。补充查询范围不均匀，并非第二次完整搜索；全量覆盖由主搜索提供。来源覆盖和末尾不完整行处理见 alignment_source_coverage.json。"
                     if supplement_search else "搜索为记录配置下的启发式检出，不声称与所有 BLAST 配置具有相同召回。"),
              "", "全部 1343 个有 DNA 的候选与 421 个 B/E/G/H 记录联合建图。连通分量整组划分；触达外部评估或待复核种子的整组隔离。这是 DNA 近同源隔离规则，不是 GCF 标签，也不保证消除所有远缘蛋白同源。",
              "", "按 BGC 数目标 80%/10%/10%，seed=42，大组优先、同大小组固定随机顺序，使用比例加权负载贪心；不读取化学分数挑选分组、seed 或比例。", "",
              "## 队列", "",
              f"原始 {len(original)} 个候选；{len(original)-len(candidates)} 个缺 DNA；原精确重合排除 {len(old_excluded)} 个。本轮联合图与待复核整组隔离共 {len(now_excluded)} 个有 DNA 的候选，比原精确筛查增加 {len(now_excluded-old_excluded)} 个。最终 {summary['eligible_candidate_count']} 个 BGC、{summary['eligible_group_count']} 个组进入划分。",
              "", "| 集合 | BGC | 同源组 | S > 0.65 正对 | 有集合内正邻居 | 无集合内正邻居 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in stat_rows:
        name = row["split"]
        report.append(f"| {name} | {row['bgc_count']} | {summary['split_group_counts'][name]} | {row['positive_pair_count']} | {row['bgc_with_positive_neighbor']} | {row['bgc_without_positive_neighbor']} |")
    report += ["", "集合内正对仅在本集合使用；只有 train 的配对用于模型训练。跨集合化学正关系允许存在，但不作为训练配对。无正邻居样本仍保留，后续训练须明确其负例池/损失角色。", "",
               "验证／测试 query 对训练 gallery 的正例覆盖见 `retrieval_coverage.tsv`。该表只描述可用监督，不代表检索成绩。", "",
               "## 文件与复现", "",
               "- `bgc_split_manifest.tsv`：保留全部原候选及全部产物 ID，增加分组、划分和排除原因。",
               "- `train_ids.txt`、`validation_ids.txt`、`test_ids.txt`：实际 BGC 名单。",
               "- `homology_edges.tsv`、`homology_groups.tsv`、`exclusion_paths.tsv`：比对边、组成员、传递隔离路径。",
               "- `positive_pairs.tsv`、`pair_statistics.tsv`、`positive_neighbors.tsv`：划分后内部正例和覆盖。",
               "- `cross_split_partial_hits.tsv`：达到 50% 但未构成隔离边的跨集合局部覆盖，保留残余相似性证据。",
               "- `provenance.json`、`input_files.tsv`：输入、代码、输出哈希与运行版本。", "",
               "```bash", f"python -B scripts/build_homology_split.py --validate {display_path(output)}", "```", "",
               "详细协议和完整运行命令见 `docs/homology-split.md`。原始清单和旧审计快照保持不变；未进行化学骨架隔离、GC/物种匹配或蛋白远缘同源分组。B/E/G/H 为既有比较集合，不称为全新盲测。", ""]
    (output / "summary.md").write_text("\n".join(report), encoding="utf-8")
    input_paths = [search / "provenance.json", *[search / name for name in previous["artifacts_sha256"]],
                   manifest / "provenance.json", manifest / "bgc_manifest.tsv", manifest / "bgc_products.tsv",
                   overlap / "provenance.json", overlap / "summary.json", overlap / "excluded_content_ids.txt", matrix_path, *supplement_paths]
    input_rows = [{"path": display_path(p), "sha256": sha256(p)} for p in sorted(set(input_paths))]
    write_tsv(output / "input_files.tsv", input_rows, ["path", "sha256"])
    code_paths = [Path(__file__), PROJECT / "scripts/dna_homology.py", PROJECT / "scripts/group_split.py",
                  PROJECT / "scripts/manifest_matrix.py", PROJECT / "scripts/run_dna_homology.py", PROJECT / "scripts/build_manifest.py"]
    meta = {"schema": "dna_homology_split_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "search": display_path(search), "manifest": display_path(manifest),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "code_sha256": {display_path(p): sha256(p) for p in code_paths},
            "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "provenance.json", meta)
    print(json.dumps({"output": display_path(output), "summary": summary, "validation": validate(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, default=PROJECT / "data/processed/dna_homology_search_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed/dna_homology_split_v1")
    parser.add_argument("--supplement-search", type=Path, help="Hash-verified interrupted search; merge all complete HSPs before grouping")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--skip-input-hashes", action="store_true")
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate(args.validate, not args.skip_input_hashes), indent=2))
    else:
        build(args.search, args.output, args.supplement_search)
