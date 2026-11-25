#!/usr/bin/env python3
"""Build a lossless candidate inventory; never select a training cohort."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import Bio
import numpy as np

from manifest_dna import load_dna
from manifest_matrix import load_matrix, summarize_pairs
from manifest_products import load_products, load_scaffolds


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = "bgc_candidate_manifest_v1"
ISSUE_FIELDS = ["entity_type", "entity_id", "issue_code", "severity", "evidence_json"]


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path)


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def git_info(directory):
    def git(*args):
        result = subprocess.run(["git", "-C", str(directory), *args], text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "remote": git("remote", "get-url", "origin"),
            "working_tree_status": git("status", "--porcelain", "--untracked-files=all")}


def input_entry(path, role):
    return {"path": display_path(path), "role": role, "size_bytes": path.stat().st_size,
            "sha256": sha256(path)}


def load_historical(root):
    if root is None:
        return [], []
    rows, inputs = [], []
    for filename, status in [("confirmed_overlap_mibig_ids.txt", "historical_confirmed_id_match"),
                             ("additional_review_mibig_ids.txt", "historical_review_id_match")]:
        path = root / filename
        inputs.append(input_entry(path, "historical_id_reference"))
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate historical IDs: {path}")
        for bgc_id in sorted(ids):
            rows.append({"bgc_id": bgc_id, "reference_status": status,
                         "source_path": display_path(path), "source_sha256": inputs[-1]["sha256"]})
    return rows, inputs


def issue(entity_id, code, evidence, severity="review"):
    return {"entity_type": "bgc", "entity_id": entity_id, "issue_code": code,
            "severity": severity, "evidence_json": json_text(evidence)}


def validate(output, verify_inputs=True):
    """Check persisted hashes and lossless joins, independently of build objects."""
    metadata = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    if metadata["schema"] != SCHEMA:
        raise ValueError("Unsupported manifest schema")
    for name, expected in metadata["artifacts_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    inputs = read_tsv(output / "input_files.tsv")
    if verify_inputs:
        for row in inputs:
            path = Path(row["path"])
            path = path if path.is_absolute() else PROJECT / path
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"Source checksum mismatch or missing file: {path}")
    bgcs = read_tsv(output / "bgc_manifest.tsv")
    products = read_tsv(output / "products.tsv")
    links = read_tsv(output / "bgc_products.tsv")
    scaffolds = read_tsv(output / "bgc_scaffolds.tsv")
    dna = read_tsv(output / "dna_files.tsv")
    bgc_map = {row["bgc_id"]: row for row in bgcs}
    product_map = {row["product_id"]: row for row in products}
    if len(bgc_map) != len(bgcs) or len(product_map) != len(products):
        raise ValueError("Duplicate manifest entity IDs")
    by_bgc, by_scaffold = defaultdict(list), defaultdict(list)
    for row in links:
        if row["bgc_id"] not in bgc_map or row["product_id"] not in product_map:
            raise ValueError("Dangling product association")
        if row["source_line"] != product_map[row["product_id"]]["source_line"]:
            raise ValueError("Product source-line disagreement")
        by_bgc[row["bgc_id"]].append(row["product_id"])
    for row in scaffolds:
        if row["bgc_id"] not in bgc_map:
            raise ValueError("Dangling scaffold association")
        by_scaffold[row["bgc_id"]].append(row)
    for row in bgcs:
        bgc_id = row["bgc_id"]
        if sorted(by_bgc[bgc_id]) != sorted(json.loads(row["product_ids_json"])):
            raise ValueError(f"Product list does not preserve associations: {bgc_id}")
        if int(row["product_count"]) != len(by_bgc[bgc_id]):
            raise ValueError(f"Product count mismatch: {bgc_id}")
        if int(row["scaffold_row_count"]) != len(by_scaffold[bgc_id]):
            raise ValueError(f"Scaffold count mismatch: {bgc_id}")
        if row["split_status"] != "unassigned" or row["training_eligibility"] != "not_assessed":
            raise ValueError("Inventory unexpectedly contains cohort decisions")
    for product_id, row in product_map.items():
        linked = sorted(r["bgc_id"] for r in links if r["product_id"] == product_id)
        if linked != sorted(json.loads(row["mibig_ids_json"])):
            raise ValueError(f"Incomplete product associations: {product_id}")
    matrix_count = sum(row["matrix_index"] != "" for row in bgcs)
    available = sum(row["dna_status"] == "available" for row in bgcs)
    observed_counts = {"bgc_count": len(bgcs), "matrix_bgc_count": matrix_count,
                       "dna_available_bgc_count": available, "gbk_file_count": len(dna),
                       "matrix_bgc_missing_gbk_count": sum(row["matrix_index"] != "" and row["dna_status"] == "missing_gbk" for row in bgcs),
                       "linked_product_count": len(products), "bgc_product_association_count": len(links),
                       "scaffold_row_count": len(scaffolds)}
    for name, observed in observed_counts.items():
        if metadata["counts"][name] != observed:
            raise ValueError(f"Manifest count disagrees with provenance: {name}")
    by_file = defaultdict(list)
    for row in dna:
        by_file[row["bgc_id"]].append(row)
    for row in bgcs:
        files = by_file[row["bgc_id"]]
        if int(row["gbk_file_count"]) != len(files):
            raise ValueError(f"GBK count disagrees with BGC manifest: {row['bgc_id']}")
        if sorted(json.loads(row["gbk_paths_json"])) != sorted(f["gbk_path"] for f in files):
            raise ValueError(f"GBK paths disagree with BGC manifest: {row['bgc_id']}")
    for row in read_tsv(output / "pair_statistics.tsv"):
        n = int(row["bgc_count"])
        if int(row["unordered_pair_count"]) != n * (n - 1) // 2:
            raise ValueError("Invalid unordered pair total")
        if int(row["positive_pair_count"]) + int(row["negative_pair_count"]) != int(row["unordered_pair_count"]):
            raise ValueError("Positive and negative pair counts do not sum to total")
        if row["threshold"] == "0.65":
            field = "positive_neighbors_065_" + ("full_matrix" if row["cohort"] == "full_matrix" else "available_dna")
            degrees = [int(bgc[field]) for bgc in bgcs if bgc[field] != ""]
            if len(degrees) != n or sum(degrees) != 2 * int(row["positive_pair_count"]):
                raise ValueError("Positive-neighbor degrees disagree with pair counts")
    return {"status": "passed", "artifact_hashes_checked": len(metadata["artifacts_sha256"]),
            "input_hashes_checked": len(inputs) if verify_inputs else 0,
            "bgc_count": len(bgcs), "product_count": len(products), "association_count": len(links)}


def build(source, output, historical_root=None):
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the raw source directory")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new snapshot directory: {output}")
    source_paths = {"products": source / "source_data/NPAtlas_bm_v1.tsv",
                    "matrix": source / "tanimoto_results/NPAtlas_bm_v1.tsv",
                    "scaffolds": source / "product_scaffolds/bm_scaffolds.csv"}
    inputs = [input_entry(path, role) for role, path in source_paths.items()]
    for relative in ["calculate_structural_similarity.py", "calc_product_clusters_and_scaffolds.py",
                     "README.md", "LICENSE", "source_data/LICENSE", "benchmark_bgc/LICENSE"]:
        inputs.append(input_entry(source / relative, "upstream_definition"))
    for path in sorted((PROJECT / "scripts").glob("*manifest*.py")):
        inputs.append(input_entry(path, "manifest_code"))
    products = load_products(source_paths["products"])
    scaffolds = load_scaffolds(source_paths["scaffolds"])
    matrix = load_matrix(source_paths["matrix"])
    dna = load_dna(source / "benchmark_bgc")
    historical, historical_inputs = load_historical(historical_root)
    inputs.extend(historical_inputs)
    for record in dna["records"]:
        path = Path(record["gbk_path"])
        if not record["gbk_sha256"]:
            raise ValueError(f"Cannot fingerprint unreadable GBK; restore read access before building a snapshot: {path}")
        record["gbk_path"] = display_path(path)
        inputs.append({"path": record["gbk_path"], "role": "gbk", "size_bytes": record["gbk_size_bytes"],
                       "sha256": record["gbk_sha256"]})
    links = products["associations"]
    product_map = {row["product_id"]: row for row in products["products"]}
    by_product, by_scaffold, by_dna, by_historical = (defaultdict(list) for _ in range(4))
    for row in links:
        by_product[row["bgc_id"]].append(row["product_id"])
    for row in scaffolds["rows"]:
        by_scaffold[row["bgc_id"]].append(row)
    for row in dna["records"]:
        by_dna[row["bgc_id"]].append(row)
    for row in historical:
        by_historical[row["bgc_id"]].append(row["reference_status"])
    matrix_index = {bgc_id: i for i, bgc_id in enumerate(matrix["ids"])}
    all_ids = sorted(set(matrix_index) | set(by_product) | set(by_scaffold) | set(by_dna))
    available_ids = {key for key, rows in by_dna.items() if len(rows) == 1 and rows[0]["dna_status"] == "available"}
    indices = [i for i, key in enumerate(matrix["ids"]) if key in available_ids]
    pair_all = summarize_pairs(matrix)
    pair_available = summarize_pairs(matrix, indices)
    issues = products["issues"] + scaffolds["issues"] + dna["issues"]
    bgcs = []
    dna_fields = ["gbk_path", "gbk_sha256", "record_id", "dna_length_bp", "dna_sha256",
                  "dna_rc_canonical_sha256", "gc_fraction", "n_count", "ambiguous_base_count", "organism"]
    for bgc_id in all_ids:
        codes = set()
        files = by_dna.get(bgc_id, [])
        if not files:
            dna_status = "missing_gbk"
            codes.add(dna_status)
            issues.append(issue(bgc_id, dna_status, {"expected_path": display_path(source / "benchmark_bgc" / (bgc_id + ".gbk"))}))
        elif len(files) > 1:
            dna_status = "multiple_gbk_files"
            codes.add(dna_status)
            issues.append(issue(bgc_id, dna_status, {"paths": [row["gbk_path"] for row in files]}))
        else:
            dna_status = files[0]["dna_status"]
        for row in files:
            codes.update(json.loads(row["issue_codes_json"]))
        if bgc_id not in matrix_index:
            codes.add("missing_matrix_id")
            issues.append(issue(bgc_id, "missing_matrix_id", {}))
        if not by_product[bgc_id]:
            codes.add("missing_product_association")
            issues.append(issue(bgc_id, "missing_product_association", {}))
        problem_products = []
        for product_id in by_product[bgc_id]:
            if json.loads(product_map[product_id]["issue_codes_json"]):
                problem_products.append(product_id)
        if problem_products:
            codes.add("product_metadata_review")
            issues.append(issue(bgc_id, "product_metadata_review", {"product_ids": sorted(problem_products)}))
        historical_status = ";".join(sorted(by_historical[bgc_id])) if by_historical[bgc_id] else (
            "no_id_match_in_reference" if historical_root else "not_checked")
        for status in by_historical[bgc_id]:
            codes.add(status)
            issues.append(issue(bgc_id, status, {"current_sequence_overlap_verified": False,
                                               "excluded": False}, "info"))
        srows = by_scaffold[bgc_id]
        if len(srows) != len(by_product[bgc_id]):
            codes.add("scaffold_product_count_mismatch")
            issues.append(issue(bgc_id, "scaffold_product_count_mismatch",
                                {"scaffold_rows": len(srows), "product_associations": len(by_product[bgc_id])}))
        row = {"bgc_id": bgc_id, "matrix_index": matrix_index.get(bgc_id, ""), "dna_status": dna_status,
               "gbk_file_count": len(files), "gbk_paths_json": json_text([f["gbk_path"] for f in files])}
        row.update({field: files[0].get(field, "") if len(files) == 1 else "" for field in dna_fields})
        row.update({"product_count": len(by_product[bgc_id]), "product_ids_json": json_text(sorted(by_product[bgc_id])),
                    "product_review_ids_json": json_text(sorted(problem_products)),
                    "scaffold_row_count": len(srows),
                    "unique_nonempty_scaffold_count": len({s["scaffold_smiles"] for s in srows if s["scaffold_smiles"]}),
                    "empty_scaffold_row_count": sum(not s["scaffold_smiles"] for s in srows),
                    "positive_neighbors_065_full_matrix": pair_all["degrees"].get(bgc_id, {}).get("0.65", ""),
                    "positive_neighbors_065_available_dna": pair_available["degrees"].get(bgc_id, {}).get("0.65", ""),
                    "historical_overlap_status": historical_status,
                    "current_evaluation_overlap_status": "not_checked", "homology_status": "not_checked",
                    "split_status": "unassigned", "training_eligibility": "not_assessed",
                    "issue_codes_json": json_text(sorted(codes))})
        bgcs.append(row)
    issues.sort(key=lambda row: (row["entity_type"], row["entity_id"], row["issue_code"], row["evidence_json"]))
    pair_stats = [{"cohort": name, **row} for name, data in [("full_matrix", pair_all), ("available_dna", pair_available)]
                  for row in data["summary"]]
    counts = {"bgc_count": len(bgcs), "matrix_bgc_count": len(matrix["ids"]), "gbk_file_count": len(dna["records"]),
              "dna_available_bgc_count": len(available_ids), "matrix_bgc_missing_gbk_count": sum(not by_dna.get(key) for key in matrix["ids"]),
              "linked_product_count": len(products["products"]), "bgc_product_association_count": len(links),
              "scaffold_row_count": len(scaffolds["rows"]), "issue_row_count": len(issues),
              "products_with_review_flags": sum(bool(json.loads(row["issue_codes_json"])) for row in products["products"]),
              "historical_confirmed_id_matches": sum("historical_confirmed_id_match" in row["historical_overlap_status"] for row in bgcs),
              "historical_review_id_matches": sum("historical_review_id_match" in row["historical_overlap_status"] for row in bgcs)}
    # Detect changes during reading before publishing anything; do not rewrite raw inputs.
    for entry in inputs:
        path = Path(entry["path"])
        path = path if path.is_absolute() else PROJECT / path
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Input changed during inventory: {path}")
    output.mkdir(parents=True)
    tables = {"bgc_manifest.tsv": bgcs, "products.tsv": products["products"], "bgc_products.tsv": links,
              "bgc_scaffolds.tsv": scaffolds["rows"], "dna_files.tsv": dna["records"],
              "pair_statistics.tsv": pair_stats, "input_files.tsv": inputs}
    for name, rows in tables.items():
        if not rows:
            raise ValueError(f"Unexpected empty table: {name}")
        write_tsv(output / name, rows, list(rows[0]))
    write_tsv(output / "issues.tsv", issues, ISSUE_FIELDS)
    write_tsv(output / "historical_overlap_ids.tsv", historical,
              ["bgc_id", "reference_status", "source_path", "source_sha256"])
    command = [sys.executable, display_path(Path(__file__)), "--source", display_path(source),
               "--output", display_path(output)]
    if historical_root:
        command += ["--historical-audit-root", display_path(historical_root)]
    metadata = {"schema": SCHEMA, "status": "inventory_complete_not_training_ready",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_root": display_path(source),
                "source_git": git_info(source), "command": shlex.join(command), "counts": counts,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "biopython": Bio.__version__},
                "product_statistics": products["stats"], "scaffold_statistics": scaffolds["stats"],
                "dna_statistics": dna["stats"], "matrix_statistics": matrix["stats"],
                "issue_code_counts": dict(sorted(Counter(row["issue_code"] for row in issues).items())),
                "policies": {"original_data_modified": False, "samples_excluded": False,
                             "deduplication_performed": False, "split_created": False, "training_started": False,
                             "rdkit_chemistry_validated": False, "similarity_matrix_recomputed": False,
                             "current_evaluation_sequence_overlap_checked": False, "near_homology_checked": False,
                             "positive_rule": "S > threshold; unordered nonself pairs",
                             "historical_no_id_match_means_safe": False}}
    pair065 = next(row for row in pair_stats if row["cohort"] == "available_dna" and row["threshold"] == 0.65)
    summary = f"""# BGC 候选数据清单 v1

状态：清单已建立；尚未筛选、去重、划分训练集或训练。所有缺失和待核验记录保留。

## 本次结果

- 来源提交：`{metadata['source_git']['commit']}`。
- BGC 清单 {counts['bgc_count']} 条；矩阵 {counts['matrix_bgc_count']} 条；可读取 DNA {counts['dna_available_bgc_count']} 条；矩阵中缺 GBK {counts['matrix_bgc_missing_gbk_count']} 条。
- 关联产物 {counts['linked_product_count']} 条；BGC—产物关联 {counts['bgc_product_association_count']} 条；骨架原始关联 {counts['scaffold_row_count']} 条。
- 有产物字段审查标记的产物 {counts['products_with_review_flags']} 条。见 `products.tsv`、`issues.tsv`；不自动修复或删除。
- 旧审计确认重合 ID 匹配 {counts['historical_confirmed_id_matches']} 条，旧待复核 ID 匹配 {counts['historical_review_id_matches']} 条。这是历史 ID 提醒，不是当前 DNA 重合验证，也没有执行排除。
- 当前有 DNA 的矩阵子集在 S > 0.65 时，正对 {pair065['positive_pair_count']}、负对 {pair065['negative_pair_count']}，无正邻居 BGC {pair065['bgc_without_positive_neighbor']} 条。均为划分前诊断。

## 文件入口

| 文件 | 含义 |
| --- | --- |
| [bgc_manifest.tsv](bgc_manifest.tsv) | 每个 BGC 一行：DNA、全部产物 ID、可用性、审查标记、正邻居数 |
| [products.tsv](products.tsv) | 有 BGC 关联的全部产物，保留源字段和源行号 |
| [bgc_products.tsv](bgc_products.tsv) | 全部 BGC—产物关系，不把多产物压成单标签 |
| [dna_files.tsv](dna_files.tsv) | 每个实际 GBK 的解析状态、文件与序列哈希 |
| [bgc_scaffolds.tsv](bgc_scaffolds.tsv) | 保留空骨架及重复行；不是单标签表，也未凭顺序赋予产物 ID |
| [issues.tsv](issues.tsv) | 缺失、冲突、重复及历史审计提示，附结构化证据 |
| [pair_statistics.tsv](pair_statistics.tsv) | 全矩阵与有 DNA 子集的阈值覆盖，不是训练配对清单 |
| [historical_overlap_ids.tsv](historical_overlap_ids.tsv) | 本次引用的旧审计 ID 快照；未匹配不等于无泄漏 |
| [input_files.tsv](input_files.tsv) | 输入、源码及外部参考文件的 SHA256 |
| [provenance.json](provenance.json) | 版本、生成命令、检查统计与输出哈希 |

## 解释边界

`dna_status=available` 只说明文件可解析；`training_eligibility=not_assessed`、`split_status=unassigned` 对所有 BGC 保持不变。
`positive_neighbors_065_available_dna` 为空表示该 BGC 不在有 DNA 的矩阵子集中；数值 0 才表示在该子集中无正邻居。
GC 比例仅以 A/C/G/T 为分母；模糊碱基单独记录并保留。序列哈希基于完整大写 DNA，RC 哈希取正向与反向互补中字典序较小的序列。
产物字段一致性检查不是 RDKit 化学验证；不会从不一致的 InChIKey 合并产物。空骨架不作为共同类别。
未执行近同源、边界包含、当前评估集序列重合、scaffold 划分、模型 tokenization 或训练资源验证。

## 复现

在项目根目录运行以下命令，使用新的输出目录；脚本拒绝覆盖已有快照：

```bash
{metadata['command']}
```

校验当前快照及原始输入：

```bash
{shlex.quote(sys.executable)} scripts/build_manifest.py --validate {shlex.quote(display_path(output))}
```

详细字段说明见项目 `docs/manifest.md`。
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    metadata["artifacts_sha256"] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "provenance.json", metadata)
    return validate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "data/raw/BGC-clustering-benchmark")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed/benchmark_manifest_v1")
    parser.add_argument("--historical-audit-root", type=Path, help="Optional old audit; ID matches only, no exclusions")
    parser.add_argument("--validate", type=Path, metavar="SNAPSHOT", help="Validate an existing snapshot instead of building")
    parser.add_argument("--skip-input-hashes", action="store_true", help="Validate relocated outputs without original inputs")
    args = parser.parse_args()
    if args.skip_input_hashes and not args.validate:
        parser.error("--skip-input-hashes requires --validate")
    try:
        result = validate(args.validate.resolve(), not args.skip_input_hashes) if args.validate else build(
            args.source, args.output, args.historical_audit_root.resolve() if args.historical_audit_root else None)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Manifest error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
