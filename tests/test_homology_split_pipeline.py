"""Integration checks for partition-bound supervision and exclusion provenance."""

import contextlib
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_homology_split as pipeline
from build_manifest import read_tsv, sha256, write_json, write_tsv
from dna_homology import aggregate_blast, BLAST_FIELDS


def chemistry_fixture():
    # Matrix order deliberately differs from alphabetical and partition order.
    ids = ["V2", "X", "B", "T", "M", "A", "V1", "C"]
    values = np.full((len(ids), len(ids)), 0.2)
    np.fill_diagonal(values, 1.0)
    index = {key: i for i, key in enumerate(ids)}

    def set_score(a, b, value):
        values[index[a], index[b]] = values[index[b], index[a]] = value

    for a, b, score in [("A", "B", 0.7), ("A", "C", 0.65),
                        ("V1", "V2", 0.8), ("V1", "A", 0.9),
                        ("V1", "B", 0.65), ("V2", "C", 0.65),
                        ("T", "B", 0.95), ("T", "V1", 0.85)]:
        set_score(a, b, score)
    for excluded in ("X", "M"):
        for other in ids:
            if other != excluded:
                set_score(excluded, other, 0.99)
    splits = {"A": "train", "B": "train", "C": "train", "V1": "validation",
              "V2": "validation", "T": "test", "X": "excluded", "M": "excluded_missing_dna"}
    groups = {key: "group:" + key for key in ids}
    groups["B"] = groups["A"]
    return {"ids": ids, "values": values}, splits, groups


class ChemicalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.matrix, self.splits, self.groups = chemistry_fixture()
        self.result = pipeline.chemical_diagnostics(self.matrix, self.splits, self.groups)

    def test_internal_pairs_exclude_self_boundary_cross_partition_and_exclusions(self):
        pairs = self.result["positive_pairs"]
        self.assertEqual({(row["split"], row["bgc_a"], row["bgc_b"]) for row in pairs},
                         {("train", "A", "B"), ("validation", "V1", "V2")})
        self.assertTrue(all(row["product_similarity"] > 0.65 for row in pairs))
        self.assertTrue(all(row["bgc_a"] != row["bgc_b"] for row in pairs))
        self.assertTrue(all(self.splits[row["bgc_a"]] == self.splits[row["bgc_b"]] == row["split"]
                            for row in pairs))
        by_split = {row["split"]: row for row in self.result["statistics"] if row["threshold"] == 0.65}
        self.assertEqual(by_split["train"]["positive_pair_count"], 1)
        self.assertEqual(by_split["train"]["negative_pair_count"], 2)
        self.assertEqual(by_split["validation"]["positive_pair_count"], 1)
        self.assertEqual(by_split["test"]["unordered_pair_count"], 0)
        self.assertTrue(next(row for row in pairs if row["split"] == "train")["same_homology_group"])

    def test_no_positive_neighbors_remain_and_missing_dna_is_not_a_query(self):
        rows = {row["bgc_id"]: row for row in self.result["neighbors"]}
        self.assertEqual(set(rows), {"A", "B", "C", "V1", "V2", "T"})
        self.assertEqual(rows["C"]["positive_neighbors_065_within_split"], 0)
        self.assertEqual(rows["T"]["positive_neighbors_065_within_split"], 0)
        self.assertEqual(rows["C"]["positive_neighbors_065_training_gallery"], 0)
        self.assertEqual(rows["V2"]["positive_neighbors_065_training_gallery"], 0)
        self.assertEqual(rows["T"]["positive_neighbors_065_training_gallery"], 1)

    def test_query_to_training_gallery_coverage_has_no_self_or_validation_gallery(self):
        rows = {row["query_split"]: row for row in self.result["retrieval"]}
        self.assertTrue(all(row["gallery_split"] == "train" and row["gallery_count"] == 3
                            for row in rows.values()))
        self.assertEqual(rows["train"]["directed_positive_query_gallery_pairs"], 2)
        self.assertEqual(rows["train"]["queries_without_positive"], 1)
        self.assertEqual(rows["validation"]["query_count"], 2)
        self.assertEqual(rows["validation"]["queries_with_positive"], 1)
        self.assertEqual(rows["validation"]["queries_without_positive"], 1)
        self.assertEqual(rows["validation"]["directed_positive_query_gallery_pairs"], 1)
        self.assertEqual(rows["test"]["directed_positive_query_gallery_pairs"], 1)

    def test_cross_partition_chemical_positives_are_diagnostics_only(self):
        locations = {(row["split_a"], row["split_b"]): row for row in self.result["positive_pair_locations"]}
        self.assertEqual({key: row["positive_pair_count"] for key, row in locations.items()}, {
            ("train", "train"): 1, ("train", "validation"): 1,
            ("test", "train"): 1, ("test", "validation"): 1, ("validation", "validation"): 1,
        })
        self.assertEqual([key for key, row in locations.items() if row["usable_for_training"]], [("train", "train")])

    def test_sub_tolerance_asymmetry_uses_one_pair_direction_everywhere(self):
        ids = ["B", "A", "V", "T"]
        values = np.full((4, 4), 0.2)
        np.fill_diagonal(values, 1.0)
        values[0, 1], values[1, 0] = 0.65 + 1e-13, 0.65
        values[1, 2], values[2, 1] = 0.65 + 1e-13, 0.65
        result = pipeline.chemical_diagnostics(
            {"ids": ids, "values": values},
            {"B": "train", "A": "train", "V": "validation", "T": "test"},
            {key: key for key in ids})
        self.assertEqual(len(result["positive_pairs"]), 1)
        train_stats = next(row for row in result["statistics"]
                           if row["split"] == "train" and row["threshold"] == 0.65)
        self.assertEqual(train_stats["positive_pair_count"], 1)
        retrieval = {row["query_split"]: row for row in result["retrieval"]}
        self.assertEqual(retrieval["train"]["directed_positive_query_gallery_pairs"], 2)
        self.assertEqual(retrieval["validation"]["directed_positive_query_gallery_pairs"], 1)
        self.assertEqual(sum(row["positive_pair_count"] for row in result["positive_pair_locations"]), 2)

    def test_thresholds_must_include_primary_supervision_rule(self):
        with self.assertRaisesRegex(ValueError, "0.65"):
            pipeline.chemical_diagnostics(self.matrix, self.splits, self.groups, thresholds=(0.6, 0.7))


class ExclusionPathTests(unittest.TestCase):
    def test_shortest_path_is_transitive_and_deterministic(self):
        nodes = {"A", "B", "C", "R", "S", "U", "isolated"}
        edges = [("A", "B"), ("B", "EVAL:x"), ("A", "C"), ("C", "EVAL:y"),
                 ("R", "S"), ("S", "U")]
        first = pipeline.exclusion_paths(nodes, edges, {"EVAL:y", "EVAL:x"}, {"R"})
        second = pipeline.exclusion_paths(nodes, [(b, a) for a, b in reversed(edges)],
                                          {"EVAL:x", "EVAL:y"}, {"R"})
        self.assertEqual(first, second)
        rows = {row["bgc_id"]: row for row in first}
        self.assertNotIn("isolated", rows)
        self.assertEqual(json.loads(rows["A"]["path_json"]), ["A", "B", "EVAL:x"])
        self.assertEqual(rows["A"]["edge_count"], 2)
        self.assertEqual(rows["A"]["seed_type"], "external_evaluation")
        self.assertEqual(json.loads(rows["U"]["path_json"]), ["U", "S", "R"])
        self.assertEqual(rows["U"]["seed_type"], "unresolved_review")
        self.assertEqual(json.loads(rows["R"]["path_json"]), ["R"])
        self.assertEqual(rows["R"]["edge_count"], 0)

    def test_equal_length_external_path_has_priority_over_review(self):
        rows = pipeline.exclusion_paths({"A", "R"}, [("A", "R"), ("A", "EVAL:x")], {"EVAL:x"}, {"R"})
        a = next(row for row in rows if row["bgc_id"] == "A")
        self.assertEqual(a["seed_type"], "external_evaluation")
        self.assertEqual(a["seed_node"], "EVAL:x")


def _hsp_line(a="A", b="B", start=1, end=400):
    length = end - start + 1
    return "\t".join(map(str, [a, b, 1000, 1000, length, length,
                                  start, end, start, end, 1e-30, 100, 0, str(length)]))


class MergeHSPSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="homology-hsp-merge-test-")
        self.root = Path(self.temp.name)
        self.primary = self.root / "primary.tsv.gz"
        self.supplement = self.root / "supplement.tsv"
        self.combined = self.root / "combined.tsv"
        self.lengths = {"A": 1000, "B": 1000}

    def tearDown(self):
        self.temp.cleanup()

    def merge(self, primary, supplement):
        with gzip.open(self.primary, "wt") as handle:
            handle.write(primary)
        self.supplement.write_text(supplement)
        return pipeline.merge_hsp_sources(self.primary, self.supplement, self.combined, self.lengths)

    def test_complementary_partial_coverage_is_unioned_before_edge_threshold(self):
        reports = self.merge(_hsp_line(start=1, end=400) + "\n",
                             _hsp_line(start=401, end=800) + "\n")
        self.assertEqual(aggregate_blast(self.primary, self.lengths)["edges"], [])
        self.assertEqual(aggregate_blast(self.supplement, self.lengths)["edges"], [])
        combined = aggregate_blast(self.combined, self.lengths)
        self.assertEqual(len(combined["edges"]), 1)
        edge = combined["edges"][0]
        self.assertEqual((edge["covered_a_bp"], edge["covered_b_bp"]), (800, 800))
        self.assertEqual(edge["qualifying_hsp_count"], 2)
        self.assertEqual([row["retained_hsp_rows"] for row in reports], [1, 1])

    def test_duplicate_hsps_across_sources_do_not_inflate_coverage(self):
        line = _hsp_line(end=600) + "\n"
        self.merge(line, line)
        combined = aggregate_blast(self.combined, self.lengths)
        self.assertEqual(combined["edges"], [])
        self.assertEqual(len(combined["pairs"]), 1)
        pair = combined["pairs"][0]
        self.assertEqual((pair["covered_a_bp"], pair["covered_b_bp"]), (600, 600))
        self.assertEqual(pair["qualifying_hsp_count"], 2)

    def test_only_unterminated_supplement_tail_is_withheld_and_hashed(self):
        complete = _hsp_line(end=600)
        tail = "A\tB\t1000\t1000\t80"
        reports = self.merge(complete, complete + "\n" + tail)
        # The complete primary may lack a final newline; its HSP is retained.
        self.assertEqual(self.combined.read_text(), complete + "\n" + complete + "\n")
        primary, partial = reports
        self.assertEqual(primary["retained_hsp_rows"], 1)
        self.assertEqual(primary["withheld_unterminated_final_line_bytes"], 0)
        self.assertIsNone(primary["withheld_final_line_sha256"])
        self.assertEqual(partial["retained_hsp_rows"], 1)
        self.assertEqual(partial["withheld_unterminated_final_line_bytes"], len(tail.encode()))
        self.assertEqual(partial["withheld_final_line_sha256"], hashlib.sha256(tail.encode()).hexdigest())
        self.assertEqual(partial["reported_query_ids"], ["A"])
        self.assertEqual(partial["queries_without_reported_hits"], ["B"])
        self.assertEqual(partial["sha256"], sha256(self.supplement))
        self.assertEqual(aggregate_blast(self.combined, self.lengths)["statistics"]["raw_hsp_count"], 2)

    def test_even_syntactically_complete_unterminated_supplement_tail_is_withheld(self):
        reports = self.merge(_hsp_line() + "\n", _hsp_line(start=401, end=800))
        self.assertEqual(reports[1]["retained_hsp_rows"], 0)
        self.assertGreater(reports[1]["withheld_unterminated_final_line_bytes"], 0)
        self.assertEqual(aggregate_blast(self.combined, self.lengths)["edges"], [])

    def test_malformed_complete_middle_line_reaches_aggregation_and_fails(self):
        reports = self.merge(_hsp_line() + "\n",
                             _hsp_line() + "\nBROKEN\tHSP\n" + _hsp_line(start=401, end=800) + "\n")
        self.assertEqual(reports[1]["retained_hsp_rows"], 3)
        self.assertEqual(reports[1]["withheld_unterminated_final_line_bytes"], 0)
        with self.assertRaisesRegex(ValueError, r"combined.tsv:3:.*Expected 14"):
            aggregate_blast(self.combined, self.lengths)


def _snapshot(root, metadata):
    metadata = {**metadata, "artifacts_sha256": {path.name: sha256(path) for path in root.iterdir()
                                                if path.is_file() and path.name != "provenance.json"}}
    write_json(root / "provenance.json", metadata)


def search_fixture(root):
    """Small completed-search snapshot; the test does not invoke a BLAST binary."""
    manifest, overlap, search = [root / name for name in ("manifest", "overlap", "search")]
    for directory in (manifest, overlap, search, root / "raw", root / "scripts"):
        directory.mkdir(parents=True)
    actual_scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("dna_homology.py", "group_split.py", "manifest_matrix.py", "run_dna_homology.py", "build_manifest.py"):
        (root / "scripts" / name).symlink_to(actual_scripts / name)
    candidates = list("ABCDEFGHIJ") + ["X", "Y", "R", "Q"]
    nodes, originals, products = [], [], []
    for key in [*candidates, "EVAL:x"]:
        gbk = root / "raw" / (key.replace(":", "_") + ".gbk")
        gbk.write_text(f"LOCUS       {key}\nORIGIN\n        1 " + "acgt" * 250 + "\n//\n")
        nodes.append({"node_id": key, "role": "external_evaluation" if key.startswith("EVAL:") else "candidate",
                      "dna_length_bp": 1000, "gbk_path": str(gbk), "gbk_sha256": sha256(gbk)})
    for key in [*candidates, "MISSING"]:
        originals.append({"bgc_id": key, "dna_status": "missing_gbk" if key == "MISSING" else "available",
                          "product_ids_json": json.dumps(["P:" + key, "P2:" + key]),
                          "split_status": "unassigned", "training_eligibility": "not_assessed",
                          "homology_status": "not_checked", "current_evaluation_overlap_status": "not_checked"})
        products.extend({"bgc_id": key, "product_id": prefix + key} for prefix in ("P:", "P2:"))
    write_tsv(manifest / "bgc_manifest.tsv", originals, list(originals[0]))
    write_tsv(manifest / "bgc_products.tsv", products, list(products[0]))
    _snapshot(manifest, {"schema": "fixture_manifest"})
    write_json(overlap / "summary.json", {"partial_protein_review_ids": ["R"], "id_only_review_ids": []})
    (overlap / "excluded_content_ids.txt").write_text("X\n")
    _snapshot(overlap, {"schema": "fixture_overlap"})
    write_tsv(search / "sequence_manifest.tsv", nodes, list(nodes[0]))
    (search / "sequences.fasta").write_text("".join(
        f">{row['node_id']}\n" + "ACGT" * 250 + "\n" for row in nodes))
    write_tsv(search / "input_files.tsv", [{"path": str(manifest / "provenance.json"),
                                           "sha256": sha256(manifest / "provenance.json")}], ["path", "sha256"])
    write_tsv(search / "exact_edges.tsv", [{"node_a": "EVAL:x", "node_b": "X", "relationship": "exact_full_sequence"}],
              ["node_a", "node_b", "relationship"])
    with gzip.open(search / "blast_hsps.tsv.gz", "wt") as handle:
        for a, b in [("X", "Y"), ("R", "Q"), ("A", "B")]:
            handle.write("\t".join(map(str, [a, b, 1000, 1000, 800, 800, 1, 800, 1, 800, 1e-30, 100, 0, "800"])) + "\n")
    write_json(search / "protocol.json", {"task": "megablast", "word_size": 16,
                                          "identity_threshold": 0.8, "min_hsp_alignment_length": 100,
                                          "max_evalue": 1e-10, "shorter_sequence_coverage_threshold": 0.8,
                                          "diagnostic_coverage_threshold": 0.5, "split_seed": 42,
                                          "split_ratios": [0.8, 0.1, 0.1]})
    _snapshot(search, {"schema": "dna_homology_search_v1", "blast_fields": list(BLAST_FIELDS),
                       "manifest": str(manifest), "overlap": str(overlap)})
    matrix_path = root / "data/raw/BGC-clustering-benchmark/tanimoto_results/NPAtlas_bm_v1.tsv"
    matrix_path.parent.mkdir(parents=True)
    ids = [row["bgc_id"] for row in originals]
    with matrix_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MiBIG_ID", *ids])
        for a in ids:
            writer.writerow([a, *[1 if a == b else 0.9 if {a, b} == {"A", "B"} else 0.2 for b in ids]])
    return search, manifest


class SplitSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="homology-split-test-")
        self.root = Path(self.temp.name)
        self.search, self.manifest = search_fixture(self.root)
        self.output = self.root / "output"
        self.project_patch = patch.object(pipeline, "PROJECT", self.root)
        self.project_patch.start()
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.build(self.search, self.output)

    def tearDown(self):
        self.project_patch.stop()
        self.temp.cleanup()

    def test_completed_snapshot_preserves_inventory_products_and_exclusion_chains(self):
        result = pipeline.validate(self.output)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["bgc_count"], 15)
        self.assertEqual(result["groups_crossing_splits"], 0)
        rows = {row["bgc_id"]: row for row in read_tsv(self.output / "bgc_split_manifest.tsv")}
        for key in ("X", "Y", "R", "Q"):
            self.assertEqual(rows[key]["split_status"], "excluded")
        self.assertEqual(rows["MISSING"]["split_status"], "excluded_missing_dna")
        self.assertEqual(rows["MISSING"]["homology_group_id"], "")
        self.assertEqual(rows["MISSING"]["positive_neighbors_065_within_split"], "")
        self.assertEqual(rows["A"]["split_status"], rows["B"]["split_status"])
        self.assertEqual(rows["A"]["homology_group_id"], rows["B"]["homology_group_id"])
        self.assertEqual({rows[key]["split_status"] for key in "ABCDEFGHIJ"}, {"train", "validation", "test"})
        self.assertEqual(len(read_tsv(self.output / "bgc_products_split.tsv")), 30)
        for source in read_tsv(self.manifest / "bgc_manifest.tsv"):
            self.assertEqual(rows[source["bgc_id"]]["product_ids_json"], source["product_ids_json"])
        paths = {row["bgc_id"]: row for row in read_tsv(self.output / "exclusion_paths.tsv")}
        self.assertEqual(json.loads(paths["Y"]["path_json"]), ["Y", "X", "EVAL:x"])
        self.assertEqual(json.loads(paths["Q"]["path_json"]), ["Q", "R"])

    def test_output_tampering_and_changed_inputs_fail_validation(self):
        path = self.output / "train_ids.txt"
        original = path.read_bytes()
        path.write_bytes(original + b"MISSING\n")
        with self.assertRaisesRegex(ValueError, "Changed snapshot artifact"):
            pipeline.validate(self.output)
        path.write_bytes(original)
        matrix = self.root / "data/raw/BGC-clustering-benchmark/tanimoto_results/NPAtlas_bm_v1.tsv"
        matrix.write_text(matrix.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Changed split input"):
            pipeline.validate(self.output)
        self.assertEqual(pipeline.validate(self.output, check_inputs=False)["input_hashes_checked"], 0)

    def test_build_refuses_overwriting_snapshot(self):
        with self.assertRaises(FileExistsError):
            pipeline.build(self.search, self.output)

    def test_logical_pair_validation_is_independent_of_output_hashes(self):
        path = self.output / "positive_pairs.tsv"
        pairs = read_tsv(path)
        original = path.read_bytes()
        metadata = json.loads((self.output / "provenance.json").read_text())
        self.assertEqual(len(pairs), 1)
        bad_cases = [
            ([{**pairs[0], "bgc_b": pairs[0]["bgc_a"]}], "Self or repeated"),
            ([pairs[0], pairs[0]], "Self or repeated"),
            ([{**pairs[0], "product_similarity": 0.65}], "Non-positive"),
        ]
        for bad_rows, expected_error in bad_cases:
            with self.subTest(expected_error=expected_error, count=len(bad_rows)):
                write_tsv(path, bad_rows, list(pairs[0]))
                # Simulate a producer error with internally matching hashes.
                _snapshot(self.output, metadata)
                with self.assertRaisesRegex(ValueError, expected_error):
                    pipeline.validate(self.output)
                path.write_bytes(original)
        _snapshot(self.output, metadata)

    def test_available_candidate_cannot_disappear_from_group_membership(self):
        path = self.output / "homology_members.tsv"
        members = read_tsv(path)
        write_tsv(path, [row for row in members if row["node_id"] != "A"], list(members[0]))
        _snapshot(self.output, json.loads((self.output / "provenance.json").read_text()))
        with self.assertRaisesRegex(ValueError, "Available BGCs differ"):
            pipeline.validate(self.output)

    def test_incomplete_supplement_adds_evidence_and_records_withheld_tail(self):
        supplement = self.root / "supplement"
        supplement.mkdir()
        for name in ("sequence_manifest.tsv", "sequences.fasta"):
            (supplement / name).write_bytes((self.search / name).read_bytes())
        write_json(supplement / "protocol.json", {"task": "blastn", "word_size": 11})
        tail = "interrupted-final-record"
        (supplement / "blast_hsps.tsv").write_text(_hsp_line("C", "D", end=800) + "\n" + tail)
        _snapshot(supplement, {"schema": "dna_homology_search_incomplete_v1", "full_blast_complete": False})
        output = self.root / "with_supplement"
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.build(self.search, output, supplement_search=supplement)
        self.assertEqual(pipeline.validate(output)["status"], "passed")
        rows = {row["bgc_id"]: row for row in read_tsv(output / "bgc_split_manifest.tsv")}
        self.assertEqual(rows["C"]["homology_group_id"], rows["D"]["homology_group_id"])
        reports = json.loads((output / "alignment_source_coverage.json").read_text())
        self.assertEqual(reports[1]["withheld_unterminated_final_line_bytes"], len(tail))
        protocol = json.loads((output / "protocol.json").read_text())
        self.assertEqual(protocol["task"], "megablast")
        self.assertEqual(protocol["word_size"], 16)
        self.assertEqual(protocol["partial_supplement"]["word_size"], 11)


if __name__ == "__main__":
    unittest.main()
