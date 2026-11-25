"""Integrity regressions for candidate inventory, not biological validation."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_manifest import build, read_tsv, validate
from manifest_matrix import load_matrix, summarize_pairs
from manifest_products import load_products


def fixture(root):
    for directory in ["source_data", "tanimoto_results", "product_scaffolds", "benchmark_bgc"]:
        (root / directory).mkdir(parents=True)
    for relative in ["calculate_structural_similarity.py", "calc_product_clusters_and_scaffolds.py",
                     "README.md", "LICENSE", "source_data/LICENSE", "benchmark_bgc/LICENSE"]:
        (root / relative).write_text("fixture\n")
    fields = ["npaid", "compound_names", "compound_smiles", "compound_inchi", "compound_inchikey",
              "compound_molecular_formula", "compound_molecular_weight", "npatlas_url", "mibig_ids"]
    with (root / "source_data/NPAtlas_bm_v1.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerow(["NPA001", "one\x0cname", "C", "InChI=1S/CH4/h1H4", "KEY1", "CH4", "16", "url1", "['BGC0000001', 'BGC0000002']"])
        writer.writerow(["BM000001", "additional", "CC", "InChI=1S/CH4/h1H4", "KEY1", "C2H6", "16", "url1", "['BGC0000001']"])
        writer.writerow(["NPA003", "three", "O", "InChI=1S/H2O/h1H2", "KEY3", "H2O", "18", "url3", "['BGC0000003']"])
    # Deliberately reordered rows must be aligned by identity before statistics.
    (root / "tanimoto_results/NPAtlas_bm_v1.tsv").write_text(
        "MiBIG_ID,BGC0000001,BGC0000002,BGC0000003\n"
        "BGC0000003,0.1,0.65,1\nBGC0000001,1,0.8,0.1\nBGC0000002,0.8,1,0.65\n")
    (root / "product_scaffolds/bm_scaffolds.csv").write_text(
        "BGC0000001,\nBGC0000001,\nBGC0000002,C1CC1\nBGC0000003,O1CC1\n")
    # BGC3 remains missing, while BGC1/BGC2 intentionally have identical DNA.
    for identifier in ["BGC0000001", "BGC0000002"]:
        record = SeqRecord(Seq("ACGTNACGT"), id=identifier, name=identifier, description="fixture")
        record.annotations["molecule_type"] = "DNA"
        SeqIO.write(record, root / "benchmark_bgc" / f"{identifier}.gbk", "genbank")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bgc-manifest-test-")
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        fixture(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def test_lossless_inventory_and_missing_count(self):
        output = self.root / "output"
        result = build(self.source, output)
        self.assertEqual(result["bgc_count"], 3)
        metadata = json.loads((output / "provenance.json").read_text())
        self.assertEqual(metadata["counts"]["matrix_bgc_missing_gbk_count"], 1)
        bgcs = {row["bgc_id"]: row for row in read_tsv(output / "bgc_manifest.tsv")}
        self.assertEqual(bgcs["BGC0000003"]["dna_status"], "missing_gbk")
        self.assertEqual(bgcs["BGC0000003"]["positive_neighbors_065_available_dna"], "")
        self.assertEqual(bgcs["BGC0000003"]["positive_neighbors_065_full_matrix"], "0")
        self.assertEqual(json.loads(bgcs["BGC0000001"]["product_ids_json"]), ["BM000001", "NPA001"])
        self.assertEqual(bgcs["BGC0000001"]["scaffold_row_count"], "2")
        self.assertEqual(bgcs["BGC0000001"]["empty_scaffold_row_count"], "2")
        self.assertTrue(all(row["training_eligibility"] == "not_assessed" for row in bgcs.values()))
        self.assertEqual(metadata["dna_statistics"]["dna_exact_duplicate_group_count"], 1)
        self.assertEqual(len(read_tsv(output / "bgc_products.tsv")), 4)
        products = {r["product_id"]: r for r in read_tsv(output / "products.tsv")}
        self.assertEqual(products["NPA001"]["compound_name"], "one\x0cname")
        self.assertEqual(products["BM000001"]["formula_consistency_status"], "mismatch")
        self.assertEqual(products["BM000001"]["inchikey"], "KEY1")
        self.assertEqual(products["BM000001"]["source_line"], "3")

    def test_matrix_alignment_and_strict_threshold(self):
        matrix = load_matrix(self.source / "tanimoto_results/NPAtlas_bm_v1.tsv")
        self.assertTrue(matrix["stats"]["rows_reordered_to_columns"])
        stats = summarize_pairs(matrix, thresholds=[0.65])
        self.assertEqual(stats["summary"][0]["positive_pair_count"], 1)
        self.assertEqual(stats["degrees"]["BGC0000003"]["0.65"], 0)
        path = self.source / "tanimoto_results/NPAtlas_bm_v1.tsv"
        path.write_text(path.read_text().replace("0.8,1,0.65", "0.7,1,0.65"))
        with self.assertRaisesRegex(ValueError, "asymmetric"):
            load_matrix(path)

    def test_duplicate_products_are_rejected_not_merged(self):
        path = self.source / "source_data/NPAtlas_bm_v1.tsv"
        path.write_text(path.read_text().replace("BM000001", "NPA001"))
        with self.assertRaisesRegex(ValueError, "duplicate product ID"):
            load_products(path)

    def test_overwrite_raw_and_checksum_protection(self):
        output = self.root / "output"
        build(self.source, output)
        with self.assertRaises(FileExistsError):
            build(self.source, output)
        with self.assertRaisesRegex(ValueError, "separate"):
            build(self.source, self.source / "derived")
        raw = self.source / "source_data/NPAtlas_bm_v1.tsv"
        raw.write_text(raw.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Source checksum"):
            validate(output)
        self.assertEqual(validate(output, verify_inputs=False)["status"], "passed")
        artifact = output / "bgc_products.tsv"
        artifact.write_text(artifact.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Artifact checksum"):
            validate(output, verify_inputs=False)

    def test_historical_id_is_a_flag_not_an_exclusion(self):
        audit = self.root / "audit"
        audit.mkdir()
        (audit / "confirmed_overlap_mibig_ids.txt").write_text("BGC0000001\n")
        (audit / "additional_review_mibig_ids.txt").write_text("BGC0000003\n")
        output = self.root / "output"
        build(self.source, output, audit)
        rows = {r["bgc_id"]: r for r in read_tsv(output / "bgc_manifest.tsv")}
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows["BGC0000001"]["historical_overlap_status"], "historical_confirmed_id_match")
        self.assertEqual(rows["BGC0000002"]["historical_overlap_status"], "no_id_match_in_reference")
        self.assertTrue(all(r["current_evaluation_overlap_status"] == "not_checked" for r in rows.values()))


if __name__ == "__main__":
    unittest.main()
