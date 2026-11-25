"""Split/sequence integrity and continuous triplet coverage regressions."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

from dna_training.data import file_sha256, load_cohort, TripletSampler


def _write_tsv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def refresh_artifact(split_dir, name):
    path = split_dir / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["artifacts_sha256"][name] = file_sha256(split_dir / name)
    path.write_text(json.dumps(provenance, sort_keys=True))


def cohort_fixture(root):
    """Small relocatable cohort with four train, one validation and one test BGC."""
    root = Path(root)
    split_dir = root / "data/processed/split"
    gbk_dir = root / "data/raw/benchmark/benchmark_bgc"
    matrix_path = root / "data/raw/benchmark/tanimoto_results/NPAtlas_bm_v1.tsv"
    split_dir.mkdir(parents=True)
    gbk_dir.mkdir(parents=True)
    matrix_path.parent.mkdir(parents=True)
    ids = [f"BGC{i:07d}" for i in range(1, 8)]
    partitions = {"train": ids[:4], "validation": ids[4:5], "test": ids[5:6]}
    rows, groups = [], []
    for index, key in enumerate(ids):
        split = next((name for name, members in partitions.items() if key in members), "excluded")
        sequence = "ACGTN" + "A" * (index + 1) + "CGT"
        record = SeqRecord(Seq(sequence), id=key, name=key, description="training fixture")
        record.annotations["molecule_type"] = "DNA"
        gbk_path = gbk_dir / f"{key}.gbk"
        SeqIO.write(record, gbk_path, "genbank")
        excluded = split == "excluded"
        reasons = ["connected_to_external_evaluation"] if excluded else []
        group = f"HG_{index}"
        rows.append({
            "bgc_id": key, "split_status": split,
            "training_eligibility": "eligible_train" if split == "train" else "not_training",
            "dna_status": "available", "homology_group_id": group,
            "product_count": "1", "product_ids_json": json.dumps([f"NPA{index}"]),
            "exclusion_reasons_json": json.dumps(reasons),
            "gbk_path": gbk_path.relative_to(root).as_posix(),
            "gbk_sha256": file_sha256(gbk_path), "record_id": key,
            "dna_length_bp": str(len(sequence)), "dna_sha256": sha256(sequence.encode()).hexdigest(),
            "gc_fraction": "0.4", "organism": "Testus fixture",
        })
        groups.append({
            "group_id": group, "split": split, "external_count": "1" if excluded else "0",
            "quarantine_ids_json": "[]", "exclusion_reasons_json": json.dumps(reasons),
            "candidate_ids_json": json.dumps([key]),
        })
    _write_tsv(split_dir / "bgc_split_manifest.tsv", rows)
    _write_tsv(split_dir / "homology_groups.tsv", groups)
    for split, members in partitions.items():
        (split_dir / f"{split}_ids.txt").write_text("\n".join(members) + "\n")
    matrix = np.full((len(ids), len(ids)), 0.2)
    np.fill_diagonal(matrix, 1)
    matrix[0, 1] = matrix[1, 0] = 0.9
    matrix[0, 2] = matrix[2, 0] = 0.4
    matrix[0, 3] = matrix[3, 0] = 0.1
    matrix[2, 3] = matrix[3, 2] = 0.3
    with matrix_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MiBIG_ID", *ids])
        # Exercise ID alignment: source rows intentionally do not match columns.
        for index in reversed(range(len(ids))):
            writer.writerow([ids[index], *matrix[index]])
    _write_tsv(split_dir / "input_files.tsv", [{
        "path": matrix_path.relative_to(root).as_posix(), "sha256": file_sha256(matrix_path),
    }])
    names = ["bgc_split_manifest.tsv", "homology_groups.tsv", "input_files.tsv",
             "train_ids.txt", "validation_ids.txt", "test_ids.txt"]
    (split_dir / "provenance.json").write_text(json.dumps({
        "schema": "dna_homology_split_v1",
        "artifacts_sha256": {name: file_sha256(split_dir / name) for name in names},
    }, sort_keys=True))
    return split_dir, matrix_path


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bgc-training-data-")
        self.root = Path(self.temp.name) / "project"
        self.split, self.matrix = cohort_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        return load_cohort(self.root, self.split, self.matrix)

    def edit_rows(self, name, edit):
        path = self.split / name
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        edit(rows)
        _write_tsv(path, rows)
        refresh_artifact(self.split, name)

    def test_current_ids_scores_and_full_dna(self):
        cohort = self.load()
        self.assertEqual([len(cohort.ids_by_split[name]) for name in ("train", "validation", "test")], [4, 1, 1])
        self.assertEqual(len(cohort.records), 7)
        self.assertEqual(len(cohort.ids), 6)
        query = ["BGC0000003", "BGC0000001"]
        gallery = ["BGC0000002", "BGC0000004"]
        np.testing.assert_allclose(cohort.scores(query, gallery), [[0.2, 0.3], [0.9, 0.1]])
        self.assertEqual(cohort.sequence("BGC0000001"), "ACGTNACGT")
        with self.assertRaisesRegex(ValueError, "excluded or unknown"):
            cohort.sequence("BGC0000007")
        with self.assertRaisesRegex(ValueError, "excluded or unknown"):
            cohort.scores(["BGC0000007"], gallery)
        copied_scores = cohort.scores(query, gallery)
        copied_scores[:] = 0
        self.assertAlmostEqual(cohort.scores(query, gallery)[1, 0], 0.9)

    def test_project_can_be_relocated_without_changing_identity(self):
        original = self.load()
        new_root = Path(self.temp.name) / "server_project"
        shutil.copytree(self.root, new_root)
        relocated = load_cohort(new_root, self.split.relative_to(self.root), self.matrix.relative_to(self.root))
        self.assertEqual(original.fingerprint, relocated.fingerprint)
        self.assertEqual(original.sequence("BGC0000004"), relocated.sequence("BGC0000004"))

    def test_changed_split_artifact_rejected_before_use(self):
        (self.split / "train_ids.txt").write_text("BGC0000001\n")
        with self.assertRaisesRegex(ValueError, "artifact hash changed"):
            self.load()

    def test_duplicate_ids_and_manifest_disagreement_rejected(self):
        path = self.split / "train_ids.txt"
        path.write_text(path.read_text() + "BGC0000001\n")
        refresh_artifact(self.split, path.name)
        with self.assertRaisesRegex(ValueError, "duplicate BGC"):
            self.load()

    def test_group_cross_partition_rejected_even_with_updated_checksums(self):
        self.edit_rows("bgc_split_manifest.tsv", lambda rows: rows[4].update(homology_group_id=rows[0]["homology_group_id"]))
        with self.assertRaisesRegex(ValueError, "group crosses partitions"):
            self.load()

    def test_external_group_cannot_enter_training(self):
        self.edit_rows("homology_groups.tsv", lambda rows: rows[0].update(external_count="1"))
        with self.assertRaisesRegex(ValueError, "External or quarantined"):
            self.load()

    def test_missing_products_cannot_be_given_continuous_targets(self):
        self.edit_rows("bgc_split_manifest.tsv", lambda rows: rows[0].update(product_count="0", product_ids_json="[]"))
        with self.assertRaisesRegex(ValueError, "product mapping"):
            self.load()

    def test_excluded_row_cannot_enter_list(self):
        path = self.split / "train_ids.txt"
        path.write_text(path.read_text() + "BGC0000007\n")
        refresh_artifact(self.split, path.name)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.load()

    def test_source_matrix_hash_must_match_split(self):
        self.matrix.write_text(self.matrix.read_text().replace("0.9", "0.8"))
        with self.assertRaisesRegex(ValueError, "matrix hash differs"):
            self.load()

    def test_dna_file_and_parsed_sequence_hashes_are_checked(self):
        cohort = self.load()
        path = self.root / cohort.records["BGC0000001"]["gbk_path"]
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "GenBank file hash changed"):
            cohort.sequence("BGC0000001")
        self.edit_rows("bgc_split_manifest.tsv", lambda rows: rows[0].update(gbk_sha256=file_sha256(path), dna_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "DNA sequence hash changed"):
            self.load().sequence("BGC0000001")


class TripletSamplerTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"train_{i}" for i in range(12)]
        generator = np.random.default_rng(7)
        scores = generator.uniform(0.0, 0.6, (12, 12))
        self.scores = (scores + scores.T) / 2
        np.fill_diagonal(self.scores, 1)

    def test_every_anchor_including_zero_positive_gets_one_triplet(self):
        sampler = TripletSampler(self.ids, self.scores, seed=42)
        triples = sampler.epoch(0)
        self.assertCountEqual([row[0] for row in triples], self.ids)
        mapping = {key: index for index, key in enumerate(self.ids)}
        for anchor, first, second, gap in triples:
            self.assertEqual(len({anchor, first, second}), 3)
            self.assertTrue({anchor, first, second} <= set(self.ids))
            self.assertAlmostEqual(gap, self.scores[mapping[anchor], mapping[first]] - self.scores[mapping[anchor], mapping[second]])
        self.assertIsNone(sampler.metadata["binary_cutoff"])

    def test_sampling_is_independent_of_prior_epochs(self):
        sampler = TripletSampler(self.ids, self.scores, seed=42)
        expected = sampler.epoch(4)
        sampler.epoch(3)
        self.assertEqual(sampler.epoch(4), expected)
        self.assertEqual(TripletSampler(self.ids, self.scores, seed=42).epoch(4), expected)
        self.assertNotEqual(sampler.epoch(5), expected)

    def test_top_k_half_is_enforced_and_scores_stay_continuous(self):
        sampler = TripletSampler(self.ids, self.scores, seed=2, top_k=2)
        mapping = {key: index for index, key in enumerate(self.ids)}
        for position, (anchor, first, second, gap) in enumerate(sampler.epoch(1)):
            if position % 2 == 0:
                candidates = [key for key in self.ids if key != anchor]
                candidates.sort(key=lambda key: (self.scores[mapping[anchor], mapping[key]], key))
                self.assertTrue({first, second} & set(candidates[-2:]))
            self.assertLess(abs(gap), 0.6)

    def test_tie_break_is_seeded_and_top_k_caps_small_cohorts(self):
        scores = np.full((12, 12), 0.4)
        np.fill_diagonal(scores, 1)
        sampler = TripletSampler(self.ids, scores, seed=11, top_k=1)
        # All candidates have equal chemical scores. A lexical top-1 policy
        # would force the final ID into every top-mode triplet except its own.
        top_mode = [triplet for epoch in range(6) for triplet in sampler.epoch(epoch)[::2]]
        self.assertTrue(any(self.ids[-1] not in triplet[:3] for triplet in top_mode))
        self.assertEqual(sampler.epoch(3), TripletSampler(self.ids, scores, seed=11, top_k=1).epoch(3))
        small_scores = np.full((3, 3), 0.4)
        np.fill_diagonal(small_scores, 1)
        self.assertEqual(TripletSampler(["a", "b", "c"], small_scores, 0).metadata["effective_top_k"], 2)
        with self.assertRaisesRegex(ValueError, "top_k"):
            TripletSampler(self.ids, self.scores, seed=2, top_k=0)

    def test_equal_scores_and_three_samples_are_supported(self):
        scores = np.full((3, 3), 0.4)
        np.fill_diagonal(scores, 1)
        triples = TripletSampler(["a", "b", "c"], scores, 0).epoch(0)
        self.assertEqual([triple[-1] for triple in triples], [0.0] * 3)
        self.assertTrue(all(len(set(triple[:3])) == 3 for triple in triples))

    def test_invalid_or_unknown_scores_are_not_imputed(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            TripletSampler(["a", "b"], np.eye(2), 0)
        scores = self.scores.copy()
        scores[0, 1] = scores[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "known, finite"):
            TripletSampler(self.ids, scores, 0)
        with self.assertRaisesRegex(ValueError, "align exactly"):
            TripletSampler(self.ids[:-1], self.scores, 0)


if __name__ == "__main__":
    unittest.main()
