"""Resource probes must exercise both compute/memory paths without changing data."""

from __future__ import annotations

import csv
import json
import unittest
from unittest import mock

from dna_training import runtime
import test_training_runtime as fixtures


@unittest.skipUnless(fixtures.HAS_MODEL_DEPS, "requires training dependencies")
class TrainingProbePerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TrainingRuntimeTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.TrainingRuntimeTests.tearDownClass()

    def setUp(self):
        # Reuse its data/checkpoint fixture, without inheriting or discovering
        # the end-to-end tests a second time in this module.
        self.fixture = fixtures.TrainingRuntimeTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_direct_and_replay_probe_counts_match_actual_encoder_calls(self):
        from dna_training.model import DNAEncoder

        original_encode, original_prepare = DNAEncoder.encode_tokens, DNAEncoder.prepare_tokens
        for replay, expected_per_triplet in ((False, 3), (True, 6)):
            calls, prepares = [], []

            def count_encode(model, tokens):
                calls.append(len(tokens))
                return original_encode(model, tokens)

            def count_prepare(model, tokens):
                prepares.append(len(tokens))
                return original_prepare(model, tokens)

            destination = self.fixture.root / f"probe_{replay}"
            config = self.fixture.config("lora", "unused", gradient_replay=replay)
            with self.subTest(replay=replay), \
                    mock.patch.object(DNAEncoder, "encode_tokens", count_encode), \
                    mock.patch.object(DNAEncoder, "prepare_tokens", count_prepare):
                report = self.fixture.quiet(runtime.probe, config, destination)
            self.assertEqual(report["status"], "passed")
            self.assertEqual([row["case"] for row in report["probes"]],
                             ["median", "p90", "longest", "longest_triplet"])
            # Each complete BGC is prepared once, including across save/reload.
            self.assertEqual(len(prepares), len(self.fixture.cohort.ids_by_split["train"]))
            self.assertEqual(len(calls), expected_per_triplet * 4 + 2)
            self.assertEqual(sum(row["encoder_forward_calls"] for row in report["probes"]), len(calls) - 2)
            self.assertEqual(report["device"], {"type": "cpu", "name": "cpu", "total_memory_bytes": None})
            with (destination / "train_token_lengths.tsv").open(newline="") as handle:
                token_lengths = {row["bgc_id"]: int(row["total_input_tokens"])
                                 for row in csv.DictReader(handle, delimiter="\t")}
            expected_longest = sorted(token_lengths, key=lambda key: (token_lengths[key], key))[-3:][::-1]
            stress = report["probes"][-1]
            self.assertEqual(stress["triplet"], expected_longest)
            for row in report["probes"]:
                self.assertEqual(row["triplet_input_tokens"], [token_lengths[key] for key in row["triplet"]])
                self.assertEqual(row["triplet_total_input_tokens"], sum(row["triplet_input_tokens"]))
                self.assertTrue(row["active_gradient_checkpointing"])
                self.assertIsNone(row["peak_cuda_allocated_bytes"])
                self.assertIsNone(row["peak_cuda_reserved_bytes"])

    def test_adaptive_probe_exercises_real_sampler_boundary_and_longest_triplet(self):
        cohort = self.fixture.cohort
        lengths = {key: int(cohort.records[key]["dna_length_bp"]) + 2 for key in cohort.ids_by_split["train"]}
        sampled = runtime.sampler_for(cohort, 37).epoch(0)
        # This fixture uses a one-base tokenizer. Select an observed budget
        # below the largest possible triplet, so both paths must run.
        budget = min(sum(lengths[key] for key in triple[:3]) for triple in sampled)
        self.assertLess(budget, sum(sorted(lengths.values())[-3:]))
        eligible = [triple for triple in sampled if sum(lengths[key] for key in triple[:3]) <= budget]
        expected = max(eligible, key=lambda triple: (sum(lengths[key] for key in triple[:3]), triple[:3]))
        config = self.fixture.config("lora", "unused", gradient_replay=False, checkpoint_token_budget=budget)
        report = self.fixture.quiet(runtime.probe, config, self.fixture.root / "adaptive_probe")
        boundary = next(row for row in report["probes"] if row["case"] == "activation_budget_boundary")
        stress = next(row for row in report["probes"] if row["case"] == "longest_triplet")
        self.assertEqual(boundary["triplet"], list(expected[:3]))
        self.assertLessEqual(boundary["triplet_total_input_tokens"], budget)
        self.assertFalse(boundary["active_gradient_checkpointing"])
        self.assertGreater(stress["triplet_total_input_tokens"], budget)
        self.assertTrue(stress["active_gradient_checkpointing"])
        self.assertEqual(report["activation_budget_boundary"]["eligible_triplets"], len(eligible))
        self.assertEqual(report["status"], "passed")

    def test_longest_triplet_failure_is_reported_and_propagated(self):
        import torch
        from dna_training import engine

        original = engine.backward_triplet
        calls = 0

        def fail_stress(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise torch.OutOfMemoryError("synthetic longest-triplet allocation failure")
            return original(*args, **kwargs)

        destination = self.fixture.root / "failed_probe"
        with mock.patch.object(engine, "backward_triplet", side_effect=fail_stress), \
                self.assertRaisesRegex(torch.OutOfMemoryError, "longest-triplet"):
            self.fixture.quiet(runtime.probe, self.fixture.config("lora", "unused"), destination)
        report = json.loads((destination / "probe.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["probes"]), 3)
        self.assertEqual(report["active_probe"]["case"], "longest_triplet")
        self.assertNotIn("save_reload_max_abs_error", report)
        self.assertEqual(report["error"]["type"], "OutOfMemoryError")
