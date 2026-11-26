"""Real CPU end-to-end training and checkpoint-resume regression tests."""

from __future__ import annotations

from contextlib import redirect_stdout
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dna_training.common import sha256, source_hashes
from dna_training.config import TrainingConfig
from dna_training.data import Cohort
from dna_training import runtime
from test_training_data import cohort_fixture, refresh_artifact


HAS_MODEL_DEPS = all(importlib.util.find_spec(name) is not None for name in (
    "torch", "transformers", "peft", "tokenizers", "safetensors",
))


@unittest.skipUnless(HAS_MODEL_DEPS, "requires training dependencies")
class TrainingRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer, models
        from transformers import ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()
        torch.set_num_threads(1)
        cls.model_temp = tempfile.TemporaryDirectory(prefix="dna-runtime-model-")
        cls.checkpoint = Path(cls.model_temp.name) / "checkpoint"
        cls.checkpoint.mkdir()
        vocabulary = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3,
                      "[MASK]": 4, "A": 5, "C": 6, "G": 7, "T": 8}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="[UNK]")),
            unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
            pad_token="[PAD]", mask_token="[MASK]", model_max_length=8,
        )
        tokenizer.save_pretrained(cls.checkpoint)
        config = ModernBertConfig(
            vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
            local_attention=4, global_attn_every_n_layers=2,
            pad_token_id=3, cls_token_id=900, sep_token_id=901,
            bos_token_id=900, eos_token_id=901,
            embedding_dropout=0.0, attention_dropout=0.0, mlp_dropout=0.0,
        )
        config._attn_implementation = "eager"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(117)
            model = ModernBertForMaskedLM(config)
        model.save_pretrained(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.model_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dna-runtime-test-")
        self.root = Path(self.temp.name)
        self.split, self.matrix = cohort_fixture(self.root)
        # The shared fixture's validation row is constant; supply nonconstant
        # held-out relevance so nDCG here actually distinguishes rankings.
        with self.matrix.open(newline="") as stream:
            rows = list(csv.reader(stream))
        ids = rows[0][1:]
        by_id = {row[0]: row for row in rows[1:]}
        for heldout, scores in ((ids[4], [0.8, 0.4, 0.2, 0.1]), (ids[5], [0.1, 0.7, 0.3, 0.2])):
            for candidate, score in zip(ids[:4], scores):
                by_id[heldout][ids.index(candidate) + 1] = str(score)
                by_id[candidate][ids.index(heldout) + 1] = str(score)
        with self.matrix.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        input_path = self.split / "input_files.tsv"
        with input_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "sha256"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"path": self.matrix.relative_to(self.root).as_posix(), "sha256": sha256(self.matrix)})
        refresh_artifact(self.split, "input_files.tsv")
        self.cohort = runtime.cohort_for(self.config("head", "unused"))
        # Pin this test's source snapshot so concurrent edits during development
        # do not invalidate its otherwise identical interrupted/resumed runs.
        self.source_patch = mock.patch.object(runtime, "source_hashes", return_value=source_hashes())
        self.source_patch.start()

    def tearDown(self):
        self.source_patch.stop()
        self.temp.cleanup()

    def config(self, mode, directory, **overrides):
        values = {
            "project_root": str(self.root), "split_dir": str(self.split),
            "matrix_path": str(self.matrix), "checkpoint": str(self.checkpoint),
            "output_dir": str(self.root / directory), "mode": mode,
            "device": "cpu", "dtype": "float32", "attention_backend": "eager",
            "projection_dim": 7, "seed": 37, "lora_rank": 2, "lora_alpha": 4,
            "lora_dropout": 0.25, "gradient_checkpointing": True,
            "gradient_replay": True, "max_epochs": 2, "patience": 4,
            "min_delta": 0.0, "triplets_per_update": 3,
            "head_lr": 0.001, "lora_lr": 0.0002, "warmup_ratio": 0.25,
            "eval_k": 3,
            **overrides,
        }
        return TrainingConfig(**values).validate()

    def quiet(self, function, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def assert_nested_equal(self, left, right, context="state"):
        import torch

        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor, context)
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=context)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right), context)
            for key in left:
                self.assert_nested_equal(left[key], right[key], f"{context}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), context)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assert_nested_equal(a, b, f"{context}[{index}]")
        else:
            self.assertEqual(left, right, context)

    def test_three_arms_share_ids_and_training_never_reads_test_sequences_or_scores(self):
        sequence_calls, score_calls = [], []
        original_sequence, original_scores = Cohort.sequence, Cohort.scores
        test_ids = set(self.cohort.ids_by_split["test"])

        def sequence_hook(cohort, key):
            self.assertNotIn(key, test_ids, "training accessed a test DNA sequence")
            sequence_calls.append(key)
            return original_sequence(cohort, key)

        def scores_hook(cohort, query_ids, gallery_ids):
            query_ids, gallery_ids = list(query_ids), list(gallery_ids)
            self.assertFalse(test_ids & set(query_ids + gallery_ids), "training requested test supervision")
            score_calls.append((query_ids, gallery_ids))
            return original_scores(cohort, query_ids, gallery_ids)

        exports = {}
        with mock.patch.object(Cohort, "sequence", sequence_hook), mock.patch.object(Cohort, "scores", scores_hook):
            for mode in ("frozen", "head", "lora"):
                config = self.config(mode, mode, max_epochs=1)
                summary = self.quiet(runtime.train, config)
                output = config.path("output_dir")
                self.assertEqual(summary["status"], "completed")
                self.assertFalse(summary["test_evaluated"])
                self.assertFalse((output / "evaluation_test").exists())
                self.assertTrue((output / "best.pt").is_file())
                last = runtime.load_state(output / "last.pt")
                self.assertEqual(last["epoch"], 0 if mode == "frozen" else 1)
                self.assertEqual(last["optimizer_step"], 0 if mode == "frozen" else 2)
                self.assertEqual(last["optimizer"] is None, mode == "frozen")
                if mode != "frozen":
                    self.assertGreater(last["history"][-1]["train_loss"], 0)
                if mode == "lora":
                    # LoRA B starts at zero; its change proves this runtime
                    # actually updates the encoder, not just the projection.
                    self.assertTrue(any(
                        bool(value.abs().sum() > 0)
                        for name, value in last["model"]["adapters"].items()
                        if "lora_B" in name
                    ))
                self.assertEqual((output / "base_features.npz").exists(), mode != "lora")
                with np.load(output / "validation/embeddings.npz", allow_pickle=False) as saved:
                    exports[mode] = {key: saved[key].copy() for key in saved.files}
                run = json.loads((output / "run.json").read_text())
                self.assertFalse(run["test_used_for_training_or_selection"])
                self.assertEqual(run["loss"], "margin_mse")
        self.assertTrue(sequence_calls)
        self.assertTrue(score_calls)
        for mode, saved in exports.items():
            self.assertEqual(saved["gallery_ids"].tolist(), self.cohort.ids_by_split["train"])
            self.assertEqual(saved["query_ids"].tolist(), self.cohort.ids_by_split["validation"])
            expected_width = 16 if mode == "frozen" else 7
            self.assertEqual(saved["gallery_embeddings"].shape, (4, expected_width))
            np.testing.assert_allclose(np.linalg.norm(saved["gallery_embeddings"], axis=1), 1, atol=1e-6)
        # Test is reachable only through an explicit call, using the selected
        # checkpoint and the same training gallery, and is exported with IDs.
        result = self.quiet(runtime.evaluate_run, self.root / "lora", split="test")
        self.assertEqual(result["query_count"], 1)
        self.assertEqual(result["gallery_count"], 4)
        self.assertIsNotNone(result["ndcg_at_k"])
        with np.load(self.root / "lora/evaluation_test/embeddings.npz", allow_pickle=False) as saved:
            self.assertEqual(saved["query_ids"].tolist(), self.cohort.ids_by_split["test"])
            self.assertEqual(saved["gallery_ids"].tolist(), self.cohort.ids_by_split["train"])

    def test_epoch_boundary_resume_is_exact_for_head_and_dropout_lora(self):
        for mode in ("head", "lora"):
            with self.subTest(mode=mode):
                continuous = self.config(mode, mode + "_continuous")
                interrupted = self.config(mode, mode + "_resumed")
                complete = self.quiet(runtime.train, continuous)
                partial = self.quiet(runtime.train, interrupted, stop_after_epochs=1)
                self.assertEqual(partial["status"], "interrupted_at_epoch_boundary")
                self.assertEqual(partial["completed_epochs"], 1)
                # last.pt is the recovery authority even if best.pt is absent
                # after an interrupted multi-file checkpoint transaction.
                (interrupted.path("output_dir") / "best.pt").unlink()
                resumed = self.quiet(runtime.train, interrupted, resume=True)
                self.assertEqual(resumed["status"], "completed")
                self.assertEqual(resumed["completed_epochs"], 2)
                self.assertEqual(complete["best_epoch"], resumed["best_epoch"])
                self.assertEqual(complete["optimizer_steps"], resumed["optimizer_steps"])
                left = runtime.load_state(continuous.path("output_dir") / "last.pt")
                right = runtime.load_state(interrupted.path("output_dir") / "last.pt")
                for key in ("model", "optimizer", "rng", "epoch", "optimizer_step", "best_score", "best_epoch", "stale_epochs"):
                    self.assert_nested_equal(left[key], right[key], key)
                for a, b in zip(left["history"], right["history"]):
                    self.assertEqual(a["epoch"], b["epoch"])
                    self.assertEqual(a["train_loss"], b["train_loss"])
                    self.assertEqual(a["validation"], b["validation"])
                with np.load(continuous.path("output_dir") / "validation/embeddings.npz", allow_pickle=False) as a:
                    with np.load(interrupted.path("output_dir") / "validation/embeddings.npz", allow_pickle=False) as b:
                        for key in a.files:
                            np.testing.assert_array_equal(a[key], b[key])

    def test_probe_uses_train_only_and_preserves_source_checkpoint(self):
        config = self.config("lora", "unused")
        before = {path.name: sha256(path) for path in self.checkpoint.iterdir() if path.is_file()}
        observed = []
        original = Cohort.sequence

        def guarded(cohort, key):
            self.assertIn(key, self.cohort.ids_by_split["train"])
            observed.append(key)
            return original(cohort, key)

        with mock.patch.object(Cohort, "sequence", guarded):
            report = self.quiet(runtime.probe, config, self.root / "probe")
        self.assertEqual(report["status"], "passed")
        self.assertCountEqual(observed, self.cohort.ids_by_split["train"])
        self.assertLessEqual(report["save_reload_max_abs_error"], 1e-5)
        longest = max(self.cohort.ids_by_split["train"], key=lambda key: int(self.cohort.records[key]["dna_length_bp"]))
        self.assertIn(longest, [row["anchor"] for row in report["probes"]])
        self.assertTrue(all(row["input_tokens"] > 8 for row in report["probes"]))
        self.assertTrue(all(row["gradient_norm"] is not None for row in report["probes"]))
        self.assertEqual(before, {path.name: sha256(path) for path in self.checkpoint.iterdir() if path.is_file()})
        self.assertFalse(config.path("output_dir").exists())

    def test_fast_lora_matches_original_updates_and_reuses_selected_validation(self):
        import torch

        original = self.config("lora", "original", max_epochs=1)
        fast = self.config("lora", "fast", max_epochs=1, gradient_replay=False, checkpoint_token_budget=30000)
        self.quiet(runtime.train, original)
        with mock.patch.object(runtime, "encode_ids", wraps=runtime.encode_ids) as encoding:
            summary = self.quiet(runtime.train, fast)
        # Initial validation plus one epoch; export must not encode again.
        self.assertEqual(encoding.call_count, 4)
        self.assertTrue(summary["reused_best_validation"])
        left = runtime.load_state(original.path("output_dir") / "last.pt")
        right = runtime.load_state(fast.path("output_dir") / "last.pt")
        for kind in ("adapters", "head"):
            for name, tensor in left["model"][kind].items():
                torch.testing.assert_close(tensor, right["model"][kind][name], atol=2e-6, rtol=2e-5)
        self.assert_nested_equal(left["rng"], right["rng"])
        self.assertAlmostEqual(left["history"][-1]["train_loss"], right["history"][-1]["train_loss"], places=6)
        self.assertEqual(right["history"][-1]["checkpointed_triplets"], 0)
        self.assertEqual(left["history"][-1]["checkpointed_triplets"], 4)
        for field in ("training_seconds", "validation_seconds", "mean_update_seconds", "training_triplets_per_second"):
            self.assertGreater(right["history"][-1][field], 0)
        # The cached best export equals an explicit independent recomputation.
        self.quiet(runtime.evaluate_run, fast.path("output_dir"), "validation")
        with np.load(fast.path("output_dir") / "validation/embeddings.npz") as cached:
            with np.load(fast.path("output_dir") / "evaluation_validation/embeddings.npz") as recomputed:
                for key in cached.files:
                    np.testing.assert_array_equal(cached[key], recomputed[key])

    def test_selected_validation_cache_rejects_other_weights_epoch_or_embeddings(self):
        config = self.config("lora", "cache_identity", max_epochs=1)
        self.quiet(runtime.train, config)
        selected = runtime.load_state(config.path("output_dir") / "best.pt")
        for field in ("epoch", "weights", "embeddings", "ids", "signature"):
            changed = copy.deepcopy(selected)
            if field == "epoch":
                changed["best_validation"]["epoch"] += 1
            elif field == "weights":
                next(iter(changed["model"]["head"].values())).add_(0.1)
            elif field == "embeddings":
                changed["best_validation"]["gallery_embeddings"][0, 0] += 0.1
            elif field == "ids":
                changed["best_validation"]["gallery_ids"].reverse()
            else:
                changed["best_validation"]["signature_sha256"] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "validation cache"):
                runtime._export_selected_validation(changed, self.cohort, config.eval_k, self.root / "bad_export")
        self.assertFalse((self.root / "bad_export").exists())
        with self.assertRaisesRegex(ValueError, "validation cache"):
            runtime._export_selected_validation(selected, self.cohort, config.eval_k + 1, self.root / "bad_export")
        changed_cohort = copy.copy(self.cohort)
        changed_cohort.fingerprint = {**self.cohort.fingerprint, "matrix_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "validation cache"):
            runtime._export_selected_validation(selected, changed_cohort, config.eval_k, self.root / "bad_export")
        selected.pop("best_validation")
        self.assertIsNone(runtime._export_selected_validation(selected, self.cohort, config.eval_k,
                                                             self.root / "legacy"))

    def test_existing_output_and_changed_resume_config_are_rejected_without_overwriting_model(self):
        config = self.config("head", "run")
        self.quiet(runtime.train, config, stop_after_epochs=1)
        protected = ("last.pt", "best.pt", "config.json", "run.json", "history.json")
        before = {name: sha256(config.path("output_dir") / name) for name in protected}
        with self.assertRaises(FileExistsError):
            self.quiet(runtime.train, config)
        self.assertEqual(before, {name: sha256(config.path("output_dir") / name) for name in protected})
        changed = copy.deepcopy(config)
        changed.head_lr *= 2
        with self.assertRaisesRegex(ValueError, "Resume identity differs"):
            self.quiet(runtime.train, changed, resume=True)
        self.assertEqual(before, {name: sha256(config.path("output_dir") / name) for name in protected})


if __name__ == "__main__":
    unittest.main()
