"""CPU integration tests with a tiny, locally generated ModernBERT checkpoint."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


HAS_MODEL_DEPS = all(importlib.util.find_spec(name) is not None for name in (
    "torch", "transformers", "peft", "tokenizers", "safetensors",
))


@unittest.skipUnless(HAS_MODEL_DEPS, "requires training dependencies")
class TrainingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer, models
        from transformers import ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="dna-model-test-")
        cls.checkpoint = Path(cls.temp.name) / "checkpoint"
        cls.checkpoint.mkdir()
        vocabulary = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4,
                      "A": 5, "C": 6, "G": 7, "T": 8}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="[UNK]")),
            unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
            mask_token="[MASK]", model_max_length=8,
        )
        tokenizer.save_pretrained(cls.checkpoint)
        config = ModernBertConfig(
            vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=8, local_attention=4, global_attn_every_n_layers=2,
            pad_token_id=3, cls_token_id=75, sep_token_id=76,
            bos_token_id=75, eos_token_id=76,
            embedding_dropout=0.0, attention_dropout=0.0, mlp_dropout=0.0,
        )
        config._attn_implementation = "eager"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = ModernBertForMaskedLM(config)
        model.save_pretrained(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def settings(self, **overrides):
        return {
            "checkpoint": str(self.checkpoint), "device": "cpu", "dtype": "float32",
            "attention_backend": "eager", "mode": "frozen", "seed": 23,
            "projection_dim": 7, "lora_rank": 2, "lora_alpha": 4,
            "lora_dropout": 0.0, "gradient_checkpointing": True,
            **overrides,
        }

    def test_full_input_pooling_retains_unk_and_excludes_specials(self):
        import torch
        from dna_training.model import load_encoder

        encoder = load_encoder(self.settings())
        ids, metadata = encoder.tokenize("ACGTNACGTACGT")
        self.assertEqual(len(ids), 13)
        self.assertEqual(metadata["unknown_token_count"], 1)
        self.assertEqual(metadata["unknown_base_count"], 1)
        self.assertTrue(metadata["exceeds_config_max_position_embeddings"])
        self.assertEqual(encoder.metadata["special_token_ids"]["cls"], 1)
        self.assertIn("cls_token_id", encoder.metadata["special_id_corrections"])
        self.assertEqual(encoder.backbone.get_input_embeddings().weight.shape[0], 9)
        vector = encoder.encode_tokens(ids)
        input_ids = torch.tensor([[1, *ids, 2]])
        with torch.no_grad():
            content = encoder.backbone(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).last_hidden_state[0, 1:-1]
            expected = content.float().mean(0)
            expected /= expected.norm()
        torch.testing.assert_close(vector, expected)
        self.assertFalse(vector.requires_grad)
        self.assertAlmostEqual(float(vector.norm()), 1.0, places=6)

    def test_limit_fails_without_truncation_and_rejects_structural_ids(self):
        from dna_training.model import load_encoder

        encoder = load_encoder(self.settings(max_tokens=6))
        with self.assertRaisesRegex(ValueError, "no truncation"):
            encoder.tokenize("ACGTN")
        with self.assertRaisesRegex(ValueError, "no truncation"):
            encoder.base_tokens([5, 6, 7, 8, 0])
        with self.assertRaisesRegex(ValueError, "structural"):
            encoder.base_tokens([5, 1, 6])
        with self.assertRaises(ValueError):
            encoder.tokenize("acgt")

    def test_head_trains_without_backbone_gradients(self):
        import torch
        from dna_training.model import load_encoder

        encoder = load_encoder(self.settings(mode="head"))
        encoder.train()
        self.assertFalse(encoder.backbone.training)
        vectors = torch.stack([encoder.encode_tokens(encoder.tokenize(sequence)[0])
                               for sequence in ("AAAAAC", "CCCCCG", "GGGGGT")])
        gap = vectors[0].dot(vectors[1]) - vectors[0].dot(vectors[2])
        (gap - 0.3).square().backward()
        self.assertIsNotNone(encoder.head.weight.grad)
        self.assertGreater(float(encoder.head.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in encoder.backbone.parameters()))
        self.assertEqual(encoder.metadata["embedding_dim"], 7)

    def test_lora_matches_initial_head_and_updates_adapter_with_checkpointing(self):
        import torch
        from dna_training.model import load_encoder

        head = load_encoder(self.settings(mode="head"))
        lora = load_encoder(self.settings(mode="lora"))
        head.eval()
        lora.eval()
        ids = head.tokenize("ACGTNACGT")[0]
        torch.testing.assert_close(head.head.weight, lora.head.weight, atol=0, rtol=0)
        torch.testing.assert_close(head.encode_tokens(ids), lora.encode_tokens(ids))
        self.assertTrue(lora.metadata["lora_target_modules"])
        self.assertTrue(all(".attn." in name for name in lora.metadata["lora_target_modules"]))
        lora.train()
        values = [lora.encode_tokens(lora.tokenize(sequence)[0])
                  for sequence in ("AAAAAC", "CCCCCG", "GGGGGT")]
        loss = ((values[0] @ values[1] - values[0] @ values[2]) - 0.3).square()
        loss.backward()
        adapter_grads = [parameter.grad for name, parameter in lora.backbone.named_parameters()
                         if "lora_" in name and parameter.grad is not None]
        self.assertTrue(adapter_grads)
        self.assertGreater(sum(float(grad.abs().sum()) for grad in adapter_grads), 0)
        self.assertTrue(all(parameter.grad is None for name, parameter in lora.backbone.named_parameters()
                            if "lora_" not in name))
        self.assertTrue(lora.backbone.is_gradient_checkpointing)

    def test_adapter_and_head_roundtrip_and_incompatible_state_rejected(self):
        import torch
        from dna_training.model import load_encoder

        original = load_encoder(self.settings(mode="lora"))
        ids = original.tokenize("ACGTNACGT")[0]
        with torch.no_grad():
            original.head.weight.add_(0.02)
            for name, parameter in original.backbone.named_parameters():
                if "lora_B" in name:
                    parameter.add_(0.1)
        state = original.trainable_state()
        self.assertTrue(all("lora_" in key for key in state["adapters"]))
        self.assertEqual(set(state["head"]), {"weight"})
        restored = load_encoder(self.settings(mode="lora", seed=99))
        restored.load_trainable_state(state)
        original.eval()
        restored.eval()
        torch.testing.assert_close(original.encode_tokens(ids), restored.encode_tokens(ids), atol=0, rtol=0)
        bad = dict(state)
        bad["head"] = {"weight": torch.zeros(1)}
        with self.assertRaisesRegex(ValueError, "incorrect shape"):
            restored.load_trainable_state(bad)
        other = load_encoder(self.settings(mode="head"))
        with self.assertRaisesRegex(ValueError, "different encoder"):
            other.load_trainable_state(state)

    def test_replay_matches_direct_gradients_and_rng_with_nonzero_dropout_adapters(self):
        import torch
        from dna_training.engine import backward_triplet, capture_rng, restore_rng, seed_everything
        from dna_training.model import load_encoder

        model = load_encoder(self.settings(mode="lora", lora_dropout=0.35,
                                           gradient_checkpointing=True))
        # Initial LoRA B is zero. Perturb it so the dropout realization affects
        # both the output and gradients, rather than only the first B gradient.
        with torch.no_grad():
            for name, parameter in model.backbone.named_parameters():
                if "lora_B" in name:
                    parameter.copy_(torch.linspace(-0.08, 0.09, parameter.numel()).reshape_as(parameter))
        model.train()
        self.assertTrue(model.backbone.is_gradient_checkpointing)
        tokens = {key: model.tokenize(sequence)[0] for key, sequence in (
            ("anchor", "AAAAACGTNACGT"), ("first", "CCCCCGTACGT"), ("second", "GGGGGTACT")
        )}
        triplet = ("anchor", "first", "second", 0.35)
        seed_everything(189)
        initial_rng = capture_rng()
        direct_loss = backward_triplet(model, tokens, triplet, divisor=3, replay=False)
        direct_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                        if parameter.requires_grad and parameter.grad is not None}
        direct_rng = capture_rng()
        model.zero_grad(set_to_none=True)
        restore_rng(initial_rng)
        replay_loss = backward_triplet(model, tokens, triplet, divisor=3, replay=True)
        replay_rng = capture_rng()
        self.assertAlmostEqual(direct_loss, replay_loss, places=7)
        self.assertTrue(any("lora_A" in name for name in direct_grads))
        self.assertGreater(sum(float(value.abs().sum()) for name, value in direct_grads.items()
                               if "lora_A" in name), 0)
        for name, parameter in model.named_parameters():
            if name in direct_grads:
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(parameter.grad, direct_grads[name], rtol=2e-5, atol=2e-6,
                                           msg=lambda message: f"{name}: {message}")
        self.assertEqual(direct_rng["python"], replay_rng["python"])
        self.assertEqual(direct_rng["numpy"], replay_rng["numpy"])
        torch.testing.assert_close(direct_rng["torch_cpu"], replay_rng["torch_cpu"], rtol=0, atol=0)
        self.assertEqual(len(direct_rng["torch_cuda"]), len(replay_rng["torch_cuda"]))
        for expected, actual in zip(direct_rng["torch_cuda"], replay_rng["torch_cuda"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_missing_weights_and_missing_checkpoint_tensor_are_fatal(self):
        from dna_training.model import load_encoder
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory(prefix="dna-bad-model-") as temp:
            destination = Path(temp) / "model"
            shutil.copytree(self.checkpoint, destination)
            weight_file = destination / "model.safetensors"
            weight_file.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "config-only"):
                load_encoder(self.settings(checkpoint=str(destination)))
            state = load_file(str(self.checkpoint / "model.safetensors"))
            state.pop("model.layers.0.attn.Wqkv.weight")
            save_file(state, str(weight_file), metadata={"format": "pt"})
            with self.assertRaisesRegex(ValueError, "strictly match"):
                load_encoder(self.settings(checkpoint=str(destination)))

    def test_null_config_path_uses_checkpoint_not_tokenizer_parent(self):
        from dna_training.model import load_encoder

        with tempfile.TemporaryDirectory(prefix="dna-split-model-layout-") as temp:
            parent = Path(temp)
            checkpoint = parent / "checkpoint-13920"
            shutil.copytree(self.checkpoint, checkpoint)
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                if (self.checkpoint / name).is_file():
                    shutil.copy2(self.checkpoint / name, parent / name)
            # The parent contains the tokenizer but no model config, as can
            # happen with Trainer checkpoint directories on the server.
            encoder = load_encoder(self.settings(checkpoint=str(checkpoint), tokenizer_path=str(parent),
                                                  config_path=None))
            self.assertEqual(encoder.metadata["config_path"], str(checkpoint / "config.json"))
            self.assertEqual(encoder.metadata["tokenizer_path"], str(parent))
            self.assertFalse((parent / "config.json").exists())

    def test_server_example_loads_both_config_and_tokenizer_from_checkpoint(self):
        from dna_training.config import PROJECT_ROOT, load_config
        from dna_training.model import load_encoder

        with tempfile.TemporaryDirectory(prefix="dna-server-model-layout-") as temp:
            parent = Path(temp)
            checkpoint = parent / "checkpoint-13920"
            shutil.copytree(self.checkpoint, checkpoint)
            config = load_config(PROJECT_ROOT / "configs/dna_margin_mse.json", {
                "checkpoint": str(checkpoint), "device": "cpu", "dtype": "float32",
                "attention_backend": "eager", "mode": "frozen",
            })
            encoder = load_encoder(config.model_settings())
            self.assertEqual(encoder.metadata["config_path"], str(checkpoint / "config.json"))
            self.assertEqual(encoder.metadata["tokenizer_path"], str(checkpoint))
            self.assertEqual(encoder.tokenize("ACGTN")[0], [5, 6, 7, 8, 0])
            self.assertFalse((parent / "tokenizer.json").exists())

    def test_wrong_tokenizer_directory_fails_before_backend_or_weights_load(self):
        from dna_training.model import load_encoder

        with tempfile.TemporaryDirectory(prefix="dna-missing-tokenizer-") as temp:
            parent = Path(temp)
            with mock.patch("transformers.ModernBertForMaskedLM.from_pretrained") as weights, \
                    mock.patch("transformers.AutoTokenizer.from_pretrained") as backend:
                for has_metadata in (False, True):
                    if has_metadata:
                        shutil.copy2(self.checkpoint / "tokenizer_config.json", parent / "tokenizer_config.json")
                    with self.subTest(has_metadata=has_metadata), self.assertRaisesRegex(
                        FileNotFoundError, "Original fast DNA tokenizer not found"
                    ) as failure:
                        load_encoder(self.settings(tokenizer_path=str(parent)))
                    self.assertIn(str(parent / "tokenizer.json"), str(failure.exception))
                    self.assertIn("--tokenizer-path", str(failure.exception))
                weights.assert_not_called()
                backend.assert_not_called()

    def test_missing_or_invalid_original_config_fails_before_loading_weights(self):
        from dna_training.model import load_encoder

        with tempfile.TemporaryDirectory(prefix="dna-bad-config-") as temp:
            directory = Path(temp)
            config_file = directory / "config.json"
            with mock.patch("transformers.ModernBertForMaskedLM.from_pretrained") as weights:
                with self.assertRaisesRegex(FileNotFoundError, "Original model config not found") as failure:
                    load_encoder(self.settings(config_path=str(directory)))
                self.assertIn(str(config_file), str(failure.exception))
                self.assertIn("--config-path", str(failure.exception))
                for contents in ("{}", '{"model_type": "bert"}', "[]", "{invalid"):
                    config_file.write_text(contents)
                    with self.subTest(contents=contents), self.assertRaisesRegex(ValueError, "model config"):
                        load_encoder(self.settings(config_path=str(directory)))
                    self.assertEqual(config_file.read_text(), contents)
                weights.assert_not_called()

    def test_vocab_pad_and_nonexistent_lora_target_are_fatal(self):
        from dna_training.model import load_encoder

        with self.assertRaisesRegex(ValueError, "matches no actual"):
            load_encoder(self.settings(mode="lora", lora_targets=["q_proj"]))
        with tempfile.TemporaryDirectory(prefix="dna-bad-tokenizer-") as temp:
            destination = Path(temp) / "tokenizer"
            shutil.copytree(self.checkpoint, destination)
            config_path = destination / "tokenizer_config.json"
            config = json.loads(config_path.read_text())
            config["pad_token"] = "[CLS]"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "PAD ID"):
                load_encoder(self.settings(tokenizer_path=str(destination)))


if __name__ == "__main__":
    unittest.main()
