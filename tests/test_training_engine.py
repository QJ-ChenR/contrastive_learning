import copy
import io
import random
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dna_training.engine import (
    _finite_gradients, backward_triplet, capture_rng, finish_step, make_optimizer, restore_rng,
    seed_everything, set_learning_rate,
)


class TinyEncoder(nn.Module):
    def __init__(self, mode="lora", dropout=0.3):
        super().__init__()
        self.mode = mode
        self.device = torch.device("cpu")
        self.backbone = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Dropout(dropout))
        self.head = nn.Linear(7, 4, bias=False)
        if mode == "head":
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def base_tokens(self, ids):
        values = torch.tensor(ids, dtype=torch.float32)
        if self.mode == "head":
            with torch.no_grad():
                return F.normalize(self.backbone(values), dim=-1)
        return F.normalize(self.backbone(values), dim=-1)

    def project(self, features):
        return F.normalize(self.head(features), dim=-1)

    def encode_tokens(self, ids):
        return self.project(self.base_tokens(ids))


TOKENS = {
    "a": [1, 2, 3, 4, 5], "b": [2, -1, 1, 0, 1],
    "c": [4, 1, -1, 2, 1], "d": [0, 1, 2, -3, 2],
}
CONFIG = SimpleNamespace(head_lr=1e-3, lora_lr=2e-4, weight_decay=0.01)


class TrainingEngineTests(unittest.TestCase):
    def setUp(self):
        seed_everything(21)

    def test_replay_matches_direct_all_parameter_gradients_with_dropout(self):
        direct = TinyEncoder()
        replay = copy.deepcopy(direct)
        before = capture_rng()
        direct_loss = backward_triplet(direct, TOKENS, ("a", "b", "c", 0.37), divisor=4, replay=False)
        after_direct = capture_rng()
        restore_rng(before)
        replay_loss = backward_triplet(replay, TOKENS, ("a", "b", "c", 0.37), divisor=4, replay=True)
        self.assertAlmostEqual(direct_loss, replay_loss, places=7)
        for (name, parameter), (_, other) in zip(direct.named_parameters(), replay.named_parameters()):
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(other.grad, name)
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-7, rtol=3e-6, msg=name)
        after_replay = capture_rng()
        torch.testing.assert_close(after_direct["torch_cpu"], after_replay["torch_cpu"], atol=0, rtol=0)
        self.assertEqual(after_direct["python"], after_replay["python"])
        self.assertEqual(after_direct["numpy"], after_replay["numpy"])

    def test_rng_round_trip_is_weights_only_safe(self):
        state = capture_rng()
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        expected = (random.random(), np.random.normal(), torch.rand(4))
        restore_rng(loaded)
        actual = (random.random(), np.random.normal(), torch.rand(4))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)

    def test_head_cache_updates_only_head(self):
        model = TinyEncoder(mode="head")
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        cache = {key: model.base_tokens(value) for key, value in TOKENS.items()}
        optimizer = make_optimizer(model, CONFIG)
        loss = backward_triplet(model, {}, ("a", "b", "c", -0.6), base_features=cache)
        self.assertGreater(loss, 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        norm = finish_step(model, optimizer, 1.0)
        self.assertGreater(norm, 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, parameter in model.backbone.named_parameters():
            torch.testing.assert_close(parameter, before["backbone." + name], atol=0, rtol=0)
        self.assertFalse(torch.equal(model.head.weight, before["head.weight"]))

    def test_cache_is_forbidden_for_lora(self):
        model = TinyEncoder()
        with self.assertRaisesRegex(ValueError, "head mode"):
            backward_triplet(model, TOKENS, ("a", "b", "c", 0.2), base_features={})

    def test_actual_remainder_divisor_matches_mean_loss(self):
        # A final update with two triplets must divide by two, not a configured
        # full update size of e.g. eight.
        model = TinyEncoder(dropout=0)
        reference = copy.deepcopy(model)
        triplets = [("a", "b", "c", 0.4), ("d", "a", "c", -0.2)]
        for triple in triplets:
            backward_triplet(model, TOKENS, triple, divisor=2, replay=True)
        losses = []
        for a, b, c, target in triplets:
            za, zb, zc = [reference.encode_tokens(TOKENS[key]) for key in (a, b, c)]
            losses.append((za @ zb - za @ zc - target).square())
        torch.stack(losses).mean().backward()
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-7, rtol=3e-6)

    def test_all_three_branches_contribute_gradients(self):
        class EmbeddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.table = nn.Parameter(torch.randn(3, 6))

            def encode_tokens(self, ids):
                return F.normalize(self.table[ids[0]], dim=-1)

        model = EmbeddingModel()
        backward_triplet(model, {"a": [0], "b": [1], "c": [2]}, ("a", "b", "c", 0.5))
        self.assertTrue(torch.all(torch.linalg.vector_norm(model.table.grad, dim=1) > 0))

    def test_optimizer_groups_and_schedule_round_trip(self):
        model = TinyEncoder()
        optimizer = make_optimizer(model, CONFIG)
        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertEqual(groups["head"]["lr"], CONFIG.head_lr)
        self.assertEqual(groups["lora"]["lr"], CONFIG.lora_lr)
        self.assertEqual({id(p) for p in groups["head"]["params"]}, {id(p) for p in model.head.parameters()})
        expected = {0: 0.5, 1: 1.0, 2: 1.0, 6: 0.5, 9: 0.125, 10: 0.0}
        for step, multiplier in expected.items():
            self.assertEqual(set_learning_rate(optimizer, step, 10, 0.2), multiplier)
            self.assertEqual(groups["head"]["lr"], CONFIG.head_lr * multiplier)
            self.assertEqual(groups["lora"]["lr"], CONFIG.lora_lr * multiplier)
        optimizer2 = make_optimizer(copy.deepcopy(model), CONFIG)
        optimizer2.load_state_dict(optimizer.state_dict())
        self.assertEqual(set_learning_rate(optimizer2, 6, 10, 0.2), 0.5)
        self.assertEqual(optimizer2.param_groups[0]["lr"], CONFIG.head_lr * 0.5)
        self.assertEqual(set_learning_rate(optimizer2, 0, 1, 0.05), 1.0)
        self.assertEqual(set_learning_rate(optimizer2, 0, 10, 0), 1.0)

    def test_nonfinite_loss_or_gradient_aborts_before_optimizer_update(self):
        model = TinyEncoder()
        optimizer = make_optimizer(model, CONFIG)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, float("nan"))
        before = [parameter.detach().clone() for parameter in model.parameters()]
        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient"):
            finish_step(model, optimizer, 1)
        for parameter, expected in zip(model.parameters(), before):
            torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        model.zero_grad()
        with torch.no_grad():
            model.head.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite vector") as caught:
            backward_triplet(model, TOKENS, ("a", "b", "c", 0.2))
        self.assertIn("BGC a", " ".join(caught.exception.__notes__))

    def test_finite_gradient_check_has_one_scalar_read_and_keeps_named_error(self):
        model = TinyEncoder()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        original_bool = torch.Tensor.__bool__
        scalar_reads = []

        def count_scalar_reads(value):
            scalar_reads.append(value.numel())
            return original_bool(value)

        with mock.patch.object(torch.Tensor, "__bool__", count_scalar_reads):
            _finite_gradients(model)
        self.assertEqual(scalar_reads, [1])
        model.head.weight.grad[0, 0] = float("inf")
        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient in head.weight"):
            _finite_gradients(model)

    def test_invalid_triplets_and_scales_rejected(self):
        model = TinyEncoder()
        for triple in [("a", "a", "c", 0.2), ("a", "b", "c", float("nan")), ("a", "b", "c", 2)]:
            with self.assertRaises(ValueError):
                backward_triplet(model, TOKENS, triple)
        for divisor in (0, -1, float("inf"), True):
            with self.assertRaises(ValueError):
                backward_triplet(model, TOKENS, ("a", "b", "c", 0.2), divisor=divisor)


if __name__ == "__main__":
    unittest.main()
