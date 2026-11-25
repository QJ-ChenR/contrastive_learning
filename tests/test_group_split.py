"""Isolation and reproducibility regressions for BGC group splitting."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from group_split import build_group_split


class GroupSplitTests(unittest.TestCase):
    def test_sequence_chains_are_indivisible_without_clique_assumption(self):
        result = build_group_split(["a", "b", "c", "d", "e"], [], [("a", "b"), ("b", "c")])
        rows = result["assignments"]
        self.assertEqual(len({rows[node]["group_id"] for node in "abc"}), 1)
        self.assertEqual(len({rows[node]["split"] for node in "abc"}), 1)
        self.assertEqual(result["summary"]["eligible_group_count"], 3)
        self.assertTrue(result["summary"]["all_splits_nonempty"])

    def test_external_connection_excludes_entire_transitive_component(self):
        result = build_group_split(["a", "b", "c", "d", "e"], ["eval:x", "eval:y"],
                                   [("a", "b"), ("b", "eval:x"), ("eval:x", "eval:y")])
        rows = result["assignments"]
        for node in "ab":
            self.assertEqual(rows[node]["split"], "excluded")
            self.assertEqual(rows[node]["exclusion_reasons"], ["connected_to_external_evaluation"])
        self.assertEqual(result["summary"]["excluded_candidate_count"], 2)
        self.assertTrue(result["summary"]["all_splits_nonempty"])

    def test_quarantine_propagates_and_multiple_reasons_are_retained(self):
        result = build_group_split(["a", "b", "c"], ["eval:x"],
                                   [("a", "b"), ("b", "eval:x")], quarantine_ids=["a"])
        self.assertEqual(result["assignments"]["b"]["exclusion_reasons"],
                         ["connected_to_external_evaluation", "connected_to_quarantined_candidate"])
        group = next(group for group in result["groups"] if "a" in group["members"])
        self.assertEqual(group["quarantine_ids"], ["a"])
        self.assertEqual(group["external_ids"], ["eval:x"])
        self.assertNotEqual(result["assignments"]["c"]["split"], "excluded")

    def test_input_order_and_edge_orientation_do_not_change_result(self):
        candidates = [f"bgc:{i:03d}" for i in range(40)]
        edges = [(candidates[index], candidates[index + 1]) for index in range(0, 16, 2)]
        first = build_group_split(candidates, ["eval:x", "eval:y"], edges, seed=42)
        second = build_group_split(reversed(candidates), ["eval:y", "eval:x"],
                                    [(b, a) for a, b in reversed(edges)], seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first, build_group_split(candidates, ["eval:x", "eval:y"], edges, seed=42))
        other = build_group_split(candidates, ["eval:x", "eval:y"], edges, seed=43)
        self.assertNotEqual(first["assignments"], other["assignments"])
        self.assertEqual(first["node_to_group"], other["node_to_group"])

    def test_singleton_ratios_and_candidate_accounting(self):
        result = build_group_split([str(i) for i in range(100)], [], [])
        self.assertEqual(result["summary"]["split_candidate_counts"], {"train": 80, "validation": 10, "test": 10})
        self.assertEqual(result["summary"]["squared_count_error"], 0)
        self.assertEqual(len(result["assignments"]), 100)

    def test_large_group_is_not_broken_to_force_ratios(self):
        candidates = [str(i) for i in range(100)]
        edges = [("0", str(i)) for i in range(1, 98)]
        result = build_group_split(candidates, [], edges)
        rows = result["assignments"]
        self.assertEqual(len({rows[str(i)]["split"] for i in range(98)}), 1)
        self.assertEqual(sorted(result["summary"]["split_candidate_counts"].values()), [1, 1, 98])
        for a, b in edges:
            self.assertEqual(rows[a]["split"], rows[b]["split"])
        self.assertTrue(result["summary"]["all_splits_nonempty"])

    def test_proportional_load_spreads_large_groups_across_splits(self):
        grouped = [[f"group:{group}:member:{member}" for member in range(10)]
                   for group in range(20)]
        candidates = [node for group in grouped for node in group]
        candidates.extend(f"singleton:{index}" for index in range(100))
        edges = [(group[0], node) for group in grouped for node in group[1:]]
        result = build_group_split(candidates, [], edges)
        rows = result["assignments"]
        self.assertEqual(result["summary"]["split_candidate_counts"],
                         {"train": 240, "validation": 30, "test": 30})
        self.assertEqual(result["summary"]["weighted_squared_count_error"], 0)
        # Raw target-error greedy fills train first, leaving held-out sets made
        # mostly of singletons. Large groups must also reach both held-out sets.
        large_group_splits = {rows[group[0]]["split"] for group in grouped}
        self.assertEqual(large_group_splits, {"train", "validation", "test"})
        for group in grouped:
            self.assertEqual(len({rows[node]["split"] for node in group}), 1)

    def test_insufficient_groups_and_all_excluded_are_explicit(self):
        small = build_group_split(["a", "b"], [], [])
        self.assertFalse(small["summary"]["nonempty_guarantee_applicable"])
        self.assertFalse(small["summary"]["all_splits_nonempty"])
        excluded = build_group_split(["a", "b"], [], [("a", "b")], quarantine_ids=["a"])
        self.assertEqual(excluded["summary"]["excluded_candidate_count"], 2)
        self.assertEqual(excluded["summary"]["eligible_group_count"], 0)
        self.assertEqual(excluded["summary"]["split_candidate_counts"], {"train": 0, "validation": 0, "test": 0})

    def test_unknown_nodes_identity_collisions_and_invalid_ratios_fail(self):
        with self.assertRaisesRegex(ValueError, "unknown node"):
            build_group_split(["a"], [], [("a", "missing")])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_group_split(["a"], ["a"], [])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_group_split(["a", "a"], [], [])
        with self.assertRaisesRegex(ValueError, "Quarantine"):
            build_group_split(["a"], [], [], quarantine_ids=["missing"])
        for ratios in [(0.8, 0.1), (0.8, 0.1, 0.2), (1, 0, 0), (float("nan"), 0.1, 0.1)]:
            with self.subTest(ratios=ratios), self.assertRaisesRegex(ValueError, "Ratios"):
                build_group_split(["a"], [], [], ratios=ratios)

    def test_duplicate_reverse_and_self_edges_do_not_affect_graph(self):
        first = build_group_split(["a", "b", "c"], [], [("a", "b")])
        repeated = build_group_split(["a", "b", "c"], [], [("a", "b"), ("b", "a"), ("a", "a")])
        self.assertEqual(first, repeated)


if __name__ == "__main__":
    unittest.main()
