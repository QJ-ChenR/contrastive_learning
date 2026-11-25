"""Hand-calculated retrieval results, eligibility and tie-breaking checks."""

import unittest

import numpy as np

from dna_training.metrics import retrieval_metrics


class RetrievalMetricsTests(unittest.TestCase):
    def test_linear_ndcg_spearman_and_recall_hand_calculation(self):
        result = retrieval_metrics(
            ["query"], ["b", "a", "c"], [[1, 0]], [[0.8, 0.6], [1, 0], [0, 1]],
            [[0.9, 0.2, 0.5]], k=2,
        )
        expected = (0.2 + 0.9 / np.log2(3)) / (0.9 + 0.5 / np.log2(3))
        aggregate = result["aggregate"]
        self.assertAlmostEqual(aggregate["ndcg_at_k"], expected)
        self.assertAlmostEqual(aggregate["spearman"], -0.5)
        self.assertEqual(aggregate["recall_at_k"], 1)
        self.assertEqual(result["per_query"][0]["top_ids"], ["a", "b"])

    def test_identical_prediction_scores_break_ties_by_id_not_relevance(self):
        gallery = ["z", "a", "b"]
        scores = [0.9, 0.1, 0.7]
        result = retrieval_metrics(["q"], gallery, [[1, 0]], [[1, 0]] * 3, [scores], k=1)
        self.assertEqual(result["per_query"][0]["top_ids"], ["a"])
        self.assertAlmostEqual(result["aggregate"]["ndcg_at_k"], 0.1 / 0.9)
        self.assertEqual(result["aggregate"]["recall_at_k"], 0)
        self.assertIsNone(result["aggregate"]["spearman"])
        permutation = [1, 2, 0]
        permuted = retrieval_metrics(["q"], [gallery[i] for i in permutation], [[1, 0]], [[1, 0]] * 3,
                                     [[scores[i] for i in permutation]], k=1)
        self.assertEqual(result, permuted)

    def test_self_and_unknown_relations_are_excluded(self):
        result = retrieval_metrics(
            ["q"], ["q", "unknown", "c", "d"], [[1, 0]],
            [[1, 0], [1, 0], [0.8, 0.6], [0, 1]], [[1, np.nan, 0.4, 0.8]], k=1,
        )
        row = result["per_query"][0]
        self.assertEqual(row["top_ids"], ["c"])
        self.assertEqual(row["self_excluded_count"], 1)
        self.assertEqual(row["unknown_relation_count"], 1)
        self.assertEqual(row["valid_candidate_count"], 2)
        self.assertAlmostEqual(row["ndcg_at_k"], 0.5)

    def test_zero_relevance_has_no_ndcg_and_no_positive_recall(self):
        result = retrieval_metrics(["q"], ["a", "b"], [[1, 0]], [[1, 0], [0, 1]], [[0, 0]])
        aggregate = result["aggregate"]
        self.assertIsNone(aggregate["ndcg_at_k"])
        self.assertIsNone(aggregate["spearman"])
        self.assertIsNone(aggregate["recall_at_k"])
        self.assertEqual(aggregate["zero_ideal_queries"], 1)
        self.assertEqual(aggregate["recall_ineligible_queries"], 1)

    def test_recall_eligibility_is_separate_from_continuous_ndcg(self):
        result = retrieval_metrics(
            ["q1", "q2"], ["a", "b"], [[1, 0], [1, 0]], [[1, 0], [0, 1]],
            [[0.65, 0.3], [0.9, 0.8]], k=1,
        )
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["ndcg_eligible_queries"], 2)
        self.assertEqual(aggregate["recall_eligible_queries"], 1)
        self.assertEqual(aggregate["recall_ineligible_queries"], 1)
        self.assertEqual(aggregate["recall_at_k"], 0.5)
        self.assertIsNone(result["per_query"][0]["recall_at_k"])

    def test_average_ranks_for_tied_spearman(self):
        result = retrieval_metrics(["q"], ["a", "b", "c"], [[1, 0]], [[1, 0], [1, 0], [0, 1]],
                                   [[0.9, 0.5, 0.5]])
        self.assertAlmostEqual(result["aggregate"]["spearman"], 0.5)

    def test_empty_gallery_and_no_known_relations_report_denominators(self):
        result = retrieval_metrics(["q"], [], [[1, 0]], np.empty((0, 2)), np.empty((1, 0)))
        self.assertEqual(result["aggregate"]["query_count"], 1)
        self.assertEqual(result["aggregate"]["no_valid_candidates_queries"], 1)
        self.assertIsNone(result["aggregate"]["ndcg_at_k"])
        result = retrieval_metrics(["q"], ["a"], [[1, 0]], [[1, 0]], [[np.nan]])
        self.assertEqual(result["per_query"][0]["unknown_relation_count"], 1)
        self.assertEqual(result["per_query"][0]["top_ids"], [])

    def test_cosine_is_invariant_to_positive_embedding_scaling(self):
        base = retrieval_metrics(["q"], ["a", "b"], [[1, 0]], [[1, 1], [0, 1]], [[0.9, 0.2]])
        scaled = retrieval_metrics(["q"], ["a", "b"], [[3, 0]], [[7, 7], [0, 5]], [[0.9, 0.2]])
        self.assertEqual(base, scaled)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            retrieval_metrics(["q"], ["a", "a"], [[1, 0]], [[1, 0], [0, 1]], [[0.5, 0.5]])
        with self.assertRaisesRegex(ValueError, "Zero-norm"):
            retrieval_metrics(["q"], ["a"], [[0, 0]], [[1, 0]], [[0.5]])
        with self.assertRaisesRegex(ValueError, "Relevance"):
            retrieval_metrics(["q"], ["a"], [[1, 0]], [[1, 0]], [[np.inf]])
        with self.assertRaisesRegex(ValueError, "align exactly"):
            retrieval_metrics(["q"], ["a"], [[1, 0]], [[1, 0]], [[0.5, 0.2]])


if __name__ == "__main__":
    unittest.main()
