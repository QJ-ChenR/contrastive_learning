"""Continuous chemical retrieval metrics with explicit query eligibility."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _ids(values: Sequence[str], description: str) -> list[str]:
    ids = list(values)
    if any(not isinstance(key, str) or not key.strip() or key != key.strip() for key in ids):
        raise ValueError(f"{description} must contain nonempty string IDs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate {description}")
    return ids


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2:
        return None
    a = _average_ranks(first)
    b = _average_ranks(second)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return None
    return float(np.clip(np.dot(a, b) / denominator, -1, 1))


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def retrieval_metrics(
    query_ids: Sequence[str], gallery_ids: Sequence[str],
    query_embeddings: np.ndarray, gallery_embeddings: np.ndarray,
    relevance: np.ndarray, k: int = 10, positive_threshold: float = 0.65,
) -> dict:
    """Evaluate held-out queries against a gallery using linear gain ``S``.

    Self matches and NaN (unknown) relevance are excluded, never relabeled as
    negatives. Infinities and finite scores outside [0,1] are invalid. Embeddings
    are normalized here, so retrieval uses cosine regardless of saved scaling.
    Exact prediction ties break by gallery ID, independent of chemistry.
    Spearman is per query over every known non-self gallery relation. Recall is
    averaged only over queries with at least one S > positive_threshold match.
    """
    queries = _ids(query_ids, "query IDs")
    gallery = _ids(gallery_ids, "gallery IDs")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(positive_threshold) or not 0 <= positive_threshold <= 1:
        raise ValueError("positive_threshold must be in [0,1]")
    q = np.asarray(query_embeddings, dtype=np.float64)
    g = np.asarray(gallery_embeddings, dtype=np.float64)
    if (q.ndim != 2 or g.ndim != 2 or q.shape[0] != len(queries)
            or g.shape[0] != len(gallery) or q.shape[1] != g.shape[1] or q.shape[1] < 1):
        raise ValueError("Embedding shapes must align to IDs and share a nonempty feature dimension")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("Non-finite retrieval embedding")
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    g_norm = np.linalg.norm(g, axis=1, keepdims=True)
    if not np.isfinite(q_norm).all() or not np.isfinite(g_norm).all():
        raise ValueError("Non-finite retrieval embedding norm")
    if np.any(q_norm == 0) or np.any(g_norm == 0):
        raise ValueError("Zero-norm retrieval embedding")
    q = q / q_norm
    g = g / g_norm
    scores = q @ g.T
    chemical = np.asarray(relevance, dtype=np.float64)
    if chemical.shape != (len(queries), len(gallery)):
        raise ValueError("Relevance must align exactly to query and gallery IDs")
    finite = np.isfinite(chemical)
    if np.isinf(chemical).any() or np.any(finite & ((chemical < 0) | (chemical > 1))):
        raise ValueError("Relevance must contain [0,1] scores or NaN for unknown relations")
    gallery_array = np.asarray(gallery, dtype=str)
    per_query = []
    for index, query in enumerate(queries):
        self_mask = gallery_array == query
        known = finite[index] & ~self_mask
        positions = np.flatnonzero(known)
        actual_k = min(int(k), len(positions))
        ordered = positions[np.lexsort((gallery_array[positions], -scores[index, positions]))]
        top = ordered[:actual_k]
        discount = 1 / np.log2(np.arange(actual_k, dtype=np.float64) + 2)
        dcg = float(np.dot(chemical[index, top], discount))
        ideal = np.sort(chemical[index, positions])[::-1][:actual_k]
        ideal_dcg = float(np.dot(ideal, discount))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else None
        if ndcg is not None:
            ndcg = float(np.clip(ndcg, 0, 1))
        positive_count = int(np.count_nonzero(chemical[index, positions] > positive_threshold))
        hits = int(np.count_nonzero(chemical[index, top] > positive_threshold))
        per_query.append({
            "query_id": query, "valid_candidate_count": len(positions),
            "unknown_relation_count": int(np.count_nonzero(~finite[index] & ~self_mask)),
            "self_excluded_count": int(self_mask.sum()),
            "ndcg_at_k": ndcg, "spearman": _spearman(scores[index, positions], chemical[index, positions]),
            "positive_count": positive_count,
            "recall_at_k": hits / positive_count if positive_count else None,
            "top_ids": [gallery[int(position)] for position in top],
        })
    aggregate = {
        "k": int(k), "query_count": len(queries), "gallery_count": len(gallery),
        "gain": "linear_product_similarity", "positive_threshold": float(positive_threshold),
        "prediction_tie_break": "gallery_id_ascending", "self_matches": "excluded",
        "unknown_relations": "excluded", "spearman_scope": "per_query_all_known_nonself_gallery",
        "ndcg_at_k": _mean_defined([row["ndcg_at_k"] for row in per_query]),
        "ndcg_eligible_queries": sum(row["ndcg_at_k"] is not None for row in per_query),
        "zero_ideal_queries": sum(row["ndcg_at_k"] is None for row in per_query),
        "spearman": _mean_defined([row["spearman"] for row in per_query]),
        "spearman_eligible_queries": sum(row["spearman"] is not None for row in per_query),
        "recall_at_k": _mean_defined([row["recall_at_k"] for row in per_query]),
        "recall_eligible_queries": sum(row["positive_count"] > 0 for row in per_query),
        "recall_ineligible_queries": sum(row["positive_count"] == 0 for row in per_query),
        "no_valid_candidates_queries": sum(row["valid_candidate_count"] == 0 for row in per_query),
    }
    return {"aggregate": aggregate, "per_query": per_query}
