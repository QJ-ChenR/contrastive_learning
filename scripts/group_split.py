"""Deterministic sequence-leakage groups and BGC-level holdout assignment.

An edge is an already accepted sequence relationship. Connected components are
indivisible leakage groups, not biological families or chemical classes. This
module never reads product scores or labels.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import random


SPLITS = ("train", "validation", "test")


def _ids(values, description):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{description} must be an iterable of IDs, not a string")
    result = list(values)
    if any(not isinstance(value, str) or not value or value.strip() != value for value in result):
        raise ValueError(f"{description} must contain nonempty strings without surrounding whitespace")
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate {description}")
    return set(result)


def _group_id(members):
    payload = json.dumps(sorted(members), ensure_ascii=True, separators=(",", ":"))
    return "HG_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_group_split(candidate_ids, external_ids, edges, *, quarantine_ids=(),
                      seed=42, ratios=(0.8, 0.1, 0.1)):
    """Build connected components, quarantine connected groups, and assign splits.

    Args:
        candidate_ids: Unique candidate BGC IDs, including candidates to exclude.
        external_ids: Unique, separately namespaced evaluation sequence IDs.
        edges: Iterable of two-node sequence homology edges. Both endpoints must
            occur in candidate_ids or external_ids. Reversed/duplicate edges are
            deduplicated; self edges are ignored.
        quarantine_ids: Candidate IDs whose entire connected group is excluded.
        seed: Integer controlling ties between groups of equal candidate count.
        ratios: Positive train/validation/test proportions summing to one.

    Any component containing an external or quarantined node is excluded in its
    entirety. Eligible groups are sorted largest first; ties are shuffled from a
    canonical order using random.Random(seed). Each next group goes to the split
    minimizing sum(actual_BGC_count**2 / target_BGC_count). At any fixed assigned
    count this is equivalent to the proportion-weighted squared target error,
    sum((actual_BGC_count - target_BGC_count)**2 / target_BGC_count). Weighting
    spreads large groups between splits as their proportional loads grow,
    instead of first filling most of the training target with large groups.
    Exact objective ties use train, validation, test order. The final groups are
    reserved for still empty splits when there are at least three eligible groups. No group is
    broken to improve the requested ratio. This is a reproducible greedy
    assignment, not a guarantee of a globally optimal partition.

    Returns a JSON-serializable dict with candidate ``assignments`` (keyed by ID),
    all ``groups``, ``node_to_group``, and an accounting ``summary``. Each group
    and candidate row records explicit exclusion reasons. No input is mutated.
    """
    candidates = _ids(candidate_ids, "candidate IDs")
    external = _ids(external_ids, "external IDs")
    quarantine = _ids(quarantine_ids, "quarantine IDs")
    if candidates & external:
        raise ValueError("Candidate and external IDs must be disjoint; namespace external IDs")
    if not quarantine <= candidates:
        raise ValueError("Quarantine IDs must be candidate IDs")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    try:
        ratios = tuple(float(value) for value in ratios)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ratios must contain three positive finite numbers") from exc
    if (len(ratios) != 3 or any(not math.isfinite(value) or value <= 0 for value in ratios)
            or not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Ratios must contain three positive finite numbers summing to one")
    # Remove harmless decimal-roundoff from the requested proportions.
    ratio_sum = sum(ratios)
    ratios = tuple(value / ratio_sum for value in ratios)

    nodes = candidates | external
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    unique_edges = set()
    for edge in edges:
        if isinstance(edge, (str, bytes)):
            raise ValueError("Each edge must contain exactly two node IDs")
        try:
            a, b = edge
        except (ValueError, TypeError) as exc:
            raise ValueError("Each edge must contain exactly two node IDs") from exc
        if not isinstance(a, str) or not isinstance(b, str) or a not in nodes or b not in nodes:
            raise ValueError(f"Edge contains an unknown node: {edge!r}")
        if a != b:
            unique_edges.add(tuple(sorted((a, b))))
    for a, b in sorted(unique_edges):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    components = {}
    for node in sorted(nodes):
        components.setdefault(find(node), []).append(node)
    groups = []
    node_to_group = {}
    for members in components.values():
        member_set = set(members)
        group_candidates = sorted(member_set & candidates)
        group_external = sorted(member_set & external)
        group_quarantine = sorted(member_set & quarantine)
        reasons = []
        if group_external:
            reasons.append("connected_to_external_evaluation")
        if group_quarantine:
            reasons.append("connected_to_quarantined_candidate")
        identifier = _group_id(members)
        group = {
            "group_id": identifier,
            "members": members,
            "candidate_ids": group_candidates,
            "external_ids": group_external,
            "quarantine_ids": group_quarantine,
            "candidate_count": len(group_candidates),
            "external_count": len(group_external),
            "split": "excluded" if reasons else "unassigned",
            "exclusion_reasons": reasons,
        }
        if not group_candidates:
            group["split"] = "external_only"
        groups.append(group)
        for node in members:
            node_to_group[node] = identifier
    if len({group["group_id"] for group in groups}) != len(groups):
        raise ValueError("Group ID hash collision")
    groups.sort(key=lambda group: group["group_id"])

    eligible = [group for group in groups if group["split"] == "unassigned"]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    eligible.sort(key=lambda group: -group["candidate_count"])
    eligible_count = sum(group["candidate_count"] for group in eligible)
    targets = dict(zip(SPLITS, (eligible_count * ratio for ratio in ratios)))
    actual = dict.fromkeys(SPLITS, 0)
    group_counts = dict.fromkeys(SPLITS, 0)
    ensure_nonempty = len(eligible) >= len(SPLITS)
    for index, group in enumerate(eligible):
        empty = [name for name in SPLITS if group_counts[name] == 0]
        choices = empty if ensure_nonempty and len(eligible) - index == len(empty) else SPLITS
        size = group["candidate_count"]

        def objective(name):
            return sum((actual[key] + (size if key == name else 0)) ** 2 / targets[key]
                       for key in SPLITS)

        selected = min(choices, key=objective)
        group["split"] = selected
        actual[selected] += size
        group_counts[selected] += 1

    assignments = {}
    for group in groups:
        for identifier in group["candidate_ids"]:
            assignments[identifier] = {
                "bgc_id": identifier,
                "group_id": group["group_id"],
                "split": group["split"],
                "exclusion_reasons": list(group["exclusion_reasons"]),
            }
    assignments = dict(sorted(assignments.items()))
    # These checks also guard future changes to allocation from weakening isolation.
    assert set(assignments) == candidates
    assert all(node_to_group[a] == node_to_group[b] for a, b in unique_edges)
    assert all(assignments[node]["split"] == "excluded" for node in quarantine)
    assert not ensure_nonempty or all(group_counts.values())
    excluded = sum(row["split"] == "excluded" for row in assignments.values())
    assert sum(actual.values()) + excluded == len(candidates)
    reason_counts = Counter(reason for row in assignments.values() for reason in row["exclusion_reasons"])
    return {
        "assignments": assignments,
        "groups": groups,
        "node_to_group": dict(sorted(node_to_group.items())),
        "summary": {
            "seed": seed,
            "ratios": dict(zip(SPLITS, ratios)),
            "candidate_count": len(candidates),
            "external_count": len(external),
            "unique_nonself_edge_count": len(unique_edges),
            "group_count": len(groups),
            "eligible_group_count": len(eligible),
            "eligible_candidate_count": eligible_count,
            "excluded_candidate_count": excluded,
            "exclusion_reason_candidate_counts": dict(sorted(reason_counts.items())),
            "target_candidate_counts": targets,
            "split_candidate_counts": actual,
            "split_group_counts": group_counts,
            "all_splits_nonempty": all(group_counts.values()),
            "nonempty_guarantee_applicable": ensure_nonempty,
            "squared_count_error": sum((actual[name] - targets[name]) ** 2 for name in SPLITS),
            "weighted_squared_count_error": (
                sum((actual[name] - targets[name]) ** 2 / targets[name] for name in SPLITS)
                if eligible_count else 0.0
            ),
            "allocation_method": "descending group size; seeded equal-size order; greedy sum(actual BGC count squared / target BGC count); reserve empty splits",
            "group_semantics": "sequence leakage connected component, not GCF or product label",
        },
    }
