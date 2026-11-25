"""Aggregate BLASTN evidence for operational DNA near-homology isolation.

This module makes no GCF, function, or remote-homology claim. It streams BLAST
HSPs, retains only those meeting the fixed HSP criteria, and merges genomic
intervals separately on each sequence. Reciprocal searches do not duplicate
coverage. BTOP decoding excludes columns with a gap on either sequence from
both coverage numerators; the full original DNA length remains the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import gzip
import math
from numbers import Integral
from pathlib import Path
import re


BLAST_FIELDS = (
    "qseqid", "sseqid", "qlen", "slen", "nident", "length", "qstart",
    "qend", "sstart", "send", "evalue", "bitscore", "gaps", "btop",
)
PAIR_FIELDS = (
    "node_a", "node_b", "length_a", "length_b", "covered_a_bp",
    "covered_b_bp", "coverage_a", "coverage_b", "shorter_coverage",
    "longer_coverage", "qualifying_hsp_count", "query_direction_count",
    "min_hsp_identity", "max_hsp_identity", "min_evalue", "max_bitscore",
    "is_edge",
)
_BTOP_TOKEN = re.compile(r"[0-9]+|[ACGTRYSWKMBDHVN-]{2}", re.IGNORECASE)


def merge_intervals(intervals):
    """Return a nonoverlapping union of zero-based, half-open intervals."""
    merged = []
    for start, end in sorted(intervals):
        if not isinstance(start, Integral) or not isinstance(end, Integral) or start < 0 or end <= start:
            raise ValueError(f"Invalid half-open interval: {(start, end)}")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def union_length(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def btop_intervals(btop, qstart, qend, sstart, send, alignment_length, nident, gaps):
    """Decode BTOP to paired-base coverage intervals on both original strands.

    Input coordinates follow BLAST's one-based inclusive convention. Mismatch
    columns count as aligned coverage; any single-sided gap column counts for
    neither sequence. Identity is assessed separately as nident / all columns.
    The BTOP counts must agree with the other BLAST fields.
    """
    qstep = 1 if qend >= qstart else -1
    sstep = 1 if send >= sstart else -1
    qpos, spos = qstart - 1, sstart - 1
    query_intervals, subject_intervals = [], []
    block_query, block_subject, block_length = qpos, spos, 0
    columns = identity_count = gap_count = last_end = 0

    def flush():
        nonlocal block_length
        if block_length:
            qlast = block_query + qstep * (block_length - 1)
            slast = block_subject + sstep * (block_length - 1)
            query_intervals.append((min(block_query, qlast), max(block_query, qlast) + 1))
            subject_intervals.append((min(block_subject, slast), max(block_subject, slast) + 1))
            block_length = 0

    for match in _BTOP_TOKEN.finditer(btop):
        if match.start() != last_end:
            raise ValueError(f"Malformed BTOP at offset {last_end}")
        last_end = match.end()
        token = match.group()
        if token[0].isdigit():
            count = int(token)
            if not block_length:
                block_query, block_subject = qpos, spos
            block_length += count
            qpos += qstep * count
            spos += sstep * count
            columns += count
            identity_count += count
        elif "-" not in token:
            if not block_length:
                block_query, block_subject = qpos, spos
            block_length += 1
            qpos += qstep
            spos += sstep
            columns += 1
        else:
            if token == "--":
                raise ValueError("BTOP cannot align two gaps")
            flush()
            qpos += qstep if token[0] != "-" else 0
            spos += sstep if token[1] != "-" else 0
            columns += 1
            gap_count += 1
    flush()
    if not btop or last_end != len(btop):
        raise ValueError(f"Malformed BTOP at offset {last_end}")
    expected = (alignment_length, nident, gaps, qend - 1 + qstep, send - 1 + sstep)
    actual = (columns, identity_count, gap_count, qpos, spos)
    if actual != expected:
        raise ValueError(f"BTOP disagrees with alignment fields: {actual} != {expected}")
    return query_intervals, subject_intervals


@dataclass
class _Pair:
    intervals_a: list = field(default_factory=list)
    intervals_b: list = field(default_factory=list)
    hsp_count: int = 0
    direction_bits: int = 0
    min_identity: float = 1.0
    max_identity: float = 0.0
    min_evalue: float = math.inf
    max_bitscore: float = 0.0

    def add(self, a_intervals, b_intervals, direction, identity, evalue, bitscore):
        self.intervals_a.extend(a_intervals)
        self.intervals_b.extend(b_intervals)
        # Bound storage for highly repetitive pairs without altering coverage.
        if len(self.intervals_a) > 2048:
            self.intervals_a = merge_intervals(self.intervals_a)
        if len(self.intervals_b) > 2048:
            self.intervals_b = merge_intervals(self.intervals_b)
        self.hsp_count += 1
        self.direction_bits |= direction
        self.min_identity = min(self.min_identity, identity)
        self.max_identity = max(self.max_identity, identity)
        self.min_evalue = min(self.min_evalue, evalue)
        self.max_bitscore = max(self.max_bitscore, bitscore)


def _threshold(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return Fraction(str(value))


def _at_least(numerator, denominator, threshold):
    return numerator * threshold.denominator >= denominator * threshold.numerator


def aggregate_blast(path, sequence_lengths, *, identity_threshold=0.80,
                    min_alignment_length=100, max_evalue=1e-10,
                    edge_coverage=0.80, report_coverage=0.50):
    """Return diagnostic ``pairs``, qualifying ``edges``, and ``statistics``.

    ``sequence_lengths`` maps every FASTA node ID to its complete DNA length.
    Input must have exactly ``BLAST_FIELDS`` in outfmt-6 order, optionally gzip
    compressed. Unknown IDs, mismatched lengths, malformed numeric values,
    inconsistent BTOP, or invalid coordinates fail with a source line number.

    The shorter-sequence coverage determines eligibility (inclusive boundary).
    For equal lengths, use the smaller of the two coverages conservatively.
    Pair rows below ``report_coverage`` are omitted only from the diagnostics;
    every HSP has still been parsed. No biological labels enter this routine.
    """
    identity_fraction = _threshold(identity_threshold, "identity_threshold")
    edge_fraction = _threshold(edge_coverage, "edge_coverage")
    report_fraction = _threshold(report_coverage, "report_coverage")
    if report_fraction > edge_fraction:
        raise ValueError("report_coverage must not exceed edge_coverage")
    if (isinstance(min_alignment_length, bool) or not isinstance(min_alignment_length, Integral)
            or min_alignment_length < 1):
        raise ValueError("min_alignment_length must be a positive integer")
    if isinstance(max_evalue, bool) or not math.isfinite(max_evalue) or max_evalue < 0:
        raise ValueError("max_evalue must be finite and nonnegative")
    for node, length in sequence_lengths.items():
        if (not isinstance(node, str) or not node or isinstance(length, bool)
                or not isinstance(length, Integral) or length <= 0):
            raise ValueError(f"Invalid sequence length entry: {node!r}={length!r}")
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    accumulated = {}
    reported_queries, self_hit_queries = set(), set()
    statistics = {
        "raw_hsp_count": 0, "self_hsp_count": 0, "filtered_hsp_count": 0,
        "qualifying_nonself_hsp_count": 0,
        "coverage_method": "BTOP paired-base interval union; gap columns excluded on both sides",
        "identity_threshold": identity_threshold,
        "min_alignment_length": min_alignment_length, "max_evalue": max_evalue,
        "edge_coverage": edge_coverage, "report_coverage": report_coverage,
        "equal_length_policy": "minimum of the two coverages",
    }
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith("#"):
                continue
            try:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != len(BLAST_FIELDS):
                    raise ValueError(f"Expected {len(BLAST_FIELDS)} tab-separated fields, got {len(fields)}")
                qid, sid = fields[:2]
                if qid not in sequence_lengths or sid not in sequence_lengths:
                    raise ValueError(f"Unknown sequence ID: {qid!r} or {sid!r}")
                qlen, slen, nident, length, qstart, qend, sstart, send = map(int, fields[2:10])
                evalue, bitscore = map(float, fields[10:12])
                gaps = int(fields[12])
                if qlen != sequence_lengths[qid] or slen != sequence_lengths[sid]:
                    raise ValueError("Reported sequence lengths do not match FASTA lengths")
                if (length <= 0 or nident < 0 or gaps < 0 or nident + gaps > length
                        or not math.isfinite(evalue) or evalue < 0
                        or not math.isfinite(bitscore) or bitscore < 0):
                    raise ValueError("Invalid alignment count, E-value, or bit score")
                if not (1 <= qstart <= qlen and 1 <= qend <= qlen
                        and 1 <= sstart <= slen and 1 <= send <= slen):
                    raise ValueError("Alignment coordinates are outside sequence bounds")
                qintervals, sintervals = btop_intervals(
                    fields[13], qstart, qend, sstart, send, length, nident, gaps)
                statistics["raw_hsp_count"] += 1
                reported_queries.add(qid)
                if qid == sid:
                    statistics["self_hsp_count"] += 1
                    self_hit_queries.add(qid)
                    continue
                if (length < min_alignment_length or evalue > max_evalue
                        or not _at_least(nident, length, identity_fraction)):
                    statistics["filtered_hsp_count"] += 1
                    continue
                statistics["qualifying_nonself_hsp_count"] += 1
                a, b = sorted((qid, sid))
                pair = accumulated.setdefault((a, b), _Pair())
                if qid == a:
                    a_intervals, b_intervals, direction = qintervals, sintervals, 1
                else:
                    a_intervals, b_intervals, direction = sintervals, qintervals, 2
                pair.add(a_intervals, b_intervals, direction, nident / length, evalue, bitscore)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

    rows = []
    for (a, b), pair in sorted(accumulated.items()):
        alen, blen = sequence_lengths[a], sequence_lengths[b]
        acov, bcov = union_length(pair.intervals_a), union_length(pair.intervals_b)
        if alen < blen:
            short_cov, short_len, long_cov, long_len = acov, alen, bcov, blen
        elif blen < alen:
            short_cov, short_len, long_cov, long_len = bcov, blen, acov, alen
        else:
            short_cov, short_len = min(acov, bcov), alen
            long_cov, long_len = max(acov, bcov), alen
        if not _at_least(short_cov, short_len, report_fraction):
            continue
        rows.append({
            "node_a": a, "node_b": b, "length_a": alen, "length_b": blen,
            "covered_a_bp": acov, "covered_b_bp": bcov,
            "coverage_a": acov / alen, "coverage_b": bcov / blen,
            "shorter_coverage": short_cov / short_len, "longer_coverage": long_cov / long_len,
            "qualifying_hsp_count": pair.hsp_count,
            "query_direction_count": pair.direction_bits.bit_count(),
            "min_hsp_identity": pair.min_identity, "max_hsp_identity": pair.max_identity,
            "min_evalue": pair.min_evalue, "max_bitscore": pair.max_bitscore,
            "is_edge": _at_least(short_cov, short_len, edge_fraction),
        })
    edges = [row for row in rows if row["is_edge"]]
    statistics.update({"pairs_with_qualifying_hsps": len(accumulated),
                       "diagnostic_pair_count": len(rows), "edge_count": len(edges),
                       "expected_query_count": len(sequence_lengths),
                       "reported_query_count": len(reported_queries),
                       "reported_query_ids": sorted(reported_queries),
                       "queries_without_reported_hits": sorted(set(sequence_lengths) - reported_queries),
                       "self_hit_query_count": len(self_hit_queries),
                       "self_hit_query_ids": sorted(self_hit_queries),
                       "queries_without_reported_self_hits": sorted(set(sequence_lengths) - self_hit_queries),
                       "query_reporting_scope": (
                           "IDs in raw HSP records before additional HSP filtering; outfmt 6 does not "
                           "record zero-hit queries. An absent ID means no reported HSP, not proof "
                           "that its search was skipped or that all homology is absent.")})
    return {"pairs": rows, "edges": edges, "statistics": statistics}
