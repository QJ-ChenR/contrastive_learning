"""Read-only GenBank inventory for whole-record DNA encoder inputs.

Sequences are uppercased for sequence hashes and statistics, following the
existing DNA input policy. File hashes always describe the original bytes.
No records are removed, repaired, reverse-complemented in place, or written.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import statistics
import warnings

from Bio import SeqIO
from Bio.Seq import Seq


DNA_ALPHABET = frozenset("ACGTRYSWKMBDHVN")
GBK_SUFFIXES = frozenset({".gb", ".gbk", ".genbank"})


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _quantile(values, fraction):
    """Linear interpolation at (n - 1) * fraction, for an already sorted list."""
    index = (len(values) - 1) * fraction
    lower = int(index)
    if lower + 1 == len(values):
        return values[lower]
    return values[lower] + (values[lower + 1] - values[lower]) * (index - lower)


def _length_stats(values):
    values = sorted(values)
    if not values:
        return {
            "count": 0, "total": 0, "min": None, "p25": None,
            "median": None, "p75": None, "p90": None, "p95": None,
            "p99": None, "max": None, "mean": None,
        }
    return {
        "count": len(values),
        "total": sum(values),
        "min": values[0],
        "p25": _quantile(values, 0.25),
        "median": statistics.median(values),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": values[-1],
        "mean": statistics.mean(values),
    }


def load_dna(directory) -> dict:
    """Return ``records``, ``issues``, and ``stats`` without writing files.

    Each .gb/.gbk/.genbank file produces one row, including unreadable or invalid
    files. Only valid, nonempty, single-record DNA is marked ``available``.
    Identity conflicts and exact/RC duplicates are reported separately: they do
    not silently remove an otherwise readable DNA input. Paths retain the
    supplied directory prefix so the caller may relativize them consistently.

    GC uses (G + C) / (A + C + G + T); ambiguous symbols are excluded from its
    denominator. Length and composition summary statistics cover available rows.
    The absence of exact/RC duplicates is not a near-homology clearance.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"GenBank directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a GenBank directory: {directory}")

    paths = sorted(
        (path for path in directory.rglob("*")
         if path.is_file() and path.suffix.lower() in GBK_SUFFIXES),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    records = []
    issues = []
    row_codes = defaultdict(set)
    ambiguous_counts = Counter()

    def add_issue(row, code, severity, evidence):
        row_codes[row["gbk_path"]].add(code)
        issues.append({
            "entity_type": "bgc",
            "entity_id": row["bgc_id"],
            "issue_code": code,
            "severity": severity,
            "evidence_json": _json({"gbk_path": row["gbk_path"], **evidence}),
        })

    for path in paths:
        row = {
            "bgc_id": path.stem,
            "gbk_path": str(path),
            "gbk_sha256": "",
            "gbk_size_bytes": None,
            "record_count": None,
            "record_id": "",
            "record_name": "",
            "dna_status": "parse_error",
            "dna_length_bp": None,
            "dna_sha256": "",
            "dna_rc_canonical_sha256": "",
            "gc_fraction": None,
            "n_count": None,
            "ambiguous_base_count": None,
            "organism": "",
            "taxonomy_json": "[]",
            "issue_codes_json": "[]",
        }
        records.append(row)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            row["dna_status"] = "read_error"
            add_issue(row, "gbk_read_error", "error", {
                "exception_type": type(exc).__name__, "message": str(exc),
            })
            continue
        row["gbk_sha256"] = sha256(raw).hexdigest()
        row["gbk_size_bytes"] = len(raw)
        caught_warnings = []
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                parsed = list(SeqIO.parse(StringIO(raw.decode("utf-8")), "genbank"))
        except Exception as exc:
            if isinstance(exc, MemoryError):
                raise
            add_issue(row, "gbk_parse_error", "error", {
                "exception_type": type(exc).__name__, "message": str(exc),
            })
            continue
        finally:
            for warning in caught_warnings:
                add_issue(row, "gbk_parser_warning", "warning", {
                    "warning_type": warning.category.__name__,
                    "message": str(warning.message),
                })
        row["record_count"] = len(parsed)
        if len(parsed) != 1:
            row["dna_status"] = "no_records" if not parsed else "multiple_records"
            add_issue(row, f"gbk_{row['dna_status']}", "error", {
                "record_count": len(parsed),
                "record_ids": [record.id for record in parsed],
            })
            continue

        record = parsed[0]
        row.update({
            "record_id": record.id,
            "record_name": record.name,
            "organism": record.annotations.get("organism", ""),
            "taxonomy_json": _json(record.annotations.get("taxonomy", [])),
            "dna_length_bp": len(record.seq),
        })
        for field, actual in (("record_id", record.id), ("record_name", record.name)):
            if actual != path.stem:
                add_issue(row, f"gbk_{field}_mismatch", "warning", {
                    "filename_id": path.stem, field: actual,
                })
        accessions = record.annotations.get("accessions", [])
        if accessions and accessions != [path.stem]:
            add_issue(row, "gbk_accession_mismatch", "warning", {
                "filename_id": path.stem, "accessions": accessions,
            })
        try:
            sequence = str(record.seq).upper()
        except Exception as exc:
            if isinstance(exc, MemoryError):
                raise
            row["dna_status"] = "invalid_sequence"
            add_issue(row, "dna_sequence_unreadable", "error", {
                "exception_type": type(exc).__name__, "message": str(exc),
                "declared_length_bp": len(record.seq),
            })
            continue

        counts = Counter(sequence)
        acgt_count = sum(counts[base] for base in "ACGT")
        row.update({
            "dna_sha256": sha256(sequence.encode("utf-8")).hexdigest(),
            "gc_fraction": (counts["G"] + counts["C"]) / acgt_count if acgt_count else None,
            "n_count": counts["N"],
            "ambiguous_base_count": sum(count for base, count in counts.items() if base not in "ACGT"),
        })
        invalid = sorted(set(sequence) - DNA_ALPHABET)
        if not sequence or invalid:
            row["dna_status"] = "invalid_sequence"
            code = "dna_empty_sequence" if not sequence else "dna_invalid_alphabet"
            add_issue(row, code, "error", {
                "length_bp": len(sequence),
                "invalid_character_counts": {base: counts[base] for base in invalid},
            })
            continue

        reverse_complement = str(Seq(sequence).reverse_complement())
        row["dna_rc_canonical_sha256"] = sha256(
            min(sequence, reverse_complement).encode("ascii")
        ).hexdigest()
        row["dna_status"] = "available"
        ambiguous_counts.update({base: count for base, count in counts.items() if base not in "ACGT"})
        if acgt_count == 0:
            add_issue(row, "dna_gc_undefined", "warning", {
                "acgt_base_count": 0,
                "message": "GC fraction is undefined because the A/C/G/T denominator is zero.",
            })

    id_groups = defaultdict(list)
    exact_groups = defaultdict(list)
    rc_groups = defaultdict(list)
    for row in records:
        id_groups[row["bgc_id"]].append(row)
        if row["dna_status"] == "available":
            exact_groups[row["dna_sha256"]].append(row)
            rc_groups[row["dna_rc_canonical_sha256"]].append(row)

    for group_id, members in sorted(id_groups.items()):
        if len(members) > 1:
            for row in members:
                add_issue(row, "duplicate_bgc_id", "error", {
                    "bgc_id": group_id,
                    "gbk_paths": [member["gbk_path"] for member in members],
                })

    duplicate_sets = []
    for groups, code, entity_type in (
        (exact_groups, "duplicate_dna_sequence", "dna_sequence"),
        (rc_groups, "duplicate_dna_or_reverse_complement", "dna_rc_sequence"),
    ):
        duplicates = {key: members for key, members in groups.items() if len(members) > 1}
        duplicate_sets.append(duplicates)
        for digest, members in sorted(duplicates.items()):
            for member in members:
                row_codes[member["gbk_path"]].add(code)
            issues.append({
                "entity_type": entity_type,
                "entity_id": digest,
                "issue_code": code,
                "severity": "warning",
                "evidence_json": _json({
                    "members": [{
                        "bgc_id": member["bgc_id"], "gbk_path": member["gbk_path"],
                        "dna_sha256": member["dna_sha256"],
                    } for member in members],
                    "member_count": len(members),
                    "distinct_forward_sequence_count": len({member["dna_sha256"] for member in members}),
                }),
            })

    for row in records:
        row["issue_codes_json"] = _json(sorted(row_codes[row["gbk_path"]]))
    issues.sort(key=lambda issue: (
        issue["entity_type"], issue["entity_id"], issue["issue_code"], issue["evidence_json"],
    ))
    available = [row for row in records if row["dna_status"] == "available"]
    exact_duplicates, rc_duplicates = duplicate_sets
    stats = {
        "file_count": len(records),
        "available_count": len(available),
        "failed_count": len(records) - len(available),
        "status_counts": dict(sorted(Counter(row["dna_status"] for row in records).items())),
        "unique_bgc_id_count": len(id_groups),
        "duplicate_bgc_id_group_count": sum(len(members) > 1 for members in id_groups.values()),
        "unique_dna_sha256_count": len(exact_groups),
        "unique_dna_rc_canonical_sha256_count": len(rc_groups),
        "dna_exact_duplicate_group_count": len(exact_duplicates),
        "dna_exact_duplicate_record_count": sum(len(members) for members in exact_duplicates.values()),
        "dna_exact_duplicate_excess_count": sum(len(members) - 1 for members in exact_duplicates.values()),
        "dna_rc_duplicate_group_count": len(rc_duplicates),
        "dna_rc_duplicate_record_count": sum(len(members) for members in rc_duplicates.values()),
        "dna_rc_duplicate_excess_count": sum(len(members) - 1 for members in rc_duplicates.values()),
        "dna_rc_only_duplicate_group_count": sum(
            len({member["dna_sha256"] for member in members}) > 1
            for members in rc_duplicates.values()
        ),
        "length_bp": _length_stats([row["dna_length_bp"] for row in available]),
        "n_bgc_count": sum(row["n_count"] > 0 for row in available),
        "n_base_count": sum(row["n_count"] for row in available),
        "ambiguous_bgc_count": sum(row["ambiguous_base_count"] > 0 for row in available),
        "ambiguous_base_count": sum(row["ambiguous_base_count"] for row in available),
        "ambiguous_base_counts": dict(sorted(ambiguous_counts.items())),
        "gc_undefined_count": sum(row["gc_fraction"] is None for row in available),
        "gc_fraction_definition": "(G+C)/(A+C+G+T); ambiguous bases excluded; null when denominator is zero",
        "sequence_normalization": "uppercase; no deletion, truncation, or orientation change",
        "sequence_statistics_scope": "available single-record DNA only; duplicates retained",
        "length_quantile_method": "linear interpolation at (n-1)*q",
        "issue_count": len(issues),
        "issue_code_counts": dict(sorted(Counter(issue["issue_code"] for issue in issues).items())),
    }
    return {"records": records, "issues": issues, "stats": stats}
