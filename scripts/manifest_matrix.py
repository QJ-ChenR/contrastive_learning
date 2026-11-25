"""Read and summarize BGC similarity matrices without generating pair files."""

from __future__ import annotations

import csv
import math
from numbers import Integral
from pathlib import Path
from typing import Iterable

import numpy as np


_MATRIX_ATOL = 1e-12


def _validate_ids(ids: list[str], description: str) -> None:
    if any(not identifier.strip() for identifier in ids):
        raise ValueError(f"{description} contains an empty BGC ID")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{description} contains duplicate BGC IDs")


def load_matrix(path: str | Path) -> dict:
    """Load the comma-delimited matrix, validating identities and numeric values.

    The first column holds row IDs; the remaining header fields hold column IDs.
    Rows are aligned to column IDs if their order differs. Symmetry and unit
    diagonal checks allow only an absolute floating-point tolerance of 1e-12.
    Invalid input raises ``ValueError``; this function never writes files.
    """
    matrix_path = Path(path)
    with matrix_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError(f"{matrix_path}: missing matrix header or BGC columns")
        ids = header[1:]
        _validate_ids(ids, "Matrix column IDs")
        rows: list[list[float]] = []
        row_ids: list[str] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{matrix_path}:{line_number}: expected {len(header)} "
                    f"CSV fields, found {len(row)}"
                )
            row_ids.append(row[0])
            try:
                rows.append([float(value) for value in row[1:]])
            except ValueError as exc:
                raise ValueError(
                    f"{matrix_path}:{line_number}: nonnumeric similarity value"
                ) from exc

    _validate_ids(row_ids, "Matrix row IDs")
    if len(rows) != len(ids):
        raise ValueError(
            f"{matrix_path}: matrix is not square "
            f"({len(rows)} rows, {len(ids)} columns)"
        )
    if set(row_ids) != set(ids):
        missing = sorted(set(ids) - set(row_ids))
        extra = sorted(set(row_ids) - set(ids))
        raise ValueError(
            f"{matrix_path}: row/column BGC IDs differ; "
            f"missing row IDs={missing}, extra row IDs={extra}"
        )

    values = np.asarray(rows, dtype=np.float64)
    order_identical = row_ids == ids
    if not order_identical:
        row_index = {identifier: index for index, identifier in enumerate(row_ids)}
        values = values[[row_index[identifier] for identifier in ids], :]

    nonfinite_count = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite_count:
        raise ValueError(f"{matrix_path}: {nonfinite_count} nonfinite similarities")
    out_of_range_count = int(np.count_nonzero((values < 0) | (values > 1)))
    if out_of_range_count:
        raise ValueError(
            f"{matrix_path}: {out_of_range_count} similarities outside [0, 1]"
        )
    max_asymmetry = float(np.max(np.abs(values - values.T)))
    if max_asymmetry > _MATRIX_ATOL:
        raise ValueError(
            f"{matrix_path}: matrix is asymmetric; maximum difference "
            f"{max_asymmetry} exceeds {_MATRIX_ATOL}"
        )
    diagonal = np.diag(values)
    max_diagonal_deviation = float(np.max(np.abs(diagonal - 1)))
    if max_diagonal_deviation > _MATRIX_ATOL:
        raise ValueError(
            f"{matrix_path}: diagonal differs from 1; maximum deviation "
            f"{max_diagonal_deviation} exceeds {_MATRIX_ATOL}"
        )

    stats = {
        "n_rows": len(row_ids),
        "n_columns": len(ids),
        "unique_row_ids": len(set(row_ids)),
        "unique_column_ids": len(set(ids)),
        "row_column_order_identical": order_identical,
        "rows_reordered_to_columns": not order_identical,
        "nonfinite_count": nonfinite_count,
        "out_of_range_count": out_of_range_count,
        "min_value": float(values.min()),
        "max_value": float(values.max()),
        "max_asymmetry": max_asymmetry,
        "max_diagonal_deviation": max_diagonal_deviation,
        "validation_absolute_tolerance": _MATRIX_ATOL,
    }
    return {"ids": ids, "values": values, "stats": stats}


def summarize_pairs(
    matrix: dict,
    indices: Iterable[int] | None = None,
    thresholds: Iterable[float] = (0.6, 0.65, 0.7, 0.75),
) -> dict:
    """Count unordered non-self pairs and per-BGC positive-neighbor degrees.

    ``indices`` selects unique matrix positions; ``None`` selects every BGC.
    Selected IDs follow the original matrix order, independent of index order.
    Positive pairs have S > threshold, and negative pairs have S <= threshold.
    Each pair is read once from the selected matrix's upper triangle. This also
    gives a deterministic convention for accepted sub-tolerance asymmetries.
    No pair list or training split is produced.
    """
    ids = matrix["ids"]
    values = np.asarray(matrix["values"])
    if values.shape != (len(ids), len(ids)):
        raise ValueError("Matrix values must be square and aligned to matrix IDs")

    if indices is None:
        selected_ids = list(ids)
        selected_values = values
    else:
        positions = list(indices)
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in positions
        ):
            raise ValueError("Matrix indices must be integers")
        positions = [int(index) for index in positions]
        if len(set(positions)) != len(positions):
            raise ValueError("Matrix indices must be unique")
        if any(index < 0 or index >= len(ids) for index in positions):
            raise ValueError("Matrix index outside valid range")
        positions.sort()
        selected_ids = [ids[index] for index in positions]
        selected_values = values[np.ix_(positions, positions)]

    try:
        threshold_values = [float(threshold) for threshold in thresholds]
    except (TypeError, ValueError) as exc:
        raise ValueError("Thresholds must be finite numbers in [0, 1]") from exc
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in threshold_values
    ):
        raise ValueError("Thresholds must be finite numbers in [0, 1]")
    if len(set(threshold_values)) != len(threshold_values):
        raise ValueError("Thresholds must be unique")

    bgc_count = len(selected_ids)
    pair_count = bgc_count * (bgc_count - 1) // 2
    summary = []
    degrees = {identifier: {} for identifier in selected_ids}
    for threshold in threshold_values:
        positive_pairs = np.triu(selected_values > threshold, k=1)
        positive_count = int(np.count_nonzero(positive_pairs))
        neighbor_counts = np.count_nonzero(positive_pairs, axis=0) + np.count_nonzero(
            positive_pairs, axis=1
        )
        covered_count = int(np.count_nonzero(neighbor_counts))
        summary.append(
            {
                "threshold": threshold,
                "comparison": ">",
                "bgc_count": bgc_count,
                "unordered_pair_count": pair_count,
                "positive_pair_count": positive_count,
                "negative_pair_count": pair_count - positive_count,
                "bgc_with_positive_neighbor": covered_count,
                "bgc_without_positive_neighbor": bgc_count - covered_count,
            }
        )
        threshold_key = str(threshold)
        for identifier, count in zip(selected_ids, neighbor_counts):
            degrees[identifier][threshold_key] = int(count)

    return {"summary": summary, "degrees": degrees}
