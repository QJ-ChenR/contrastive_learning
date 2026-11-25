"""Read product and scaffold source tables without changing their contents.

This module uses only the Python standard library.  Formula comparisons are
field-consistency checks, not chemical validation or structure normalization.
Both public functions return in-memory records and never write output files.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_PRODUCT_COLUMNS = {
    "npaid", "compound_names", "compound_smiles", "compound_inchi",
    "compound_inchikey", "compound_molecular_formula",
    "compound_molecular_weight", "npatlas_url", "mibig_ids",
}
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")
_ELEMENTS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
    "Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
    "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf "
    "Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
_BGC_ID = re.compile(r"BGC[0-9]{7}")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _issue(entity_type: str, entity_id: str, code: str,
           severity: str, evidence: dict[str, Any]) -> dict[str, str]:
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "issue_code": code,
        "severity": severity,
        "evidence_json": _json(evidence),
    }


def _read_table(path: str | Path, delimiter: str) -> list[tuple[int, list[str]]]:
    """Keep physical starting line numbers, including for quoted multiline cells."""
    records = []
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter, strict=True)
            while True:
                source_line = reader.line_num + 1
                try:
                    row = next(reader)
                except StopIteration:
                    break
                records.append((source_line, row))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    return records


def _simple_formula(value: str) -> dict[str, int] | None:
    """Accept only simple, uncharged element-count formulas.

    Component separators, isotope syntax, brackets, charges and other formula
    conventions are deliberately unsupported; they must not become mismatches.
    """
    value = value.strip()
    if not value:
        return None
    counts: Counter[str] = Counter()
    position = 0
    for match in _FORMULA_TOKEN.finditer(value):
        if match.start() != position or match[1] not in _ELEMENTS:
            return None
        counts[match[1]] += int(match[2] or "1")
        position = match.end()
    if position != len(value):
        return None
    return dict(sorted(counts.items()))


def _formula_check(formula: str, inchi: str) -> tuple[str, dict[str, Any]]:
    evidence: dict[str, Any] = {
        "compound_molecular_formula": formula,
        "scope": "source formula field versus InChI formula layer only; "
                 "no RDKit, charge normalization, or biological validation",
    }
    layers = inchi.strip().split("/")
    if len(layers) < 2 or layers[0] not in {"InChI=1", "InChI=1S"}:
        evidence["not_checked_reason"] = "missing_or_unsupported_inchi_prefix"
        return "not_checked", evidence
    evidence["inchi_formula_layer"] = layers[1]
    if "AuxInfo=" in inchi or any(
        layer.startswith(("q", "p", "i", "f", "r")) for layer in layers[2:]
    ):
        evidence["not_checked_reason"] = "charge_isotope_or_other_complex_inchi_layer"
        return "not_checked", evidence
    formula_counts = _simple_formula(formula)
    inchi_counts = _simple_formula(layers[1])
    if formula_counts is None or inchi_counts is None:
        evidence["not_checked_reason"] = "unsupported_simple_formula_syntax"
        return "not_checked", evidence
    evidence["formula_element_counts"] = formula_counts
    evidence["inchi_element_counts"] = inchi_counts
    return ("match" if formula_counts == inchi_counts else "mismatch"), evidence


def load_products(path: str | Path) -> dict[str, Any]:
    """Return linked product records, all their BGC associations, and audit data.

    Product IDs are checked for uniqueness across the entire source table,
    including unlinked records.  Original field strings are preserved.  The
    additional ``formula_consistency_status`` field is ``match``, ``mismatch``
    or ``not_checked``; none of these statuses validates a SMILES molecule.
    ``mibig_ids_json`` preserves the parsed ID list and its order, while
    ``mibig_ids_raw`` preserves the source cell verbatim.
    """
    table = _read_table(path, "\t")
    if not table:
        raise ValueError(f"Empty product table: {path}")
    header = table[0][1]
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate column names in product table: {path}")
    missing = sorted(_PRODUCT_COLUMNS - set(header))
    if missing:
        raise ValueError(f"Missing product columns in {path}: {missing}")

    all_products: list[dict[str, Any]] = []
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: dict[str, int] = {}
    issues: list[dict[str, str]] = []
    control_rows = 0
    form_feed_count = 0

    for source_line, values in table[1:]:
        if len(values) != len(header):
            raise ValueError(
                f"{path}:{source_line}: expected {len(header)} columns, got {len(values)}"
            )
        row = dict(zip(header, values))
        product_id = row["npaid"]
        if not product_id.strip():
            raise ValueError(f"{path}:{source_line}: empty product ID")
        if product_id in seen_ids:
            raise ValueError(
                f"{path}:{source_line}: duplicate product ID {product_id!r}; "
                f"first occurrence at line {seen_ids[product_id]}"
            )
        seen_ids[product_id] = source_line
        try:
            bgc_ids = ast.literal_eval(row["mibig_ids"])
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{path}:{source_line}: invalid mibig_ids list") from exc
        if not isinstance(bgc_ids, list) or any(
            not isinstance(bgc, str) or not _BGC_ID.fullmatch(bgc) for bgc in bgc_ids
        ):
            raise ValueError(f"{path}:{source_line}: invalid MIBiG identifiers")
        if len(bgc_ids) != len(set(bgc_ids)):
            raise ValueError(f"{path}:{source_line}: duplicate IDs within mibig_ids")

        status, formula_evidence = _formula_check(
            row["compound_molecular_formula"], row["compound_inchi"]
        )
        product = {
            "product_id": product_id,
            "source_line": source_line,
            "compound_name": row["compound_names"],
            "smiles": row["compound_smiles"],
            "inchi": row["compound_inchi"],
            "inchikey": row["compound_inchikey"],
            "molecular_formula": row["compound_molecular_formula"],
            "molecular_weight": row["compound_molecular_weight"],
            "npatlas_url": row["npatlas_url"],
            "mibig_ids_raw": row["mibig_ids"],
            "mibig_ids_json": _json(bgc_ids),
            "formula_consistency_status": status,
            "issue_codes_json": "[]",
        }
        internal = {"product": product, "bgc_ids": bgc_ids,
                    "formula_evidence": formula_evidence, "issue_codes": []}
        all_products.append(internal)
        if product["inchikey"]:
            by_key[product["inchikey"]].append(internal)

        controls = Counter(char for char in product["compound_name"] if ord(char) < 32)
        if controls:
            control_rows += 1
            form_feed_count += controls.get("\x0c", 0)
            code = "source_name_control_character"
            internal["issue_codes"].append(code)
            issues.append(_issue("source_product", product_id, code, "info", {
                "source_line": source_line,
                "field": "compound_names",
                "characters": {f"U+{ord(char):04X}": n for char, n in controls.items()},
                "has_bgc_association": bool(bgc_ids),
                "handling": "preserved; parsed with csv.reader rather than str.splitlines",
            }))

    products = []
    associations = []
    formula_statuses: Counter[str] = Counter()
    not_checked_reasons: Counter[str] = Counter()
    for internal in all_products:
        if not internal["bgc_ids"]:
            continue
        product = internal["product"]
        product_id = product["product_id"]
        status = product["formula_consistency_status"]
        formula_statuses[status] += 1
        if status == "not_checked":
            not_checked_reasons[internal["formula_evidence"]["not_checked_reason"]] += 1
        if status == "mismatch":
            code = "source_formula_inchi_formula_mismatch"
            internal["issue_codes"].append(code)
            peers = []
            for other in by_key.get(product["inchikey"], []):
                peer = other["product"]
                if peer["product_id"] == product_id:
                    continue
                peers.append({
                    "product_id": peer["product_id"],
                    "source_line": peer["source_line"],
                    "compound_name": peer["compound_name"],
                    "molecular_formula": peer["molecular_formula"],
                    "formula_consistency_status": peer["formula_consistency_status"],
                    "has_bgc_association": bool(other["bgc_ids"]),
                    "same_inchi": peer["inchi"] == product["inchi"],
                    "same_smiles": peer["smiles"] == product["smiles"],
                    "same_molecular_weight": peer["molecular_weight"] == product["molecular_weight"],
                    "same_npatlas_url": peer["npatlas_url"] == product["npatlas_url"],
                })
            issues.append(_issue("product", product_id, code, "warning", {
                **internal["formula_evidence"],
                "source_line": product["source_line"],
                "inchikey": product["inchikey"],
                "shared_inchikey_source_records": peers,
                "handling": "flagged for review; all original fields and associations retained",
            }))
        product["issue_codes_json"] = _json(internal["issue_codes"])
        products.append(product)
        associations.extend({
            "bgc_id": bgc,
            "product_id": product_id,
            "source_line": product["source_line"],
        } for bgc in internal["bgc_ids"])

    bgc_counts = Counter(row["bgc_id"] for row in associations)
    product_bgc_counts = Counter(row["product_id"] for row in associations)
    stats = {
        "source_record_count": len(all_products),
        "source_column_count": len(header),
        "source_unique_product_count": len(seen_ids),
        "unlinked_product_count": len(all_products) - len(products),
        "linked_product_count": len(products),
        "association_count": len(associations),
        "unique_bgc_count": len(bgc_counts),
        "unique_linked_smiles_string_count": len({row["smiles"] for row in products}),
        "linked_products_with_empty_smiles": sum(not row["smiles"].strip() for row in products),
        "products_per_bgc_distribution": dict(sorted(Counter(bgc_counts.values()).items())),
        "bgcs_per_product_distribution": dict(sorted(Counter(product_bgc_counts.values()).items())),
        "multi_product_bgc_count": sum(n > 1 for n in bgc_counts.values()),
        "multi_bgc_product_count": sum(n > 1 for n in product_bgc_counts.values()),
        "formula_consistency_status_counts": dict(sorted(formula_statuses.items())),
        "formula_not_checked_reason_counts": dict(sorted(not_checked_reasons.items())),
        "source_name_control_character_rows": control_rows,
        "source_name_form_feed_count": form_feed_count,
        "issue_count": len(issues),
        "issue_code_counts": dict(sorted(Counter(row["issue_code"] for row in issues).items())),
        "chemical_validation_performed": False,
    }
    return {"products": products, "associations": associations,
            "issues": issues, "stats": stats}


def load_scaffolds(path: str | Path) -> dict[str, Any]:
    """Read every row of the headerless BGC/scaffold association table.

    Empty strings and duplicate rows are preserved.  Repeated rows can result
    from different products sharing a scaffold; they are informational findings,
    not grounds for deleting products or assigning one scaffold label per BGC.
    """
    rows = []
    by_bgc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_line, values in _read_table(path, ","):
        if len(values) != 2:
            raise ValueError(f"{path}:{source_line}: expected 2 scaffold columns, got {len(values)}")
        bgc, scaffold = values
        if not _BGC_ID.fullmatch(bgc):
            raise ValueError(f"{path}:{source_line}: invalid BGC ID {bgc!r}; expected no header")
        row = {"bgc_id": bgc, "scaffold_smiles": scaffold,
               "source_line": source_line, "is_empty": scaffold == ""}
        rows.append(row)
        by_bgc[bgc].append(row)

    issues = []
    distinct_counts: Counter[int] = Counter()
    nonempty_counts: Counter[int] = Counter()
    duplicate_groups = 0
    for bgc, bgc_rows in by_bgc.items():
        value_lines: dict[str, list[int]] = defaultdict(list)
        for row in bgc_rows:
            value_lines[row["scaffold_smiles"]].append(row["source_line"])
        distinct_counts[len(value_lines)] += 1
        nonempty_counts[len(set(value_lines) - {""})] += 1
        if "" in value_lines:
            issues.append(_issue("bgc", bgc, "empty_scaffold_value", "info", {
                "source_lines": value_lines[""],
                "only_empty_scaffolds": len(value_lines) == 1,
                "handling": "preserved; not interpreted as a shared chemical class",
            }))
        if len(value_lines) > 1:
            issues.append(_issue("bgc", bgc, "multiple_scaffold_values", "info", {
                "distinct_value_count_including_empty": len(value_lines),
                "distinct_nonempty_value_count": len(set(value_lines) - {""}),
                "source_lines": [row["source_line"] for row in bgc_rows],
                "handling": "all values retained; no single-label assignment",
            }))
        for scaffold, lines in value_lines.items():
            if len(lines) > 1:
                duplicate_groups += 1
                issues.append(_issue("bgc", bgc, "repeated_bgc_scaffold_pair", "info", {
                    "scaffold_smiles": scaffold,
                    "source_lines": lines,
                    "excess_row_count": len(lines) - 1,
                    "handling": "preserved; multiple products can share the same scaffold",
                }))

    unique_pairs = {(row["bgc_id"], row["scaffold_smiles"]) for row in rows}
    unique_values = {row["scaffold_smiles"] for row in rows}
    stats = {
        "row_count": len(rows),
        "has_header": False,
        "unique_bgc_count": len(by_bgc),
        "unique_bgc_scaffold_pair_count": len(unique_pairs),
        "unique_scaffold_count_including_empty": len(unique_values),
        "unique_nonempty_scaffold_count": len(unique_values - {""}),
        "empty_row_count": sum(row["is_empty"] for row in rows),
        "bgcs_with_empty_scaffold_count": sum(any(row["is_empty"] for row in group) for group in by_bgc.values()),
        "bgcs_with_only_empty_scaffolds_count": sum(all(row["is_empty"] for row in group) for group in by_bgc.values()),
        "duplicate_row_excess_count": len(rows) - len(unique_pairs),
        "repeated_bgc_scaffold_pair_group_count": duplicate_groups,
        "distinct_scaffolds_per_bgc_distribution": dict(sorted(distinct_counts.items())),
        "distinct_nonempty_scaffolds_per_bgc_distribution": dict(sorted(nonempty_counts.items())),
        "multi_scaffold_bgc_count_including_empty": sum(n for k, n in distinct_counts.items() if k > 1),
        "multi_nonempty_scaffold_bgc_count": sum(n for k, n in nonempty_counts.items() if k > 1),
        "issue_count": len(issues),
        "issue_code_counts": dict(sorted(Counter(row["issue_code"] for row in issues).items())),
        "chemical_validation_performed": False,
    }
    return {"rows": rows, "issues": issues, "stats": stats}
