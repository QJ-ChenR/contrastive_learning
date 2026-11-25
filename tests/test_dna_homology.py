"""Numerical and coordinate regressions for DNA near-homology evidence."""

import gzip
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dna_homology import aggregate_blast, btop_intervals, merge_intervals, union_length


def hsp(qid="A", sid="B", qlen=1000, slen=2000, qstart=1, qend=800,
        sstart=501, send=1300, nident=None, length=None, evalue="1e-10",
        bitscore=100, gaps=0, btop=None):
    length = abs(qend - qstart) + 1 if length is None else length
    nident = length if nident is None else nident
    btop = str(nident) + "AC" * (length - nident) if btop is None else btop
    return "\t".join(map(str, [qid, sid, qlen, slen, nident, length,
                                  qstart, qend, sstart, send,
                                  evalue, bitscore, gaps, btop]))


class DNAHomologyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dna-homology-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_rows(self, rows, lengths=None, compressed=False, **kwargs):
        path = self.root / ("hits.tsv.gz" if compressed else "hits.tsv")
        content = "\n".join(rows) + "\n"
        if compressed:
            with gzip.open(path, "wt") as stream:
                stream.write(content)
        else:
            path.write_text(content)
        return aggregate_blast(path, lengths or {"A": 1000, "B": 2000}, **kwargs)

    def test_identity_coverage_and_evalue_boundaries_are_inclusive(self):
        result = self.run_rows([hsp(nident=640)])
        self.assertEqual(len(result["edges"]), 1)
        edge = result["edges"][0]
        self.assertEqual(edge["shorter_coverage"], 0.8)
        self.assertEqual(edge["longer_coverage"], 0.4)
        self.assertEqual(edge["min_hsp_identity"], 0.8)
        self.assertEqual(edge["min_evalue"], 1e-10)
        self.assertEqual(self.run_rows([hsp(nident=639)])["edges"], [])
        self.assertEqual(self.run_rows([hsp(evalue="1.0001e-10")])["edges"], [])
        short = self.run_rows([hsp(qend=799, send=1299)])
        self.assertEqual(len(short["pairs"]), 1)
        self.assertEqual(short["edges"], [])

    def test_minimum_hsp_length_boundary(self):
        lengths = {"A": 125, "B": 200}
        accepted = hsp(qlen=125, slen=200, qend=100, sstart=1, send=100)
        rejected = hsp(qlen=125, slen=200, qend=99, sstart=1, send=99)
        self.assertEqual(len(self.run_rows([accepted], lengths)["edges"]), 1)
        self.assertEqual(self.run_rows([rejected], lengths)["pairs"], [])

    def test_overlap_and_reciprocal_hsps_do_not_inflate_coverage(self):
        rows = [
            hsp(qstart=1, qend=600, sstart=1, send=600),
            hsp(qstart=201, qend=800, sstart=201, send=800),
            hsp(qid="B", sid="A", qlen=2000, slen=1000,
                qstart=1, qend=600, sstart=1, send=600),
        ]
        edge = self.run_rows(rows)["edges"][0]
        self.assertEqual((edge["covered_a_bp"], edge["covered_b_bp"]), (800, 800))
        self.assertEqual(edge["qualifying_hsp_count"], 3)
        self.assertEqual(edge["query_direction_count"], 2)
        self.assertEqual(self.run_rows(list(reversed(rows)))["edges"], [edge])

    def test_reverse_strand_and_reciprocal_orientation(self):
        rows = [hsp(sstart=1300, send=501),
                hsp(qid="B", sid="A", qlen=2000, slen=1000,
                    qstart=1300, qend=501, sstart=1, send=800)]
        edge = self.run_rows(rows)["edges"][0]
        self.assertEqual((edge["covered_a_bp"], edge["covered_b_bp"]), (800, 800))
        qintervals, sintervals = btop_intervals("800", 1, 800, 1300, 501, 800, 800, 0)
        self.assertEqual(qintervals, [(0, 800)])
        self.assertEqual(sintervals, [(500, 1300)])

    def test_gap_columns_do_not_inflate_either_coverage(self):
        # Each sequence consumes 900 positions, but only 800 align to a base.
        row = hsp(slen=1000, qend=900, sstart=1, send=900,
                  nident=800, length=1000, gaps=200,
                  btop="400" + "A-" * 100 + "-C" * 100 + "400")
        result = self.run_rows([row], {"A": 1000, "B": 1000}, edge_coverage=0.85)
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["pairs"][0]["covered_a_bp"], 800)
        self.assertEqual(result["pairs"][0]["covered_b_bp"], 800)

    def test_equal_length_uses_smaller_coverage(self):
        # Repeated A segment maps onto two B segments; no artificial full A.
        rows = [hsp(slen=1000, qend=500, sstart=1, send=500),
                hsp(slen=1000, qend=500, sstart=501, send=1000)]
        result = self.run_rows(rows, {"A": 1000, "B": 1000})
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["pairs"][0]["shorter_coverage"], 0.5)
        self.assertEqual(result["pairs"][0]["longer_coverage"], 1.0)

    def test_short_shared_region_repeated_across_target_does_not_make_edge(self):
        rows = [hsp(qend=100, sstart=start, send=start + 99)
                for start in range(1, 1902, 100)]
        result = self.run_rows(rows)
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["statistics"]["pairs_with_qualifying_hsps"], 1)

    def test_self_hits_are_not_edges_and_gzip_is_supported(self):
        row = hsp(qid="A", sid="A", slen=1000, qend=1000, sstart=1, send=1000)
        result = self.run_rows([row, hsp()], compressed=True)
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["statistics"]["self_hsp_count"], 1)
        self.assertEqual(result["statistics"]["raw_hsp_count"], 2)
        self.assertEqual(result["statistics"]["reported_query_ids"], ["A"])
        self.assertEqual(result["statistics"]["self_hit_query_ids"], ["A"])
        self.assertEqual(result["statistics"]["queries_without_reported_hits"], ["B"])
        self.assertEqual(result["statistics"]["queries_without_reported_self_hits"], ["B"])

    def test_query_reporting_precedes_additional_hsp_filtering(self):
        result = self.run_rows([hsp(nident=639)])
        self.assertEqual(result["statistics"]["reported_query_count"], 1)
        self.assertEqual(result["statistics"]["filtered_hsp_count"], 1)
        self.assertEqual(result["statistics"]["self_hit_query_ids"], [])
        self.assertEqual(result["statistics"]["queries_without_reported_self_hits"], ["A", "B"])

    def test_empty_hits_are_valid(self):
        result = self.run_rows([])
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["statistics"]["raw_hsp_count"], 0)
        self.assertEqual(result["statistics"]["expected_query_count"], 2)
        self.assertEqual(result["statistics"]["queries_without_reported_hits"], ["A", "B"])

    def test_union_half_open_adjacent_intervals_and_invalid_bounds(self):
        self.assertEqual(merge_intervals([(10, 20), (0, 10), (3, 7), (25, 30)]),
                         [(0, 20), (25, 30)])
        self.assertEqual(union_length([(10, 20), (0, 15)]), 20)
        with self.assertRaises(ValueError):
            union_length([(-1, 4)])

    def test_malformed_hits_fail_with_line_number(self):
        bad_rows = [
            "A\tB", hsp(qid="unknown"), hsp(qlen=999), hsp(qstart=0),
            hsp(send=2001), hsp(nident=801), hsp(evalue="nan"),
            hsp(bitscore="inf"), hsp(btop="799"), hsp(btop="800!"),
            hsp(btop="800--"), hsp(gaps=1), hsp(length=0, nident=0, btop="0"),
        ]
        for row in bad_rows:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, r"hits.tsv:1:"):
                    self.run_rows([row])

    def test_invalid_configuration_is_rejected(self):
        for kwargs in [dict(identity_threshold=0), dict(edge_coverage=1.1),
                       dict(report_coverage=0.9), dict(min_alignment_length=0),
                       dict(min_alignment_length=True), dict(max_evalue=float("nan"))]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.run_rows([], **kwargs)


if __name__ == "__main__":
    unittest.main()
