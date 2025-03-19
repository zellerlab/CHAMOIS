import unittest
import warnings

import gb_io

from chamois.model import ClusterSequence
from chamois.orf import ORFFinder, PyrodigalFinder, CDSFinder, NoGeneFoundWarning, IncompleteGeneWarning

from .utils import resource_files


class TestPyrodigalFinder(unittest.TestCase):
    
    def test_empty_sequence(self):
        record = gb_io.Record(sequence=b"", name="empty")
        seq = ClusterSequence(record)

        with warnings.catch_warnings(record=True) as ctx:
            orf_finder = PyrodigalFinder()
            genes = list(orf_finder.find_genes([seq]))

        self.assertEqual(len(ctx), 1)
        self.assertIs(ctx[0].category, NoGeneFoundWarning)


class TestCDSFinder(unittest.TestCase):
    
    def test_empty_sequence(self):
        record = gb_io.Record(sequence=b"", name="empty")
        seq = ClusterSequence(record)

        with warnings.catch_warnings(record=True) as ctx:
            orf_finder = CDSFinder()
            genes = list(orf_finder.find_genes([seq]))

        self.assertEqual(len(ctx), 1)
        self.assertIs(ctx[0].category, NoGeneFoundWarning)

    def test_incomplete_protein(self):
        record = gb_io.Record(
            sequence=b"ATTACCAGAATAGAATTAGAATAG", 
            name="test",
            features=[gb_io.Feature("CDS", location=gb_io.Range(1, 10))],
        )
        seq = ClusterSequence(record)

        with warnings.catch_warnings(record=True) as ctx:
            orf_finder = CDSFinder()
            genes = list(orf_finder.find_genes([seq]))

        self.assertEqual(len(ctx), 1)
        self.assertIs(ctx[0].category, IncompleteGeneWarning)