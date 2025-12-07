import unittest
import warnings

import gb_io

from chamois.model import ClusterSequence, Protein
from chamois.domains import PfamAnnotator

from .utils import resource_files


class TestPfamAnnotator(unittest.TestCase):

    def test_annnotate_domains(self):

        seq = (
            "TIGHVDHGKTTLTAAIATICAKTYGGEAKDYSQIDSAPEEKARGITINTSHVEYDSPTRH"
            "YAHVDCPGHADYVKNMITGAAQMDGAILVCAATDGPMPQTREHILLSRQVGVPYIIVFLN"
            "KCDLVDDEELLELVEMEVRELLSTYDFPGDDTPVIRGSALAALNG"
        )
        cluster = ClusterSequence(gb_io.Record(sequence=b"", name="test"))
        protein = Protein("test_1", seq, cluster)

        annotator = PfamAnnotator(whitelist=["PF00009.34", "PF00015.27"])
        domains = list(annotator.annotate_domains([protein]))

        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].accession, "PF00009.34")
