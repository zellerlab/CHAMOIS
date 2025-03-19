import unittest
import warnings

import gb_io

from chamois.model import ClusterSequence, Protein
from chamois.domains import PfamAnnotator

from .utils import resource_files


class TestPfamAnnotator(unittest.TestCase):
    
    def test_annnotate_domains(self):

        seq = (
            "MTISIIANFKAKSDQKETLEALLKSVIELTLQEEGCLKYELYISENDSSRYFFLEEWRSR"
            "EDLDIHIASDYIQSLFGNIQSLIESSDIIEIKKI"
        )
        cluster = ClusterSequence(gb_io.Record(sequence=b"", name="test"))
        protein = Protein("test_1", seq, cluster)

        annotator = PfamAnnotator(whitelist=["PF03992.20", "PF00389.34"])
        domains = list(annotator.annotate_domains([protein]))

        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0].accession, "PF03992.20")
