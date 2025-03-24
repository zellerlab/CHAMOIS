import unittest
import tempfile

import anndata
import numpy
from rich.console import Console

import chamois.cli
from chamois.predictor import ChemicalOntologyPredictor

from ..utils import resource_files

class TestChamoisAnnotate(unittest.TestCase):

    @unittest.skipUnless(resource_files, "missing `importlib.resources.files`")
    def test_run(self):
        console = Console(quiet=True)
        src = resource_files("tests").joinpath("data", "BGC0000703.4.gbk")
        with tempfile.NamedTemporaryFile(suffix=".hdf5") as dst:
            retcode = chamois.cli.run(["annotate", "-i", str(src), "-o", dst.name], console)
            self.assertEqual(retcode, 0)
            output = anndata.read_h5ad(dst.name)
            self.assertEqual(output.n_obs, 1)
            self.assertEqual(output.n_vars, 51)            

            predictor = ChemicalOntologyPredictor.trained()
            features = output.var_names[output.X.toarray()[0] > 0]
            self.assertEqual(
                [
                    predictor.features_.index.get_loc(name)
                    for name in features
                ],
                [   
                    1,   23,   26,   46,   70,  141,  150,  170,  187,  224,  227,
                    284,  297,  298,  367,  373,  417,  450,  474,  490,  553,  558,
                    586,  631,  662,  669,  689,  690,  709,  734,  779,  845,  866,
                    904,  928,  948,  995, 1095, 1201, 1250, 1254, 1285, 1286, 1308,
                    1310, 1322, 1337, 1339, 1437, 1572, 1585
                ]
            )
