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
            retcode = chamois.cli.run(
                ["annotate", "-i", str(src), "-o", dst.name], console
            )
            self.assertEqual(retcode, 0)
            output = anndata.read_h5ad(dst.name)
            self.assertEqual(output.n_obs, 1)
            self.assertEqual(output.n_vars, 47)

            predictor = ChemicalOntologyPredictor.trained()
            features = output.var_names[output.X.toarray()[0] > 0]
            self.assertEqual(
                [predictor.features_.index.get_loc(name) for name in features],
                [
                    1,
                    14,
                    18,
                    32,
                    46,
                    92,
                    112,
                    122,
                    149,
                    150,
                    185,
                    194,
                    195,
                    230,
                    234,
                    259,
                    281,
                    297,
                    333,
                    336,
                    349,
                    370,
                    401,
                    402,
                    408,
                    421,
                    439,
                    469,
                    480,
                    499,
                    506,
                    516,
                    544,
                    591,
                    644,
                    674,
                    677,
                    702,
                    718,
                    720,
                    730,
                    741,
                    780,
                    827,
                    833,
                    861,
                    880,
                ],
            )
