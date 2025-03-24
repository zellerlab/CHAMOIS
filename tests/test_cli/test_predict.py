import unittest
import tempfile

import anndata
import numpy
from rich.console import Console

import chamois.cli

from ..utils import resource_files

class TestChamoisPredict(unittest.TestCase):

    @unittest.skipUnless(resource_files, "missing `importlib.resources.files`")
    def test_run(self):
        console = Console(quiet=True)
        src = resource_files("tests").joinpath("data", "BGC0000703.4.gbk")
        with tempfile.NamedTemporaryFile(suffix=".hdf5") as dst:
            retcode = chamois.cli.run(["predict", "-i", str(src), "-o", dst.name], console)
            self.assertEqual(retcode, 0)
            output = anndata.read_h5ad(dst.name)
            self.assertEqual(output.n_obs, 1)

            labels = numpy.where( output.X[0] >= 0.5 )[0]
            self.assertEqual(
                list(labels),
                [
                    0,   3,  42,  79,  89,  91,  94, 102, 129, 185, 206, 219, 223,
                    230, 253, 268, 273, 279, 290, 310, 311, 316, 327, 328, 340, 342,
                    379, 394, 432, 437, 470, 471, 477
                ]
            )