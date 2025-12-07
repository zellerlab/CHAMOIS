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
            retcode = chamois.cli.run(
                ["predict", "-i", str(src), "-o", dst.name], console
            )
            self.assertEqual(retcode, 0)
            output = anndata.read_h5ad(dst.name)
            self.assertEqual(output.n_obs, 1)

            labels = numpy.where(output.X[0] >= 0.5)[0]
            self.assertEqual(
                list(labels),
                [
                    0,
                    3,
                    46,
                    87,
                    98,
                    99,
                    110,
                    137,
                    204,
                    229,
                    246,
                    250,
                    255,
                    257,
                    279,
                    297,
                    305,
                    312,
                    322,
                    346,
                    347,
                    352,
                    363,
                    364,
                    378,
                    380,
                    425,
                    478,
                    483,
                    519,
                    520,
                    528,
                ],
            )
