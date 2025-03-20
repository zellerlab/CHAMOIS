import unittest
import tempfile

import anndata
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