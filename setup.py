#!/usr/bin/env python

import configparser
import contextlib
import csv
import glob
import hashlib
import io
import json
import math
import os
import re
import shutil
import ssl
import struct
import sys
import tarfile
import time
import urllib.request
from functools import partial
from xml.etree import ElementTree as etree

import setuptools
from distutils import log
from distutils.command.build import build as _build
from distutils.command.clean import clean as _clean
from setuptools.command.sdist import sdist as _sdist

try:
    import rich.progress
except ImportError as err:
    rich = None

try:
    from pyhmmer.plan7 import HMMFile
except ImportError as err:
    HMMFile = err

try:
    from isal import igzip as gzip
except ImportError:
    import gzip

try:
    import lz4.frame
except ImportError as err:
    lz4 = err


class sdist(_sdist):
    """An extension to the `sdist` command that generates a `pyproject.toml`.
    """

    def run(self):
        # build `pyproject.toml` from `setup.cfg`
        c = configparser.ConfigParser()
        c.add_section("build-system")
        c.set("build-system", "requires", str(self.distribution.setup_requires))
        c.set("build-system", "build-backend", '"setuptools.build_meta"')
        with open("pyproject.toml", "w") as pyproject:
            c.write(pyproject)

        # run the rest of the packaging
        _sdist.run(self)


class list_requirements(setuptools.Command):
    """A custom command to write the project requirements.
    """

    description = "list the project requirements"
    user_options = [
        ("setup", "s", "show the setup requirements as well."),
        (
            "requirements=",
            "r",
            "the name of the requirements file (defaults to requirements.txt)"
        )
    ]

    def initialize_options(self):
        self.setup = False
        self.output = None

    def finalize_options(self):
        if self.output is None:
            self.output = "requirements.txt"

    def run(self):
        cfg = configparser.ConfigParser()
        cfg.read(__file__.replace(".py", ".cfg"))

        with open(self.output, "w") as f:
            if self.setup:
                f.write(cfg.get("options", "setup_requires"))
            f.write(cfg.get("options", "install_requires"))
            for _, v in cfg.items("options.extras_require"):
                f.write(v)


class download_pfam(setuptools.Command):
    """A custom `setuptools` command to download data before wheel creation.
    """

    description = "download the Pfam HMMs required by CHAMOIS to annotate domains"
    user_options = [
        ("force", "f", "force downloading the files even if they exist"),
        ("inplace", "i", "ignore build-lib and put data alongside your Python code"),
        ("version=", "v", "the Pfam version to dowload")
    ]

    def initialize_options(self):
        self.force = False
        self.inplace = False
        self.version = None

    def finalize_options(self):
        _build_py = self.get_finalized_command("build_py")
        self.build_lib = _build_py.build_lib
        if self.version is None:
            self.version = "36.0"

    def info(self, msg):
        self.announce(msg, level=2)

    def get_url(self):
        return f"http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam{self.version}/Pfam-A.hmm.gz"

    def run(self):
        # make sure the build/lib/ folder exists
        self.mkpath(self.build_lib)

        # Check `rich` and `pyhmmer` are installed
        if isinstance(HMMFile, ImportError):
            raise RuntimeError("pyhmmer is required to run the `download_pfam` command") from HMMFile
        if isinstance(rich, ImportError):
            raise RuntimeError("`rich` is required to run the `download_pfam` command") from rich
        if isinstance(lz4, ImportError):
            raise RuntimeError("`lz4` is required to run the `download_pfam` command") from rich

        # Load domain whitelist from the predictor
        predictor_file = os.path.join("chamois", "predictor", "predictor.json")
        self.info(f"loading domain accesssions from {predictor_file}")
        with open(predictor_file, "rb") as f:
            data = json.load(f)
        features = data["features_"]
        kind_index = features['columns'].index('kind')
        domains = [
            accession
            for accession, row in zip(features["index"], features["data"])
            if row[kind_index] == "Pfam"
        ]

        # Download and binarize required HMMs
        local = os.path.join(self.build_lib, "chamois", "domains", f"Pfam{self.version}.hmm.lz4")
        self.mkpath(os.path.dirname(local))
        self.download_pfam(local, domains)
        if self.inplace:
            copy = os.path.relpath(local, self.build_lib)
            self.make_file([local], copy, shutil.copy, (local, copy))

    def download_pfam(self, local, domains):
        # download the HMM to `local`, and delete the file if any error
        # or interruption happens during the download
        if not os.path.exists(local):
            error = None
            # streaming the HMMs may not work on all platforms (e.g. MacOS)
            # so we fall back to a buffered implementation if needed.
            for stream in (True, False):
                try:
                    self.download_hmms(local, domains, stream=stream)
                except Exception as exc:
                    error = exc
                    if os.path.exists(local):
                        os.remove(local)
                else:
                    return
            raise RuntimeError("Failed to download Pfam HMMs") from error

    def download_hmms(self, output, domains, stream=True):
        # get URL for the requested Pfam version
        url = self.get_url()
        self.info(f"fetching {url}")
        response = urllib.request.urlopen(url)

        # download to `output`
        nsource = nwritten = 0
        with contextlib.ExitStack() as ctx:
            # use `rich` to make a progress bar if available
            if rich is not None:
                pbar = rich.progress.wrap_file(
                    response,
                    total=int(response.headers["Content-Length"]),
                    description=os.path.basename(output),
                )
                dl = ctx.enter_context(pbar)
            else:
                dl = ctx.enter_context(response)

            src = ctx.enter_context(gzip.open(dl))
            dst = ctx.enter_context(lz4.frame.open(output, "wb"))
            if stream:
                hmm_file = ctx.enter_context(HMMFile(src))
            else:
                buffer = io.BytesIO()
                shutil.copyfileobj(src, buffer)
                buffer.seek(0)
                hmm_file = HMMFile(buffer)
            for hmm in hmm_file:
                nsource += 1
                if hmm.accession.decode() in domains:
                    nwritten += 1
                    hmm.write(dst, binary=False)

        # log number of HMMs kept in final files
        self.info(f"downloaded {nwritten} HMMs out of {nsource} in the source file")


class build(_build):
    """A hacked `build` command that will also download required data.
    """

    user_options = _build.user_options + [
        ("inplace", "i", "ignore build-lib and put data alongside your Python code"),
    ]

    def initialize_options(self):
        _build.initialize_options(self)
        self.inplace = False

    def run(self):
        # build data if needed
        if not self.distribution.have_run.get("download_pfam", False):
            command = self.get_finalized_command("download_pfam")
            command.force = self.force
            command.inplace = self.inplace
            command.run()
        # build rest as normal
        _build.run(self)


if __name__ == "__main__":
    setuptools.setup(
        cmdclass={
            "build": build,
            "download_pfam": download_pfam,
            "list_requirements": list_requirements,
            "sdist": sdist,
        },
    )
