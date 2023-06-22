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

    description = "download the Pfam HMMs required by CONCH to annotate domains"
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
            self.version = "35.0"

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
        if isinstance(rich, ImportError):
            raise RuntimeError("`lz4` is required to run the `download_pfam` command") from rich

        # Load domain whitelist from the predictor
        predictor_file = os.path.join("conch", "predictor.json")
        self.info(f"loading domain accesssions from {predictor_file}")
        with open(predictor_file, "rb") as f:
            data = json.load(f)
        domains = [
            accession for accession,kind in data["features_"]["kind"].items()
            if kind == "HMMER"
        ]

        # Download and binarize required HMMs
        local = os.path.join(self.build_lib, "conch", "domains", f"Pfam{self.version}.hmm.lz4")
        self.mkpath(os.path.dirname(local))
        self.download_pfam(local, domains)
        if self.inplace:
            copy = os.path.relpath(local, self.build_lib)
            self.make_file([local], copy, shutil.copy, (local, copy))

    def download_pfam(self, local, domains):
        # download the HMM to `local`, and delete the file if any error
        # or interruption happens during the download
        try:
            self.make_file([], local, self.download_hmms, (local, domains))
        except BaseException:
            if os.path.exists(local):
                os.remove(local)
            raise

    def download_hmms(self, output, domains):
        # get URL for the requested Pfam version
        url = self.get_url()
        self.info(f"fetching {url}")
        response = urllib.request.urlopen(url)
        
        # use `rich` to make a progress bar
        pbar = rich.progress.wrap_file(
            response,
            total=int(response.headers["Content-Length"]),
            description=os.path.basename(output),
        )

        # download to `output`
        nsource = nwritten = 0
        with contextlib.ExitStack() as ctx:
            dl = ctx.enter_context(pbar)
            src = ctx.enter_context(gzip.open(dl))
            dst = ctx.enter_context(lz4.frame.open(output, "wb"))
            for hmm in HMMFile(src):
                nsource += 1
                if hmm.accession.decode() in domains:
                    nwritten += 1
                    hmm.write(dst, binary=False)
        
        # log number of HMMs kept in final files
        self.info(f"downloaded {nwritten} HMMs out of {nsource} in the source file")


class download_nrps2(setuptools.Command):
    """A custom `setuptools` command to download NRPSPredictor2 data.
    """

    description = "download the NRPSPredictor2 model required by NRPyS"
    user_options = [
        ("force", "f", "force downloading the files even if they exist"),
        ("inplace", "i", "ignore build-lib and put data alongside your Python code"),
    ]

    def initialize_options(self):
        self.force = False
        self.inplace = False

    def finalize_options(self):
        _build_py = self.get_finalized_command("build_py")
        self.build_lib = _build_py.build_lib

    def info(self, msg):
        self.announce(msg, level=2)

    def run(self):
        # make sure the build/lib/ folder exists
        self.mkpath(self.build_lib)

        # Check `rich` and `pyhmmer` are installed
        if isinstance(rich, ImportError):
            raise RuntimeError("`rich` is required to run the `download_pfam` command") from rich

        local = os.path.join(self.build_lib, "conch", "domains", "models")
        self.mkpath(os.path.dirname(local))
        self.make_file([], local, self.download_nrps2_models, (local,))
        if self.inplace:
            copy = os.path.relpath(local, self.build_lib)
            self.copy_tree(local, copy)

        local = os.path.join(self.build_lib, "conch", "domains", "aa-activating-core.hmm")
        self.make_file([], local, self.download_hmm, (local,))
        if self.inplace:
            copy = os.path.relpath(local, self.build_lib)
            self.make_file([local], copy, shutil.copy, (local, copy))

    def download_nrps2_models(self, output):
        # download from the Medema webserver
        url = "https://dl.secondarymetabolites.org/releases/nrps_svm/2.0/models.tar.xz"
        self.info(f"fetching {url}")
        response = urllib.request.urlopen(url)
        
        # use `rich` to make a progress bar
        pbar = rich.progress.wrap_file(
            response,
            total=int(response.headers["Content-Length"]),
            description=os.path.basename(output),
        )

        # download to buffer
        with contextlib.ExitStack() as ctx:
            dl = ctx.enter_context(pbar)
            buffer = io.BytesIO(dl.read())
            
        # extract
        with tarfile.open(fileobj=buffer) as tar:
            for entry in tar.getmembers():
                if "NRPS2_LARGE_CLUSTER" in entry.name:
                    tar.extract(entry, output)

    def download_hmm(self, output):
        # download from git
        url = "https://github.com/antismash/antismash/raw/master/antismash/modules/nrps_pks/data/aa-activating.aroundLys.hmm"
        self.info(f"fetching {url}")
        self.info(f"fetching {url}")
        response = urllib.request.urlopen(url)
        
        # use `rich` to make a progress bar
        pbar = rich.progress.wrap_file(
            response,
            total=int(response.headers["Content-Length"]),
            description=os.path.basename(output),
        )

        # download to buffer
        with contextlib.ExitStack() as ctx:
            dl = ctx.enter_context(pbar)
            out = ctx.enter_context(open(output, "wb"))
            shutil.copyfileobj(dl, out)


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
        for command_name in ["download_pfam", "download_nrps2"]:
            # build data if needed
            if not self.distribution.have_run.get(command_name, False):
                command = self.get_finalized_command(command_name)
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
            "download_nrps2": download_nrps2,
            "list_requirements": list_requirements,
            "sdist": sdist,
        },
    )
