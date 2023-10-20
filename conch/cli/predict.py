import argparse
import collections
import datetime
import functools
import itertools
import multiprocessing.pool
import pathlib
import shlex
import sys
from typing import List, Iterable, Set, Optional

import anndata
import pandas
import pyhmmer
import pyrodigal
import rich.panel
import rich.progress
import rich.tree
import scipy.sparse
from rich.console import Console

from .. import __version__
from ..compositions import build_compositions, build_observations
from ..orf import PyrodigalFinder, CDSFinder
from .render import build_tree
from ._common import (
    load_model,
    load_sequences,
    find_proteins,
    annotate_hmmer,
    annotate_nrpys,
    record_metadata,
)


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        action="append",
        help="The input BGC sequences to process."
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=pathlib.Path,
        help="The path to an alternative model for predicting classes."
    )
    parser.add_argument(
        "-H",
        "--hmm",
        default=None,
        type=pathlib.Path,
        help="The path to the HMM file containing protein domains for annotation."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="The path where to write the predicted class probabilities in HDF5 format."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Display prediction results in tree format for each input BGC.",
    )
    parser.add_argument(
        "--cds",
        action="store_true",
        help="Use CDS features in the GenBank input as genes instead of running Pyrodigal.",
    )
    parser.set_defaults(run=run)


def save_predictions(predictions: anndata.AnnData, path: pathlib.Path, console: Console) -> None:
    console.print(f"[bold blue]{'Saving':>12}[/] result probabilities to {str(path)!r}")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.write(path)


def run(args: argparse.Namespace, console: Console) -> int:
    model = load_model(args.model, console)
    clusters = list(load_sequences(args.input, console))
    uns = record_metadata(model)

    if args.cds:
        console.print(f"[bold blue]{'Extracting':>12}[/] genes from [bold cyan]CDS[/] features")
        orf_finder = CDSFinder()
    else:
        console.print(f"[bold blue]{'Finding':>12}[/] genes with Pyrodigal")
        orf_finder = PyrodigalFinder(cpus=args.jobs) 
    proteins = find_proteins(clusters, orf_finder, console)

    featurelist = set(model.features_[model.features_.kind == "Pfam"].index)
    domains = [
        *annotate_hmmer(args.hmm, proteins, args.jobs, console, featurelist),
        *annotate_nrpys(proteins, args.jobs, console)
    ]

    # make compositional data
    obs = build_observations(clusters)
    compositions = build_compositions(domains, obs, model.features_, uns=uns)

    # predict labels
    console.print(f"[bold blue]{'Predicting':>12}[/] chemical class probabilities")
    probas = model.predict_probas(compositions)
    predictions = anndata.AnnData(X=probas, obs=compositions.obs, var=model.classes_, uns=uns)
    save_predictions(predictions, args.output, console)

    # render if required
    if args.render:
        for bgc_index in range(predictions.n_obs):
            tree = build_tree(model, predictions.X[bgc_index])
            panel = rich.panel.Panel(tree, title=predictions.obs_names[bgc_index])
            console.print(panel)


