import argparse
import collections
import functools
import itertools
import multiprocessing.pool
import pathlib
from typing import List, Iterable, Set, Optional

import anndata
import pandas
import pyhmmer
import pyrodigal
import rich.progress
import rich.tree
import scipy.sparse
from rich.console import Console

from ._common import load_model
from ..model import ClusterSequence, Protein
from ..predictor import ChemicalHierarchyPredictor
from .annotate import (
    load_sequences,
    build_observations,
    find_proteins,
    annotate_hmmer,
    annotate_nrpys,
    make_compositions,
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
        required=True,
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
        help="Display results in"
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
    proteins = find_proteins(clusters, args.jobs, console)

    domains = []
    domains.extend(annotate_hmmer(args.hmm, proteins, args.jobs, console, whitelist=set(model.features_.index)))
    domains.extend(annotate_nrpys(proteins, args.jobs, console))

    obs = build_observations(clusters)
    compositions = make_compositions(domains, obs, model.features_, console)

    # predict labels
    console.print(f"[bold blue]{'Predicting':>12}[/] chemical class probabilities")
    probas = model.predict_probas(compositions)
    predictions = anndata.AnnData(X=probas, obs=compositions.obs, var=model.classes_)
    save_predictions(predictions, args.output, console)



