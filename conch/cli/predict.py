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
import torch
import scipy.sparse
from rich.console import Console
from torch_treecrf import TreeMatrix

from ..model import ClusterSequence, Protein
from ..predictor import ChemicalHierarchyPredictor
from .annotate import (
    load_sequences,
    build_observations,
    find_proteins,
    annotate_domains,
    resolve_overlaps,
    make_compositions,
)


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-i", "--input", required=True, type=pathlib.Path, action="append")
    parser.add_argument("-m", "--model", default=None, type=pathlib.Path)
    parser.add_argument("-H", "--hmm", required=True, type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    parser.set_defaults(run=run)


def load_model(path: Optional[pathlib.Path], console: Console) -> ChemicalHierarchyPredictor:
    if path is not None:
        console.print(f"[bold blue]{'Loading':>12}[/] trained model from {str(path)!r}")
        with open(path, "rb") as src:
            return ChemicalHierarchyPredictor.load(src)
    else:
        console.print(f"[bold blue]{'Loading':>12}[/] embedded model")
        return ChemicalHierarchyPredictor.trained()


def save_predictions(predictions: anndata.AnnData, path: pathlib.Path, console: Console) -> None:
    console.print(f"[bold blue]{'Saving':>12}[/] result probabilities to {str(path)!r}")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.write(path)


def run(args: argparse.Namespace, console: Console) -> int:
    model = load_model(args.model, console)
    clusters = list(load_sequences(args.input, console))
    proteins = find_proteins(clusters, args.jobs, console)
    domains = annotate_domains(args.hmm, proteins, args.jobs, console)
    obs = build_observations(clusters)
    compositions = make_compositions(domains, obs, model.features, console)

    # predict labels
    console.print(f"[bold blue]{'Predicting':>12}[/] chemical class probabilities")
    probas = model.predict_proba(compositions).detach().cpu().numpy()
    predictions = anndata.AnnData(X=probas, obs=compositions.obs, var=model.labels)
    save_predictions(predictions, args.output, console)
    

    
