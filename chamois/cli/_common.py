import argparse
import collections
import datetime
import functools
import itertools
import io
import json
import operator
import os
import math
import multiprocessing.pool
import pathlib
import shlex
import sys
from typing import List, Iterable, Set, Optional, Container, Dict, Any, Tuple

import anndata
import gb_io
import pandas
import pyhmmer
import pyrodigal
import rich.progress
import rich.tree
import scipy.sparse
from pyhmmer.plan7 import HMM
from rich.console import Console

from .. import __version__
from .._meta import zopen
from ..domains import PfamAnnotator
from ..compositions import build_compositions, build_observations, build_variables
from ..orf import ORFFinder, PyrodigalFinder, CDSFinder
from ..model import ClusterSequence, Protein, Domain, PfamDomain
from ..predictor import ChemicalOntologyPredictor


def filter_dataset(
    features: anndata.AnnData,
    classes: anndata.AnnData,
    console: Console,
    similarity: Optional[anndata.AnnData] = None,
    remove_unknown_structure: bool = True,
    min_class_occurrences: int = 1,
    min_feature_occurrences: int = 1,
    min_length: int = 1000,
    min_genes: int = 2,
    fix_mismatch: bool = False,
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    if sorted(features.obs.index) != sorted(classes.obs.index):
        if not fix_mismatch:
            raise ValueError("Index mismatch: {!r} != {!r}".format(
                sorted(features.obs.index),
                sorted(classes.obs.index)
            ))
        obs_names = sorted(set(features.obs_names) & set(classes.obs_names))
        features = features[obs_names]
        classes = classes[obs_names]

    if remove_unknown_structure:
        obs = classes.obs[~classes.obs.unknown_structure]
        features = features[obs.index]
        classes = classes[obs.index]
        console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} observations with known compounds")

    if min_length > 0:
        obs = features.obs[features.obs.length >= min_length]
        features = features[obs.index]
        classes = classes[obs.index]
        console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} observations at least {min_length} nucleotides long")

    if min_genes > 0:
        obs = features.obs[features.obs.genes >= min_genes]
        features = features[obs.index]
        classes = classes[obs.index]
        console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} observations with at least {min_genes} genes")

    if similarity is not None:
        unique = similarity.obs.loc[classes.obs_names].drop_duplicates("groups")
        classes = classes[unique.index]
        features = features[unique.index]
        console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} unique observations")

    class_support = classes.X.sum(axis=0).A1
    features_support = features.X.sum(axis=0).A1
    if min_class_occurrences > 0:
        classes = classes[:, (class_support >= min_class_occurrences) & (class_support <= classes.n_obs - min_class_occurrences)]
        console.print(f"[bold blue]{'Using':>12}[/] {classes.n_vars} classes with at least {min_class_occurrences} observations")
    if min_feature_occurrences > 0:
        features = features[:, (features_support >= min_feature_occurrences) & (features_support <= features.n_obs - min_feature_occurrences)]
        console.print(f"[bold blue]{'Using':>12}[/] {features.n_vars} features with at least {min_feature_occurrences} observations")

    return features, classes


def load_model(path: Optional[pathlib.Path], console: Console) -> ChemicalOntologyPredictor:
    if path is not None:
        console.print(f"[bold blue]{'Loading':>12}[/] trained model from {str(path)!r}")
        with open(path, "rb") as src:
            return ChemicalOntologyPredictor.load(src)
    else:
        console.print(f"[bold blue]{'Loading':>12}[/] embedded model")
        return ChemicalOntologyPredictor.trained()


def load_sequences(input_files: List[pathlib.Path], console: Console) -> Iterable[ClusterSequence]:
    sequences = []
    for input_file in input_files:
        console.print(f"[bold blue]{'Loading':>12}[/] BGCs from {str(input_file)!r}")
        n_sequences = 0
        with rich.progress.Progress(
            *rich.progress.Progress.get_default_columns(),
            rich.progress.DownloadColumn(),
            rich.progress.TransferSpeedColumn(),
            console=console,
            transient=True
        ) as progress:
            with progress.open(input_file, "rb", description=f"[bold blue]{'Reading':>12}[/]") as src:
                with zopen(src) as reader:
                    for record in gb_io.iter(reader):
                        yield ClusterSequence(record, os.fspath(input_file))
                        n_sequences += 1
        console.print(f"[bold green]{'Loaded':>12}[/] {n_sequences} BGCs from {str(input_file)!r}")


def find_proteins(clusters: List[ClusterSequence], orf_finder: ORFFinder, console: Console) -> List[Protein]:
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(),
        rich.progress.MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"[bold blue]{'Working':>12}[/]", total=None)
        proteins = list(orf_finder.find_genes(
            clusters,
            progress=lambda c, t: progress.update(task_id, total=t, advance=1),
        ))
    console.print(f"[bold green]{'Found':>12}[/] {len(proteins)} proteins in {len(clusters)} clusters")
    return proteins


def annotate_domains(domain_annotator, proteins: List[Protein], console: Console, total: Optional[int] = None) -> List[Domain]:
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(),
        rich.progress.MofNCompleteColumn(),
        console=console,
        transient=True
    ) as progress:
        task_id = progress.add_task(f"[bold blue]{'Working':>12}[/]", total=total)
        def callback(hmm: HMM, total_: int):
            if total is None:
                progress.update(task_id, total=total_, advance=1)
            else:
                progress.update(task_id, advance=1)
        domains = list(domain_annotator.annotate_domains(proteins, progress=callback))

    return domains


def annotate_hmmer(path: pathlib.Path, proteins: List[Protein], cpus: Optional[int], console: Console, whitelist: Optional[Container[str]] = None) -> List[PfamDomain]:
    console.print(f"[bold blue]{'Searching':>12}[/] protein domains with HMMER")
    domain_annotator = PfamAnnotator(path, cpus=cpus, whitelist=whitelist)
    total = len(whitelist) if whitelist else None
    domains = annotate_domains(domain_annotator, proteins, console, total=total)
    console.print(f"[bold green]{'Found':>12}[/] {len(domains)} domains under inclusion threshold in {len(proteins)} proteins")
    return domains


def record_metadata(predictor: Optional[ChemicalOntologyPredictor] = None) -> Dict[str, Any]:
    metadata = {
        "version": __version__,
        "datetime": datetime.datetime.now().isoformat(),
        "command": shlex.join(sys.argv),
    }
    if predictor is not None:
        metadata["predictor"] = predictor.checksum()
    return {"chamois": metadata}


def save_metrics(
    metrics: Dict[str, float],
    path: Optional[pathlib.Path],
    console: Console,
):
    if path is not None:
        console.print(f"[bold blue]{'Saving':>12}[/] metrics to {str(path)!r}")
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as dst:
            json.dump(metrics, dst)
