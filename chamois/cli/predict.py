import argparse
import collections
import datetime
import functools
import itertools
import multiprocessing.pool
import pathlib
import shlex
import sys
import typing
from typing import List, Iterable, Set, Optional

import rich.panel
import rich.progress
import rich.tree
from rich.console import Console

from .. import __version__
from ..compositions import build_compositions, build_observations
from .._meta import requires
from .render import build_tree
from ._common import (
    load_model,
    load_sequences,
    find_proteins,
    annotate_hmmer,
    record_metadata,
    initialize_orf_finder,
)
from ._parser import (
    configure_group_predict_input,
    configure_group_gene_finding,
)

if typing.TYPE_CHECKING:
    from anndata import AnnData


def configure_parser(parser: argparse.ArgumentParser):
    group_input = configure_group_predict_input(parser)
    group_input.add_argument(
        "-m",
        "--model",
        default=None,
        type=pathlib.Path,
        help="The path to an alternative model for predicting classes."
    )

    configure_group_gene_finding(parser)
    
    params_output = parser.add_argument_group(
        'Output', 
        'Mandatory and optional outputs.'
    )
    params_output.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="The path where to write the predicted class probabilities in HDF5 format."
    )
    params_output.add_argument(
        "--render",
        action="store_true",
        help="Display prediction results in tree format for each input BGC.",
    )
    
    parser.set_defaults(run=run)


def save_predictions(predictions: "AnnData", path: pathlib.Path, console: Console) -> None:
    console.print(f"[bold blue]{'Saving':>12}[/] result probabilities to {str(path)!r}")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.write(path)


@requires("anndata")
def run(args: argparse.Namespace, console: Console) -> int:
    model = load_model(args.model, console)
    clusters = list(load_sequences(args.input, console))
    uns = record_metadata(model)

    orf_finder = initialize_orf_finder(args.cds, args.jobs, console)
    proteins = find_proteins(clusters, orf_finder, console)

    featurelist = set(model.features_[model.features_.kind == "Pfam"].index)
    domains = annotate_hmmer(
        args.hmm, 
        proteins, 
        args.jobs, 
        console, 
        whitelist=featurelist,
        disentangle=args.disentangle,
    )

    # make compositional data
    obs = build_observations(clusters, proteins)
    compositions = build_compositions(domains, obs, model.features_, uns=uns)

    # predict labels
    console.print(f"[bold blue]{'Predicting':>12}[/] chemical class probabilities")
    probas = model.predict_probas(compositions)
    predictions = anndata.AnnData(X=probas, obs=compositions.obs, var=model.classes_, uns=uns)
    save_predictions(predictions, args.output, console)

    # render if required
    if args.render:
        ic = model.information_content(predictions.X > 0.5)
        for bgc_index in range(predictions.n_obs):
            tree = build_tree(model, predictions.X[bgc_index])
            panel = rich.panel.Panel(tree, title=f"{predictions.obs_names[bgc_index]} (ic={ic[bgc_index]:.1f})")
            console.print(panel)

    return 0