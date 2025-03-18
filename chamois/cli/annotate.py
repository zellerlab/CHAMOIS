import argparse
import datetime
import pathlib
import sys
import typing
from typing import Optional, Set

from rich.console import Console

from .. import __version__
from ..compositions import build_observations, build_variables, build_compositions
from ._common import (
    annotate_hmmer,
    find_proteins,
    load_model,
    load_sequences,
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
    configure_group_predict_input(parser)
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
        help="The path where to write the sequence annotations in HDF5 format."
    )
    
    parser.set_defaults(run=run)


def save_compositions(compositions: "AnnData", path: pathlib.Path, console: Console) -> None:
    console.print(f"[bold blue]{'Saving':>12}[/] compositional matrix to {str(path)!r}")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    compositions.write(path)


def get_whitelist(hmm: Optional[pathlib.Path], console: Console) -> Set[str]:
    if hmm is None:
        model = load_model(None, console)
        return set(model.features_.index)
    else:
        return None


def run(args: argparse.Namespace, console: Console) -> int:
    whitelist = get_whitelist(args.hmm, console)
    clusters = list(load_sequences(args.input, console))
    uns = record_metadata()    

    orf_finder = initialize_orf_finder(args.cds, args.jobs, console)
    proteins = find_proteins(clusters, orf_finder, console)
    domains = annotate_hmmer(args.hmm, proteins, args.jobs, console, whitelist=whitelist)

    obs = build_observations(clusters, proteins)
    var = build_variables(domains)
    compositions = build_compositions(domains, obs, var, uns=uns)
    save_compositions(compositions, args.output, console)
