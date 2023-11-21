import argparse
import datetime
import pathlib
import sys

import anndata
from rich.console import Console

from .. import __version__
from ..orf import CDSFinder, PyrodigalFinder
from ..compositions import build_observations, build_variables, build_compositions
from ._common import (
    annotate_hmmer,
    annotate_nrpys,
    find_proteins,
    load_sequences,
    record_metadata,
)
from ._parser import (
    configure_group_predict_input,
    configure_group_gene_finding,
)


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


def save_compositions(compositions: anndata.AnnData, path: pathlib.Path, console: Console) -> None:
    console.print(f"[bold blue]{'Saving':>12}[/] compositional matrix to {str(path)!r}")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    compositions.write(path)


def run(args: argparse.Namespace, console: Console) -> int:
    clusters = list(load_sequences(args.input, console))
    uns = record_metadata()    

    if args.cds:
        console.print(f"[bold blue]{'Extracting':>12}[/] genes from [bold cyan]CDS[/] features")
        orf_finder = CDSFinder()
    else:
        console.print(f"[bold blue]{'Finding':>12}[/] genes with Pyrodigal")
        orf_finder = PyrodigalFinder(cpus=args.jobs) 
    proteins = find_proteins(clusters, orf_finder, console)

    domains = annotate_hmmer(args.hmm, proteins, args.jobs, console)

    obs = build_observations(clusters, proteins)
    var = build_variables(domains)
    compositions = build_compositions(domains, obs, var, uns=uns)
    save_compositions(compositions, args.output, console)




