import argparse
import pathlib

import anndata
from rich.console import Console

from ..orf import CDSFinder, PyrodigalFinder
from ..compositions import build_observations, build_variables, build_compositions
from ._common import (
    annotate_hmmer,
    annotate_nrpys,
    find_proteins,
    load_sequences,
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
        help="The path where to write the sequence annotations in HDF5 format."
    )
    parser.add_argument(
        "--cds",
        action="store_true",
        help="Use CDS features in the GenBank input as genes instead of running Pyrodigal.",
    )
    parser.set_defaults(run=run)


def save_compositions(compositions: anndata.AnnData, path: pathlib.Path, console: Console) -> None:
    console.print(f"[bold blue]{'Saving':>12}[/] compositional matrix to {str(path)!r}")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    compositions.write(path)


def run(args: argparse.Namespace, console: Console) -> int:
    clusters = list(load_sequences(args.input, console))
    
    if args.cds:
        console.print(f"[bold blue]{'Extracting':>12}[/] genes from [bold cyan]CDS[/] features")
        orf_finder = CDSFinder()
    else:
        console.print(f"[bold blue]{'Finding':>12}[/] genes with Pyrodigal")
        orf_finder = PyrodigalFinder(cpus=args.jobs) 
    proteins = find_proteins(clusters, orf_finder, console)

    domains = [
        *annotate_hmmer(args.hmm, proteins, args.jobs, console),
        *annotate_nrpys(proteins, args.jobs, console),
    ]

    obs = build_observations(clusters)
    var = build_variables(domains)
    compositions = build_compositions(domains, obs, var)
    save_compositions(compositions, args.output, console)




