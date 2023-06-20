import argparse
import collections
import functools
import itertools
import operator
import os
import multiprocessing.pool
import pathlib
from typing import List, Iterable, Set, Optional, Container

import anndata
import Bio.SeqIO
import pandas
import pyhmmer
import pyrodigal
import rich.progress
import rich.tree
import scipy.sparse
from pyhmmer.plan7 import HMM
from rich.console import Console

from ..domains import HMMERAnnotator, NRPSPredictor2Annotator
from ..orf import PyrodigalFinder
from ..model import ClusterSequence, Protein, Domain, ProteinDomain, AdenylationDomain
from ..predictor import ChemicalHierarchyPredictor


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-i", "--input", required=True, type=pathlib.Path, action="append")
    parser.add_argument("-H", "--hmm", required=True, type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    parser.set_defaults(run=run)


# def load_hmms(hmm_file: pathlib.Path, whitelist: Iterable[str], console: Console) -> List[pyhmmer.plan7.HMM]:
#     whitelist = set(whitelist)
#     console.print(f"[bold blue]{'Loading':>12}[/] HMMs from {str(hmm_file)!r}")
#     with rich.progress.Progress(
#         *rich.progress.Progress.get_default_columns(),
#         rich.progress.DownloadColumn(),
#         rich.progress.TransferSpeedColumn(),
#         console=console,
#         transient=True
#     ) as progress:
#         with progress.open(hmm_file, "rb", description=f"[bold blue]{'Reading':>12}[/]") as src:
#             with pyhmmer.plan7.HMMFile(src) as hmms:
#                 hmms = [ hmm for hmm in hmms if hmm.accession.decode() in whitelist ]
#     console.print(f"[bold green]{'Loaded':>12}[/] {len(hmms)} HMMs from {str(hmm_file)!r}")
#     return hmms


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
            with progress.open(input_file, "r", description=f"[bold blue]{'Reading':>12}[/]") as src:
                for record in Bio.SeqIO.parse(src, "genbank"):
                    yield ClusterSequence(record, os.fspath(input_file))
                    n_sequences += 1
        console.print(f"[bold green]{'Loaded':>12}[/] {n_sequences} BGCs from {str(input_file)!r}")


def find_proteins(clusters: List[ClusterSequence], cpus: Optional[int], console: Console) -> List[Protein]:
    console.print(f"[bold blue]{'Finding':>12}[/] genes with Pyrodigal")
    gene_finder = PyrodigalFinder(cpus=cpus)
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(),
        rich.progress.MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"[bold blue]{'Working':>12}[/]", total=None)
        proteins = list(gene_finder.find_genes(
            clusters,
            progress=lambda c, t: progress.update(task_id, total=t, advance=1),
        ))
    console.print(f"[bold green]{'Found':>12}[/] {len(proteins)} proteins in {len(clusters)} clusters")
    return proteins


def annotate_domains(domain_annotator, proteins: List[Protein], console: Console) -> List[Domain]:
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(),
        rich.progress.MofNCompleteColumn(),
        console=console,
        transient=True
    ) as progress:
        total = domain_annotator.total
        task_id = progress.add_task(f"[bold blue]{'Working':>12}[/]", total=total)
        def callback(hmm: HMM, total_: int):
            if total is None:
                progress.update(task_id, total=total_, advance=1)
            else:
                progress.update(task_id, advance=1)
        domains = list(domain_annotator.annotate_domains(proteins, progress=callback))

    return domains


def annotate_hmmer(path: pathlib.Path, proteins: List[Protein], cpus: Optional[int], console: Console, whitelist: Optional[Container[str]] = None) -> List[ProteinDomain]:
    console.print(f"[bold blue]{'Searching':>12}[/] protein domains with HMMER")
    domain_annotator = HMMERAnnotator(path, cpus=cpus, whitelist=whitelist)
    domains = annotate_domains(domain_annotator, proteins, console)
    console.print(f"[bold green]{'Found':>12}[/] {len(domains)} domains under inclusion threshold in {len(proteins)} proteins")
    return domains

def annotate_nrpys(proteins: List[Protein], cpus: Optional[int], console: Console) -> List[AdenylationDomain]:
    console.print(f"[bold blue]{'Predicting':>12}[/] adenylation domain specificity with NRPyS")
    domain_annotator = NRPSPredictor2Annotator(cpus=cpus)
    return annotate_domains(domain_annotator, proteins, console)


def build_observations(clusters: List[ClusterSequence]) -> pandas.DataFrame:
    return pandas.DataFrame(
        index=[cluster.id for cluster in clusters],
        data=dict(
            source=[cluster.source for cluster in clusters],
        )
    )


def build_variables(domains: List[Domain]) -> pandas.DataFrame:
    accessions = []
    names = []
    kind = []

    for domain in domains:
        accessions.append(domain.accession)
        names.append(domain.name)
        if isinstance(domain, ProteinDomain):
            kind.append("HMMER")
        elif isinstance(domain, AdenylationDomain):
            kind.append("NRPyS")

    var = pandas.DataFrame(dict(name=names, accession=accessions, kind=kind))
    var.drop_duplicates(inplace=True)
    var.set_index("accession" if all(accessions) else name, inplace=True)
    var.sort_index(inplace=True)

    return var


def make_compositions(domains: List[Domain], obs: pandas.DataFrame, var: pandas.DataFrame, console: Console) -> anndata.AnnData:
    # check if using accessions or names for HMM features
    use_accession = "name" in var.columns

    # sort domains
    console.print(f"[bold blue]{'Sorting':>12}[/] {len(domains)} remaining domains by source BGC")
    domains.sort(key=lambda d: id(d.protein.cluster))

    # build compositional data
    console.print(f"[bold blue]{'Build':>12}[/] compositional matrix with {len(obs)} observations and {len(var)} variables")
    compositions = scipy.sparse.dok_matrix((len(obs), len(var)), dtype=int)
    for _, bgc_domains in itertools.groupby(domains, key=lambda d: id(d.protein.cluster)):
        bgc_domains = list(bgc_domains)
        bgc_index =  obs.index.get_loc(bgc_domains[0].protein.cluster.id)
        for domain in bgc_domains:
            try:
                domain_index = var.index.get_loc(domain.accession if use_accession else domain.name)
                compositions[bgc_index, domain_index] += 1
            except KeyError:
                continue

    return anndata.AnnData(
        X=compositions.tocsr(),
        obs=obs,
        var=var,
        dtype=int
    )


def save_compositions(compositions: anndata.AnnData, path: pathlib.Path, console: Console) -> None:
    console.print(f"[bold blue]{'Saving':>12}[/] compositional matrix to {str(path)!r}")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    compositions.write(path)


def run(args: argparse.Namespace, console: Console) -> int:
    clusters = list(load_sequences(args.input, console))
    proteins = find_proteins(clusters, args.jobs, console)

    domains = []
    domains.extend(annotate_hmmer(args.hmm, proteins, args.jobs, console))
    domains.extend(annotate_nrpys(proteins, args.jobs, console))

    obs = build_observations(clusters)
    var = build_variables(domains)
    compositions = make_compositions(domains, obs, var, console)
    save_compositions(compositions, args.output, console)




