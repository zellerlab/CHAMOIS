import argparse
import collections
import functools
import itertools
import multiprocessing.pool
import pathlib
from typing import List, Iterable, Set

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

from ..model import GeneCluster, Protein
from ..predictor import ChemicalHierarchyPredictor


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-i", "--input", required=True, type=pathlib.Path, action="append")
    parser.add_argument("-m", "--model", required=True, type=pathlib.Path)
    parser.add_argument("-H", "--hmm", required=True, type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    parser.set_defaults(run=run)


def load_hmms(hmm_file: pathlib.Path, whitelist: Iterable[str], console: Console) -> List[pyhmmer.plan7.HMM]:
    whitelist = set(whitelist)
    console.print(f"[bold blue]{'Loading':>12}[/] HMMs from {str(hmm_file)!r}")
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(), 
        rich.progress.DownloadColumn(),
        rich.progress.TransferSpeedColumn(),
        console=console,
        transient=True
    ) as progress:
        with progress.open(hmm_file, "rb", description=f"[bold blue]{'Reading':>12}[/]") as src:
            with pyhmmer.plan7.HMMFile(src) as hmms:
                hmms = [ hmm for hmm in hmms if hmm.accession.decode() in whitelist ]
    console.print(f"[bold green]{'Loaded':>12}[/] {len(hmms)} HMMs from {str(hmm_file)!r}")
    return hmms


def load_sequences(input_files: List[pathlib.Path], console: Console) -> List[GeneCluster]:
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
                with pyhmmer.easel.SequenceFile(src) as seq_file:
                    for record in seq_file:
                        sequences.append(GeneCluster(
                            record.name.decode(), 
                            record.sequence,
                            input_file, 
                        ))
                        n_sequences += 1
        console.print(f"[bold green]{'Loaded':>12}[/] {n_sequences} BGCs from {str(input_file)!r}")
    return sequences


def find_genes(cluster: GeneCluster, progress: rich.progress.Progress, task_id: rich.progress.TaskID):
    orf_finder = pyrodigal.OrfFinder(meta=True)
    proteins = [
        Protein( f"{cluster.id}_{i+1}", gene.translate() )
        for i, gene in enumerate(orf_finder.find_genes(cluster.sequence))
    ]
    progress.update(task_id=task_id, advance=1)
    return proteins


def domains_overlap(dom1: pyhmmer.plan7.Domain, dom2: pyhmmer.plan7.Domain):
    start1 = dom1.alignment.target_from
    end1 = dom1.alignment.target_to
    start2 = dom2.alignment.target_from
    end2 = dom2.alignment.target_to
    return start1 <= end2 and start2 <= end1


def run(args: argparse.Namespace, console: Console) -> int:
    # load data
    console.print(f"[bold blue]{'Loading':>12}[/] trained model from {str(args.model)!r}")
    with open(args.model, "rb") as src:
        model = ChemicalHierarchyPredictor.load(src)

    # load HMMs and sequences
    # hmms = load_hmms(args.hmm, model.features, console)
    bgcs = load_sequences(args.input, console)

    # index the features and the observations
    feature_index = { feat:i for i,feat in enumerate(model.features.index) }
    bgcs_index = { bgc.id:i for i,bgc in enumerate(bgcs) }
    if len(bgcs_index) < len(bgcs):
        raise RuntimeError("Duplicate identifiers found in input BGCs")

    # extract proteins from bgcs
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(), 
        rich.progress.MofNCompleteColumn(),
        console=console,
        transient=True
    ) as progress:
        task_id = progress.add_task(f"[bold blue]{'Processing':>12}[/]", total=len(bgcs))
        with multiprocessing.pool.ThreadPool(args.jobs) as pool:
            _find_genes = functools.partial(find_genes, progress=progress, task_id=task_id)
            proteins = list(itertools.chain.from_iterable(pool.map(_find_genes, bgcs)))
    console.print(f"[bold green]{'Found':>12}[/] {len(proteins)} genes in input BGCs")

    # annotate proteins
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(), 
        rich.progress.MofNCompleteColumn(),
        console=console,
        transient=True
    ) as progress:
        # digitize protein sequences
        alphabet = pyhmmer.easel.Alphabet.amino()
        protein_sequences = pyhmmer.easel.TextSequenceBlock( 
            pyhmmer.easel.TextSequence(name=prot.id.encode(), sequence=prot.sequence)
            for prot in proteins
        ).digitize(alphabet)
        # prepare report callback
        task_id = progress.add_task(f"[bold blue]{'Processing':>12}[/]", total=len(feature_index))
        def callback(hmm: pyhmmer.plan7.HMM, total: int):
            progress.update(task_id=task_id, advance=1)
        # run hmmsearch on all proteins with selected HMMs
        console.print(f"[bold blue]{'Annotating':>12}[/] {len(proteins)} proteins with {len(feature_index)} HMMs")
        with pyhmmer.plan7.HMMFile(args.hmm) as hmm_file:
            hmms = (hmm for hmm in hmm_file if hmm.accession.decode() in feature_index)
            protein_domains = collections.defaultdict(list)
            for hits in pyhmmer.hmmer.hmmsearch(hmms, protein_sequences, bit_cutoffs="trusted", callback=callback, cpus=args.jobs or 0):
                for hit in hits.reported:
                    protein_domains[hit.name].extend(hit.domains.reported)
        console.print(f"[bold green]{'Found':>12}[/] {sum(map(len, protein_domains.values()))} domains under inclusion thresholds")

    # deinterlace hits and generate compositional matrix
    console.print(f"[bold blue]{'Resolving':>12}[/] overlapping domains")
    compositions = torch.zeros((len(bgcs_index), len(feature_index)), dtype=int)
    for sequence_id, domains in protein_domains.items():
        while domains:
            # get a candidate domain for the current gene
            candidate_domain = domains.pop()
            # check if does overlap with other domains
            overlapping = (d for d in domains if domains_overlap(candidate_domain, d))
            for other_domain in overlapping:
                if other_domain.pvalue > candidate_domain.pvalue:
                    # remove other domain if it's worse than the one we
                    # currently have
                    domains.remove(other_domain)
                else:
                    # stop going through overlapping domains, as we found
                    # one better than the candidate; this will cause the
                    # candidate domain to be discarded as well
                    break
            else:
                # no overlapping domain found, candidate is a true positive,
                # we can record this one as a true hit and count it in the BGC
                bgc_id = sequence_id.rsplit(b"_", 1)[0].decode()
                hmm_accession = candidate_domain.hit.hits.query_accession.decode()
                compositions[bgcs_index[bgc_id], feature_index[hmm_accession]] += 1

    # predict labels
    console.print(f"[bold blue]{'Predicting':>12}[/] chemical class probabilities")
    probas = model.predict_proba(compositions).detach().cpu().numpy()

    console.print(f"[bold blue]{'Saving':>12}[/] result probabilities to {str(args.output)!r}")
    data = anndata.AnnData(
        X=probas,
        obs=pandas.DataFrame(index=[bgc.id for bgc in bgcs]),
        var=model.labels
    )
    data.write(args.output)

    
