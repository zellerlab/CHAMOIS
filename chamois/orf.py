"""Generic protocol for ORF detection in DNA sequences.
"""

import abc
import io
import itertools
import os
import queue
import tempfile
import typing
import warnings
from multiprocessing.pool import Pool, ThreadPool
from multiprocessing.sharedctypes import Value
from typing import (
    Callable, 
    Iterable, 
    Iterator, 
    List, 
    Optional, 
    Sequence,
    Tuple, 
    Type, 
    Union,
)

import pyrodigal
import pyhmmer.easel

from .model import ClusterSequence, Protein


__all__ = ["ORFFinder", "PyrodigalFinder", "CDSFinder", "NoGeneFoundWarning"]


class NoGeneFoundWarning(UserWarning):
    """A warning for when no genes were found in a record.
    """


class IncompleteGeneWarning(UserWarning):
    """A warning for when a gene was found with incomplete sequence.
    """


class ORFFinder(metaclass=abc.ABCMeta):
    """An abstract base class to provide a generic ORF finder.
    """

    @abc.abstractmethod
    def find_genes(  # type: ignore
        self,
        clusters: Iterable[ClusterSequence],
        progress: Optional[Callable[[ClusterSequence, int], None]] = None,
    ) -> Iterable[Protein]:
        """Find all genes from a DNA sequence.
        """
        return NotImplemented


class PyrodigalFinder(ORFFinder):
    """An `ORFFinder` that uses the Pyrodigal bindings to Prodigal.

    Prodigal is a fast and reliable protein-coding gene prediction for 
    prokaryotic genomes, with support for draft genomes and metagenomes.
    Since BGCs are short sequences, only "meta" mode can be used for 
    detecting genes in the input.

    References:
        - `Martin Larralde. "Pyrodigal: Python bindings and interface to 
          Prodigal, an efficient method for gene prediction in prokaryotes",
          Journal of Open Source Software, 7(72), 4296, 
          <https://doi.org/10.21105/joss.04296>`_.
        - `Doug Hyatt, Gwo-Liang Chen, Philip F. LoCascio, Miriam L. Land,
          Frank W. Larimer and Loren J. Hauser.
          "Prodigal: Prokaryotic Gene Recognition and Translation Initiation
          Site Identification", BMC Bioinformatics 11 (8 March 2010), p119
          <https://doi.org/10.1186/1471-2105-11-119>`_.

    """

    def __init__(self, mask: bool = False, cpus: Optional[int] = None) -> None:
        """Create a new `PyrodigalFinder` instance.

        Arguments:
            mask (bool): Whether or not to mask genes running across regions
                containing unknown nucleotides, defaults to `False`.
            cpus (int): The number of threads to use to run Pyrodigal in
                parallel. Pass ``0`` to use the number of CPUs on the machine.

        """
        super().__init__()
        self.mask = mask
        self.cpus = cpus
        self.orf_finder =  pyrodigal.GeneFinder(meta=True, closed=True, mask=mask)

    def _process_clusters(self, cluster: ClusterSequence) -> Tuple[ClusterSequence, pyrodigal.Genes]:
        return cluster, self.orf_finder.find_genes(str(cluster.record.sequence))

    def find_genes(
        self,
        clusters: Sequence[ClusterSequence],
        progress: Optional[Callable[[ClusterSequence, int], None]] = None,
        *,
        pool_factory: Union[Type[Pool], Callable[[Optional[int]], Pool]] = ThreadPool,
    ) -> Iterator[Protein]:
        """Find all genes contained in a sequence of DNA records.

        Arguments:
            clusters (iterable of `ClusterSequence`): An iterable of raw cluster
                sequences in which to find genes
            progress (callable, optional): A progress callback of signature
                ``progress(cluster, total)`` that will be called everytime a
                record has been processed successfully, with ``record`` being
                the `ClusterSequence` instance, and ``total`` being the total 
                number of records to process.

        Keyword Arguments:
            pool_factory (`type`): The callable for creating pools, defaults
                to the `multiprocessing.pool.ThreadPool` class, but
                `multiprocessing.pool.Pool` is also supported.

        Yields:
            `Protein`: An iterator over all the genes found in the given 
            records.

        """
        # detect the number of CPUs
        _cpus = os.cpu_count() if self.cpus is None or self.cpus <= 0 else self.cpus
        _progress = (lambda x,y: None) if progress is None else progress

        # run in parallel using a pool
        with pool_factory(_cpus) as pool:
            for cluster, orfs in pool.imap_unordered(self._process_clusters, clusters):
                _progress(cluster, len(clusters))
                for j, orf in enumerate(orfs):
                    yield Protein(
                        f"{cluster.record.name}_{j+1}", 
                        orf.translate(), 
                        cluster
                    )
                if not orfs:
                    warnings.warn(
                        f"no gene found in cluster {cluster.id!r}", 
                        NoGeneFoundWarning,
                        stacklevel=2,
                    )


class CDSFinder(ORFFinder):
    """An `ORFFinder` that simply extracts CDS annotations from records.
    """

    def __init__(
        self,
        feature: str = "CDS",
        translation_table: int = 11,
        locus_tag: str = "locus_tag",
    ):
        self.feature = feature
        self.translation_table = translation_table
        self.locus_tag = locus_tag

    def _translate(
        self,
        sequence: str,
        table: int,
        reverse_complement: bool = False
    ) -> str:
        abc = pyhmmer.easel.Alphabet.dna()
        code = pyhmmer.easel.GeneticCode(table)
        seq = pyhmmer.easel.TextSequence(sequence=sequence).digitize(abc)
        if reverse_complement:
            seq.reverse_complement(inplace=True)
        prot = seq.translate(code)
        return prot.textize().sequence

    def find_genes(
        self,
        clusters: Sequence[ClusterSequence],
        progress: Optional[Callable[[ClusterSequence, int], None]] = None,
    ) -> Iterator[Protein]:
        """Find all genes contained in a sequence of DNA records.
        """
        ids = set()
        _progress = (lambda x,y: None) if progress is None else progress

        for cluster in clusters:
            genes_found = 0
            features = filter(lambda feat: feat.kind == self.feature, cluster.record.features)
            for i, feature in enumerate(features):
                # extract qualifiers from feature
                qualifiers = { qualifier.key:qualifier.value for qualifier in feature.qualifiers }
                # get the gene name
                if self.locus_tag in qualifiers:
                    prot_id = qualifiers[self.locus_tag]
                else:
                    prot_id = f"{cluster.record.name}_{i+1}"
                # get the gene translation
                tt = qualifiers.get("transl_table", self.translation_table)
                if "translation" in qualifiers:
                    prot_seq = qualifiers["translation"]
                else:
                    # get gene sequence
                    start = feature.location.start
                    end = feature.location.end
                    if feature.location.strand == "-":
                        start, end = end, start
                        reverse_complement = True
                    else:
                        reverse_complement = False
                    gene_seq = cluster.record.sequence[start - 1:end]
                    if len(gene_seq) % 3:
                        gene_seq = gene_seq[: (len(gene_seq)//3)*3  ]
                        warnings.warn(
                            f"incomplete protein sequence found for {prot_id!r}",
                            IncompleteGeneWarning,
                            stacklevel=2
                        )
                    # translate using PyHMMER translation
                    prot_seq = self._translate(
                        gene_seq.decode(), 
                        table=int(tt), 
                        reverse_complement=reverse_complement
                    )

                # wrap the gene into a Gene
                yield Protein(
                    prot_id,
                    prot_seq,
                    cluster,
                )
                genes_found += 1
            if not genes_found:
                warnings.warn(
                    f"no gene found in cluster {cluster.id!r}", 
                    NoGeneFoundWarning, 
                    stacklevel=2
                )
            _progress(cluster, len(clusters))
