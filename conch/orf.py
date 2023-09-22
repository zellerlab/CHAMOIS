"""Generic protocol for ORF detection in DNA sequences.
"""

"""Generic protocol for ORF detection in DNA sequences.
"""

import abc
import io
import itertools
import os
import queue
import tempfile
import typing
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

from .model import ClusterSequence, Protein


__all__ = ["ORFFinder", "PyrodigalFinder", "CDSFinder"]


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

    See Also:
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
            `~conch.model.Protein`: An iterator over all the genes found in
            the given records.

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
            features = filter(lambda feat: feat.type == self.feature, cluster.record.features)
            for i, feature in enumerate(features):
                # get the gene translation
                qualifiers = feature.qualifiers.to_dict()
                tt = qualifiers.get("transl_table", [self.translation_table])[0]
                if "translation" in qualifiers:
                    prot_seq = qualifiers["translation"][0]
                else:
                    prot_seq = feature.location.extract(record.sequence).translate(table=tt)
                # get the gene name
                if self.locus_tag in qualifiers:
                    prot_id = qualifiers[self.locus_tag][0]
                else:
                    prot_id = f"{cluster.record.name}_{i+1}"
                # wrap the gene into a Gene
                yield Protein(
                    prot_id,
                    prot_seq,
                    cluster,
                )
                genes_found += 1
            _progress(cluster, len(clusters))
