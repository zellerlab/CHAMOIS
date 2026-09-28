"""Generic protocol for domain annotation in proteins.
"""

import abc
import collections.abc
import contextlib
import io
import itertools
import pathlib
import warnings
from typing import Iterable, Optional, Callable, Container, List, Set

import pyhmmer
from pyhmmer.plan7 import HMM, HMMFile
from pyhmmer.easel import Alphabet, DigitalSequenceBlock, TextSequenceBlock, TextSequence

from .._meta import zopen
from ..model import Protein, Domain, PfamDomain

try:
    from importlib.resources import files, as_file
except ImportError:
    from importlib_resources import files, as_file


class _UniversalContainer(collections.abc.Container):
    def __contains__(self, item: object) -> bool:
        return True


class DomainAnnotator(metaclass=abc.ABCMeta):
    """An abstract class for annotating genes with protein domains.
    """

    @property
    def total(self) -> Optional[int]:
        """`int` or `None`: The total number of features to annotate.
        """
        return None

    @abc.abstractmethod
    def annotate_domains(
        self,
        proteins: List[Protein],
        progress: Optional[Callable[[HMM, int], None]] = None,
    ) -> Iterable[Domain]:
        """Run annotation on proteins of ``genes`` and update their domains.

        Arguments:
            genes (`~collections.abc.Iterable` of `~chamois.model.Protein`): An
                iterable that yields proteins to annotate.

        """
        return NotImplemented


class PfamAnnotator(DomainAnnotator):
    """A domain annotator that uses `pyhmmer` to find Pfam domains in proteins.

    References:
        - `Martin Larralde and Georg Zeller. "PyHMMER: a Python library
          binding to HMMER for efficient sequence analysis",
          Bioinformatics, Volume 39, Issue 5, May 2023,
          <https://doi.org/10.1093/bioinformatics/btad214>`_.

    """

    def __init__(
        self,
        path: Optional[pathlib.Path] = None,
        cpus: Optional[int] = None,
        whitelist: Optional[Container[str]] = None,
    ) -> None:
        """Prepare a new HMMER annotation handler with the given ``hmms``.

        Arguments:
            file (`pathlib.Path`): The path to the file containing the
                Pfam HMMs.
            cpus (`int`, optional): The number of CPUs to allocate for the
                ``hmmsearch`` command. Give ``None`` to use the default.
            whitelist (`~collections.abc.Container` of `str`): If given, a
                container containing the accessions of the individual
                HMMs to annotate with. If `None` is given, annotate with the
                entire file.

        """
        super().__init__()
        self.path = path
        self.cpus = cpus
        self.whitelist = whitelist

    @property
    def total(self):
        if self.whitelist is None:
            return None
        return len(self.whitelist)

    def _load_hmm(self, ctx: contextlib.ExitStack) -> HMMFile:
        if self.path is not None:
            file: BinaryIO = ctx.enter_context(zopen(self.path))
        else:
            handle = files(__package__).joinpath("Pfam38.0.hmm.zst")
            file = ctx.enter_context(zopen(ctx.enter_context(handle.open("rb"))))
        return ctx.enter_context(HMMFile(file))

    def annotate_domains(
        self,
        proteins: List[Protein],
        progress: Optional[Callable[[HMM, int], None]] = None,
    ) -> Iterable[PfamDomain]:
        # convert proteins to Easel sequences, naming them after
        # their location in the original input to ignore any duplicate
        # protein identifiers
        esl_abc = Alphabet.amino()
        esl_sqs = TextSequenceBlock([
            TextSequence(name=str(i), sequence=str(protein.sequence))
            for i, protein in enumerate(proteins)
        ])

        # use a PyHMMER callback to check all the requested HMMs
        # from the whitelist were loaded and processed successfully
        if self.whitelist is not None:
            done = dict.fromkeys(self.whitelist, False)
            def _progress_and_check_whitelist(hmm, total):
                if progress is not None:
                    progress(hmm, total)
                done[hmm.accession] = True
        else:
            _progress_and_check_whitelist = progress

        with contextlib.ExitStack() as ctx:
            # only retain the HMMs which are in the whitelist
            hmm_file = self._load_hmm(ctx)
            hmms = (
                hmm
                for hmm in hmm_file
                if self.whitelist is None
                or hmm.accession in self.whitelist
            )
            # Run search pipeline using the filtered HMMs
            cpus = 0 if self.cpus is None else self.cpus
            hmms_hits = pyhmmer.hmmer.hmmsearch(
                hmms,
                esl_sqs.digitize(esl_abc),
                cpus=cpus,
                callback=_progress_and_check_whitelist, # type: ignore
                bit_cutoffs="trusted",  # type: ignore
            )

            # Transcribe HMMER hits to model
            for hits in hmms_hits:
                for hit in hits.reported:
                    target_index = int(hit.name)
                    for domain in hit.domains.reported:
                        yield PfamDomain(
                            name=hits.query.name,
                            accession=hits.query.accession,
                            description=hits.query.description,
                            kind="Pfam",
                            start=domain.alignment.target_from,
                            end=domain.alignment.target_to,
                            score=domain.score,
                            pvalue=domain.pvalue,
                            evalue=domain.i_evalue,
                            protein=proteins[target_index],
                        )

        # Raise warning if some HMMs from the whitelist are missing
        if self.whitelist is not None:
            total_done = sum(done.values())
            if total_done != self.total:
                warnings.warn(
                    f"Missing HMMs, check your HMM file ({self.total} expected, {total_done} found)",
                    UserWarning,
                )

    def disentangle_domains(
        self,
        domains: List[PfamDomain]
    ) -> Iterable[PfamDomain]:
        """Pick the best domain from overlapping domains in each protein.
        """
        domains.sort(key=lambda domain: (domain.protein.id, domain.start))
        for protein_id, protein_domains in itertools.groupby(domains, lambda domain: domain.protein):
            protein_domains = list(protein_domains)
            while protein_domains:
                candidate_domain = protein_domains.pop()
                for other_domain in filter(candidate_domain.overlaps, protein_domains.copy()):
                    if other_domain.pvalue > candidate_domain.pvalue:
                        protein_domains.remove(other_domain)
                    else:
                        break
                else:
                    yield candidate_domain
