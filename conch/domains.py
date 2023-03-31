import abc
import collections.abc
import contextlib
import io
import pathlib
from typing import Iterable, Optional, Callable, Container

import pyhmmer
from pyhmmer.plan7 import HMM, HMMFile
from pyhmmer.easel import Alphabet, TextSequenceBlock, TextSequence

from .model import Protein, Domain


class _UniversalContainer(collections.abc.Container):
    def __contains__(self, item: object) -> bool:
        return True


class DomainAnnotator(metaclass=abc.ABCMeta):
    """An abstract class for annotating genes with protein domains.
    """

    @abc.abstractmethod
    def annotate_domains(
        self, 
        proteins: Iterable[Protein],
        progress: Optional[Callable[[HMM, int], None]] = None,
    ) -> Iterable[Domain]:
        """Run annotation on proteins of ``genes`` and update their domains.

        Arguments:
            genes (iterable of `~gecco.model.Gene`): An iterable that yield
                genes to annotate with ``self.hmm``.

        """
        return NotImplemented


class HMMERAnnotator(DomainAnnotator):
    """A domain annotator that uses PyHMMER to annotate domains in proteins.
    """

    def __init__(
        self, 
        path: pathlib.Path, 
        cpus: Optional[int] = None, 
        whitelist: Optional[Container[str]] = None,
        use_name: bool = False,
    ) -> None:
        """Prepare a new HMMER annotation handler with the given ``hmms``.

        Arguments:
            file (`pathlib.Path`): The path to the file containing the HMMs.
            cpus (`int`, optional): The number of CPUs to allocate for the
                ``hmmsearch`` command. Give ``None`` to use the default.
            whitelist (container of `str`): If given, a container containing
                the accessions of the individual HMMs to annotate with. If
                `None` is given, annotate with the entire file.

        """
        super().__init__()
        self.path = path
        self.cpus = cpus
        self.use_name = False
        self.whitelist = _UniversalContainer() if whitelist is None else whitelist

    def annotate_domains(
        self,
        proteins: Iterable[Protein],
        progress: Optional[Callable[[HMM, int], None]] = None,
    ) -> Iterable[Domain]:
        # collect proteins and keep them in original order
        protein_index = list(proteins)

        # convert proteins to Easel sequences, naming them after
        # their location in the original input to ignore any duplicate
        # protein identifiers
        esl_abc = Alphabet.amino()
        esl_sqs = TextSequenceBlock([
            TextSequence(name=str(i).encode(), sequence=str(protein.sequence))
            for i, protein in enumerate(protein_index)
        ])

        with contextlib.ExitStack() as ctx:
            # decompress the input if needed
            file: BinaryIO = io.BufferedReader(ctx.enter_context(open(self.path, "rb")))
            if file.peek().startswith(b"\x1f\x8b"):
                file = ctx.enter_context(gzip.GzipFile(fileobj=file, mode="rb"))  # type: ignore
            # Only retain the HMMs which are in the whitelist
            hmm_file = ctx.enter_context(HMMFile(file))
            hmms = (
                hmm
                for hmm in hmm_file
                if hmm.accession.decode() in self.whitelist
            )
            # Run search pipeline using the filtered HMMs
            cpus = 0 if self.cpus is None else self.cpus
            hmms_hits = pyhmmer.hmmer.hmmsearch(
                hmms,
                esl_sqs.digitize(esl_abc),
                cpus=cpus,
                callback=progress, # type: ignore
                bit_cutoffs="trusted",  # type: ignore
            )

            # Transcribe HMMER hits to model
            for hits in hmms_hits:
                for hit in hits.reported:
                    target_index = int(hit.name)
                    for domain in hit.domains.reported:
                        # extract HMM name and coordinates
                        name = domain.alignment.hmm_name
                        acc = domain.alignment.hmm_accession
                        yield Domain(
                            name=name.decode(),
                            accession=None if acc is None else acc.decode(),
                            start=domain.alignment.target_from,
                            end=domain.alignment.target_to,
                            score=domain.score,
                            pvalue=domain.pvalue,
                            evalue=domain.i_evalue,
                            protein=protein_index[target_index],
                        )
