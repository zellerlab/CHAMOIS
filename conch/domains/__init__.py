import abc
import collections.abc
import contextlib
import io
import pathlib
from typing import Iterable, Optional, Callable, Container, List, Set

import nrpys
import pyhmmer
from pyhmmer.plan7 import HMM, HMMFile
from pyhmmer.easel import Alphabet, DigitalSequenceBlock, TextSequenceBlock, TextSequence

from ..model import Protein, Domain, ProteinDomain, AdenylationDomain

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
            genes (iterable of `~conch.model.Protein`): An iterable that
                yields proteins to annotate.

        """
        return NotImplemented


class HMMERAnnotator(DomainAnnotator):
    """A domain annotator that uses PyHMMER to annotate domains in proteins.
    """

    def __init__(
        self,
        path: Optional[pathlib.Path] = None,
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

    @property
    def total(self):
        if isinstance(self.whitelist, _UniversalContainer):
            return None 
        return len(self.whitelist)

    def _load_hmm(self, ctx: contextlib.ExitStack) -> HMMFile:
        if self.path is not None:
            file: BinaryIO = io.BufferedReader(ctx.enter_context(open(self.path, "rb")))
        else:
            file = ctx.enter_context(files(__name__).joinpath("Pfam35.0.hmm").open("rb"))
        if file.peek().startswith(b"\x1f\x8b"):
            file = ctx.enter_context(gzip.GzipFile(fileobj=file, mode="rb"))  # type: ignore
        return ctx.enter_context(HMMFile(file))

    def annotate_domains(
        self,
        proteins: List[Protein],
        progress: Optional[Callable[[HMM, int], None]] = None,
    ) -> Iterable[ProteinDomain]:
        # convert proteins to Easel sequences, naming them after
        # their location in the original input to ignore any duplicate
        # protein identifiers
        esl_abc = Alphabet.amino()
        esl_sqs = TextSequenceBlock([
            TextSequence(name=str(i).encode(), sequence=str(protein.sequence))
            for i, protein in enumerate(proteins)
        ])

        with contextlib.ExitStack() as ctx:
            # only retain the HMMs which are in the whitelist
            hmm_file = self._load_hmm(ctx)
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
                        yield ProteinDomain(
                            name=name.decode(),
                            accession=None if acc is None else acc.decode(),
                            start=domain.alignment.target_from,
                            end=domain.alignment.target_to,
                            score=domain.score,
                            pvalue=domain.pvalue,
                            evalue=domain.i_evalue,
                            protein=proteins[target_index],
                        )


class NRPSPredictor2Annotator(DomainAnnotator):

    _POSITIONS = [
        12, 15, 16, 40, 45, 46, 47, 48, 49, 50, 51, 54, 92, 93, 124, 125, 126, 127,
        128, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
    ]

    def __init__(
        self,
        cpus: Optional[int] = None,
        threshold: float = 0.80,
        category: nrpys.PredictionCategory = nrpys.PredictionCategory.LargeClusterV2
    ) -> None:
        super().__init__()
        self.cpus = cpus
        self.threshold = threshold
        self.category = category

    @classmethod
    def _make_signature(cls, alignment: pyhmmer.plan7.Alignment) -> Optional[str]:
        # adjust position of interest to account for gaps in the ref sequence alignment
        positions = []
        position_skipping_gaps = 0
        for i, amino in enumerate(alignment.hmm_sequence):
            if amino == "-" or amino == ".":
                continue
            if position_skipping_gaps in cls._POSITIONS:
                positions.append(i)
            position_skipping_gaps += 1
        if len(positions) != len(cls._POSITIONS):
            return None
        # extract positions from alignment
        return "".join([alignment.target_sequence[i] for i in positions])

    @classmethod
    def _get_name(cls, specificity: Set[str]):
        if specificity == {"Asp", "Asn", "Glu", "Gln", "Aad"}:
            return "Aliphatic chain with H-bond donor"
        elif specificity == {"Cys"}:
            return "Polar, uncharged (aliphatic with -SH)"
        elif specificity == {"Dhb", "Sal"}:
            return "Hydroxy-benzoic acid derivates"
        elif specificity == {"Gly", "Ala", "Val", "Leu", "Ile", "Abu", "Iva"}:
            return "Apolar, aliphatic"
        elif specificity == {"Orn", "Lys", "Arg"}:
            return "Long positively charged side chain"
        elif specificity == {"Phe", "Trp", "Phg", "Tyr", "Bht"}:
            return "Aromatic side chain"
        elif specificity == {"Pro", "Pip"}:
            return "Cyclic aliphatic chain (polar NH2 group)"
        elif specificity == {"Ser", "Thr", "Dhpg", "Hpg"}:
            return "Aliphatic chain or phenyl group with -OH"
        elif specificity == {"Dhpg", "Hpg"}:
            return "Polar, uncharged (hydroxy-phenyl)"
        elif specificity == {"Gly", "Ala"}:
            return "Tiny, hydrophilic, transition to aliphatic"
        elif specificity == {"Orn", "Horn"}:
            return "Orn and hydroxy-Orn specific"
        elif specificity == {"Phe", "Trp"}:
            return "Unpolar aromatic ring"
        elif specificity == {"Tyr", "Bht"}:
            return "Polar aromatic ring"
        elif specificity == {"Val", "Leu", "Ile", "Abu", "Iva"}:
            return "Aliphatic, branched hydrophobic"
        return ",".join(specificity)

    def _load_hmm(self) -> HMM:
        with files(__name__).joinpath("aa-activating-core.hmm").open("rb") as f:
            with pyhmmer.plan7.HMMFile(f) as hmm_file:
                return hmm_file.read()

    def annotate_domains(
        self,
        proteins: List[Protein],
        progress: Optional[Callable[[HMM, int], None]] = None,
    ) -> Iterable[AdenylationDomain]:
        # convert proteins to Easel sequences, naming them after
        # their location in the original input to ignore any duplicate
        # protein identifiers
        esl_abc = Alphabet.amino()
        esl_sqs = TextSequenceBlock([
            TextSequence(name=str(i).encode(), sequence=str(protein.sequence))
            for i, protein in enumerate(proteins)
        ])

        # find adenylation domains
        with contextlib.ExitStack() as ctx:
            # load AMP-binding specific HMM
            hmm = self._load_hmm()
            # Run search pipeline using the filtered HMMs
            cpus = 0 if self.cpus is None else self.cpus
            hmms_hits = pyhmmer.hmmer.hmmscan(
                esl_sqs.digitize(esl_abc),
                [hmm],
                cpus=cpus,
                callback=progress, # type: ignore
                T=20.0,
                domT=20.0,
            )
            # extract subsequences with a match
            signatures = []
            names = []
            for hits in hmms_hits:
                target_index = int(hits.query_name)
                for hit in hits:
                    for domain in hit.domains:
                        signature = self._make_signature(domain.alignment)
                        if signature is not None:
                            names.append(hits.query_name.decode())
                            signatures.append(signature)

            # run NRPyS
            config = nrpys.Config()
            config.model_dir = ctx.enter_context(as_file(files(__name__).joinpath("models")))
            config.skip_v1 = True
            config.skip_v3 = True
            config.skip_stachelhaus = True
            results = nrpys.run(config, signatures=signatures, names=names)

            # extract results, using large cluster since it has the best
            # resolution / accuracy tradeoff
            for result in results:
                pred = next(iter(result.get_best(self.category)), None)
                if pred is None or pred.score < self.threshold:
                    continue
                target_index = int(result.name)
                specificity = set(map(str.capitalize, pred.name.split(",")))
                yield AdenylationDomain(
                    accession=f"NRPyS:{'|'.join(sorted(specificity))}",
                    name=self._get_name(specificity),
                    specificity=specificity,
                    protein=proteins[target_index],
                    score=pred.score,
                )