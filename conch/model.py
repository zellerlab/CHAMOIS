import pathlib
import dataclasses
from typing import Optional, List

import gb_io


@dataclasses.dataclass(frozen=True)
class ClusterSequence:
    """The sequence of a biosynthetic gene cluster.
    """
    record: gb_io.Record
    source: Optional[pathlib.Path] = None

    @property
    def id(self) -> str:
        return self.record.name


@dataclasses.dataclass(frozen=True)
class Protein(object):
    """A protein from a biosynthetic gene cluster.
    """
    id: str
    sequence: str
    cluster: ClusterSequence


@dataclasses.dataclass(frozen=True)
class Domain(object):
    """A domain from a protein.
    """
    name: str
    accession: Optional[str]
    kind: str
    protein: Protein


@dataclasses.dataclass(frozen=True)
class HMMDomain(Domain):
    """A protein domain that was found with a Hidden Markov Model.

    See Also:
        `~conch.domains.HMMERAnnotator`: The domain annotator used for 
        searching HMM domains with HMMER.

    """
    start: int
    end: int
    score: float
    pvalue: float
    evalue: float

    def overlaps(self, other: "HMMDomain") -> bool:
        return (
            other.protein is self.protein
            and self.start <= other.end
            and other.start <= self.end
        )


@dataclasses.dataclass(frozen=True)
class AdenylationDomain(Domain):
    """An adenylation domain with specificity for a certain subtrate.

    See Also:
        `~conch.domains.NRPySAnnotator`: The domain annotator used 
        for finding adenylation domains with NRPyS.

    """
    specificity: List[str]
    score: float
