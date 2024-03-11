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
    description: Optional[str]
    kind: str
    protein: Protein


@dataclasses.dataclass(frozen=True)
class PfamDomain(Domain):
    """A protein domain that was found with a Pfam HMM.

    See Also:
        `~chamois.domains.PfamAnnotator`: The domain annotator used for 
        searching Pfam domains with HMMER.

    """
    start: int
    end: int
    score: float
    pvalue: float
    evalue: float

    def overlaps(self, other: "PfamDomain") -> bool:
        return (
            other.protein is self.protein
            and self.start <= other.end
            and other.start <= self.end
        )
