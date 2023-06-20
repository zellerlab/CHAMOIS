import pathlib
import dataclasses
from typing import Optional, List

from Bio.SeqRecord import SeqRecord


@dataclasses.dataclass(frozen=True)
class ClusterSequence:
    record: SeqRecord
    source: Optional[pathlib.Path] = None

    @property
    def id(self) -> str:
        return self.record.id


@dataclasses.dataclass(frozen=True)
class Protein(object):
    id: str
    sequence: str
    cluster: ClusterSequence


@dataclasses.dataclass(frozen=True)
class Domain(object):
    name: str
    accession: Optional[str]
    protein: Protein


@dataclasses.dataclass(frozen=True)
class ProteinDomain(Domain):
    start: int
    end: int
    score: float
    pvalue: float
    evalue: float

    def overlaps(self, other: "Domain") -> bool:
        return (
            other.protein is self.protein
            and self.start <= other.end
            and other.start <= self.end
        )

@dataclasses.dataclass(frozen=True)
class AdenylationDomain(Domain):
    specificity: List[str]
    score: float
