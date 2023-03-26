import pathlib
import dataclasses
from typing import Optional

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
    start: int
    end: int
    score: float
    pvalue: float
    evalue: float
    protein: Protein

    def overlaps(self, other: "Domain") -> bool:
        return (
            other.protein is self.protein
            and self.start <= other.end
            and other.start <= self.end
        )