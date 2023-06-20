import pathlib
import dataclasses
from typing import Optional, List

import gb_io


@dataclasses.dataclass(frozen=True)
class ClusterSequence:
    record: gb_io.Record
    source: Optional[pathlib.Path] = None

    @property
    def id(self) -> str:
        return self.record.name


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
