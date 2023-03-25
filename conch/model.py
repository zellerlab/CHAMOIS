import pathlib
import dataclasses
from typing import Optional

@dataclasses.dataclass(frozen=True)
class GeneCluster:
    id: str
    sequence: str
    source: Optional[pathlib.Path] = None

@dataclasses.dataclass
class Protein(object):
    id: str
    sequence: str

