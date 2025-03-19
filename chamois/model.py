"""Object model for representing genomic regions in CHAMOIS.
"""

import pathlib
import dataclasses
from typing import Optional, List

import gb_io


@dataclasses.dataclass(frozen=True)
class ClusterSequence:
    """The sequence of a biosynthetic gene cluster.

    Attributes:
        record (`~gb_io.Record`): The GenBank record corresponding
            to the cluster sequence.
        source (`~pathlib.Path` or `None`): The path to the file
            this record was loaded from, if any.

    """
    record: gb_io.Record
    source: Optional[pathlib.Path] = None

    @property
    def id(self) -> str:
        """`str`: The identifier of the cluster sequence.
        """
        return self.record.name


@dataclasses.dataclass(frozen=True)
class Protein(object):
    """A protein from a biosynthetic gene cluster.

    Attributes:
        id (`str`): The identifier of the protein.
        sequence (`str`): The sequence of the protein.
        cluster (`~chamois.model.ClusterSequence`): The cluster 
            sequence this protein belongs to.

    """
    id: str
    sequence: str
    cluster: ClusterSequence


@dataclasses.dataclass(frozen=True)
class Domain(object):
    """A domain from a protein.

    Attributes:
        name (`str`): The name of the domain.
        accession (`str` or `None`): The name of the domain, if any.
        description (`str` or `None`): The description of the 
            domain, if any.
        kind (`str`): The kind of domain.
        protein (`~chamois.model.Protein`): The protein this domain
            belongs to.

    """
    name: str
    accession: Optional[str]
    description: Optional[str]
    kind: str
    protein: Protein


@dataclasses.dataclass(frozen=True)
class PfamDomain(Domain):
    """A protein domain that was found with a Pfam HMM.

    Attributes:
        name (`str`): The name of the domain.
        accession (`str` or `None`): The name of the domain, if any.
        description (`str` or `None`): The description of the 
            domain, if any.
        kind (`str`): The kind of domain.
        protein (`~chamois.model.Protein`): The protein this domain
            belongs to.
        start (`int`): The start coordinate of the domain in the
            protein sequence.
        end (`int`): The end coordinate of the domain in the protein
            sequence.
        score (`float`): The raw bitscore for the domain, given by 
            `~pyhmmer.hmmer.hmmsearch`.
        pvalue (`float`): The p-value for the domain, given
            by `~pyhmmer.hmmer.hmmsearch`.
        evalue (`float`): The database-corrected E-value for the domain, 
            given by `~pyhmmer.hmmer.hmmsearch`.

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
        """`bool`: Returns whether two domains overlap in the same protein.
        """
        return (
            other.protein is self.protein
            and self.start <= other.end
            and other.start <= self.end
        )
