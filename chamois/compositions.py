"""Transformation of `chamois.model` objects into compositional data.

This module contains helper functions to transform objects from the
object model in `chamois.model` into compositional data suitable to
pass to the `~chamois.predictor.ChemicalOntologyPredictor.fit` and
`~chamois.predictor.ChemicalOntologyPredictor.predict` methods of
`~chamois.predictor.ChemicalOntologyPredictor` objects.

The compositional matrices are stored in `~anndata.AnnData` objects
to ensure that metadata related to the observations and features are
retained along the actual binary indicator matrices.

"""

import collections
import typing
from typing import Any, List, Mapping, Optional, Iterable

from ._meta import requires
from .model import ClusterSequence, Protein, Domain

if typing.TYPE_CHECKING:
    from anndata import AnnData
    from pandas import DataFrame


@requires("pandas")
def build_observations(
    clusters: List[ClusterSequence],
    proteins: Optional[Iterable[Protein]] = None,
) -> "DataFrame":
    """Build an observation table from a list of cluster sequences.

    Arguments:
        clusters (`list` of `~chamois.model.ClusterSequence`): The cluster
            sequences to add to the observations table.
        proteins (`~typing.Iterable` of `~chamois.model.Protein`): The
            proteins extracted from the clusters, or `None`. If given,
            the observations table will contain an additional column
            with the number of proteins per cluster.

    Returns:
        `~pandas.DataFrame`: The data frame containing the cluster
        sequences and their medata, to be used as the ``obs`` table of
        an `~anndata.AnnData` object.

    """
    # build columns
    data = {
        "source": [cluster.source for cluster in clusters],
        "length": [len(cluster.record.sequence) for cluster in clusters],
    }
    if not any(data["source"]):
        del data["source"]
    # count proteins per cluster if given
    if proteins is not None:
        counter = collections.Counter()
        for protein in proteins:
            counter[protein.cluster.id] += 1
        data["genes"] = [ counter[cluster.id] for cluster in clusters ]
    # build dataframe
    return pandas.DataFrame(
        index=[cluster.id for cluster in clusters],
        data=data,
    )


@requires("pandas")
def build_variables(domains: Iterable[Domain]) -> "DataFrame":
    """Build a variable table from an iterable of domains.

    The domain accessions will be used if all domains have an accession
    set, otherwise the domain names will be used (for compatibility with
    other HMM libraries than Pfam).

    Arguments:
        domains (`~typing.Iterable` of `~chamois.model.Domain`): The domains
            to add to the variables table.

    Returns:
        `~pandas.DataFrame`: The data frame containing the sorted,
        deduplicated domains to be used as the ``var`` table of an
        `~anndata.AnnData` object.

    """
    var = pandas.DataFrame([
        dict(
            name=domain.name,
            accession=domain.accession,
            description=domain.description,
            kind=domain.kind,
        )
        for domain in domains
    ])
    var.drop_duplicates(inplace=True)
    var.set_index("accession" if var["accession"].all() else name, inplace=True)
    var.sort_index(inplace=True)
    return var


@requires("anndata")
@requires("scipy.sparse")
def build_compositions(
    domains: List[Domain],
    obs: "DataFrame",
    var: "DataFrame",
    uns: Optional[Mapping[str, Any]] = None,
) -> "AnnData":
    """Build a compositional matrix from the given domain.

    Arguments:
        domains (`~typing.Iterable` of `~chamois.model.Domain`): The domains
            found in the clusters to turn into a binary indicator matrix.
        obs (`~pandas.DataFrame`): The input clusters, given as an
            observation table (obtained with `build_observations`).
        var (`~pandas.DataFrame`): The feature domains, given as a
            variable table (obtained with `build_variables`).
        uns (`~typing.Mapping` of `str` to `object`): Additional
            unstructured metadata to be added to the created
            `~anndata.AnnData` object.

    Returns:
        `~anndata.AnnData`: The compositional matrix, encoding the
        presence of protein domains in each gene cluster as a binary
        indicator matrix stored in a `~scipy.sparse.csr_matrix`.

    """
    use_accession = "name" in var.columns
    compositions = scipy.sparse.dok_matrix((len(obs), len(var)), dtype=int)
    for domain in domains:
        bgc_index = obs.index.get_loc(domain.protein.cluster.id)
        try:
            domain_index = var.index.get_loc(domain.accession if use_accession else domain.name)
            compositions[bgc_index, domain_index] = True
        except KeyError:
            continue
    return anndata.AnnData(
        X=compositions.tocsr(),
        obs=obs,
        var=var,
        uns=uns
    )