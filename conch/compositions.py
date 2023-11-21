import collections
from typing import Any, List, Mapping, Optional

import anndata
import pandas
import scipy.sparse

from .model import ClusterSequence, Protein, Domain


def build_observations(
    clusters: List[ClusterSequence],
    proteins: Optional[List[Protein]] = None,
) -> pandas.DataFrame:
    # build columns
    data = {
        "source": [cluster.source for cluster in clusters],
        "length": [len(cluster.record.sequence) for cluster in clusters],
    }
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


def build_variables(domains: List[Domain]) -> pandas.DataFrame:
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


def build_compositions(
    domains: List[Domain], 
    obs: pandas.DataFrame, 
    var: pandas.DataFrame,
    uns: Optional[Mapping[str, Any]] = None,
) -> anndata.AnnData:
    use_accession = "name" in var.columns
    compositions = scipy.sparse.dok_matrix((len(obs), len(var)), dtype=bool)
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
        dtype=int,
        uns=uns
    )