from typing import List

import anndata
import pandas
import scipy.sparse

from .model import ClusterSequence, Domain, ProteinDomain, AdenylationDomain


def build_observations(clusters: List[ClusterSequence]) -> pandas.DataFrame:
    return pandas.DataFrame(
        index=[cluster.id for cluster in clusters],
        data=dict(
            source=[cluster.source for cluster in clusters],
        )
    )


def build_variables(domains: List[Domain]) -> pandas.DataFrame:
    accessions = []
    names = []
    kind = []

    for domain in domains:
        accessions.append(domain.accession)
        names.append(domain.name)
        if isinstance(domain, ProteinDomain):
            kind.append("HMMER")
        elif isinstance(domain, AdenylationDomain):
            kind.append("NRPyS")

    var = pandas.DataFrame(dict(name=names, accession=accessions, kind=kind))
    var.drop_duplicates(inplace=True)
    var.set_index("accession" if all(accessions) else name, inplace=True)
    var.sort_index(inplace=True)

    return var


def build_compositions(domains: List[Domain], obs: pandas.DataFrame, var: pandas.DataFrame) -> anndata.AnnData:
    # check if using accessions or names as the index for features
    use_accession = "name" in var.columns

    # build compositional data
    compositions = scipy.sparse.dok_matrix((len(obs), len(var)), dtype=int)
    for domain in domains:
        bgc_index = obs.index.get_loc(domain.protein.cluster.id)
        try:
            domain_index = var.index.get_loc(domain.accession if use_accession else domain.name)
            compositions[bgc_index, domain_index] += 1
        except KeyError:
            continue

    return anndata.AnnData(
        X=compositions.tocsr(),
        obs=obs,
        var=var,
        dtype=int
    )