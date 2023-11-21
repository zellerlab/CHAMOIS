import argparse
import pathlib
from typing import List, Iterable, Set, Optional

import anndata
import numpy
import pandas
import rich.table
import scipy.stats
from rich.console import Console
from scipy.spatial.distance import cdist, hamming, cosine

from ._common import load_model
from .._meta import requires
from ..predictor import ChemicalOntologyPredictor
from ..ontology import Ontology
from ._parser import (
    configure_group_preprocessing,
    configure_group_training_input,
    configure_group_hyperparameters,
    configure_group_cross_validation,
)


def configure_parser(parser: argparse.ArgumentParser):
    params_input = configure_group_training_input(parser)
    params_input.add_argument(
        "--catalog",
        required=True,
        type=pathlib.Path,
        help="The path to the compound class catalog to compare predictions to."
    )

    configure_group_preprocessing(parser)
    configure_group_hyperparameters(parser)
    configure_group_cross_validation(parser)
    
    params_output = parser.add_argument_group(
        'Output', 
        'Mandatory and optional outputs.'
    )
    params_output.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="The path where to write the catalog search results in TSV format."
    )
    # params_output.add_argument(
    #     "--report",
    #     type=pathlib.Path,
    #     help="An optional file where to generate a label-wise evaluation report."
    # )
    # params_output.add_argument(
    #     "--rank",
    #     default=10,
    #     type=int,
    #     help="The maximum search rank to record in the table output.",
    # )
    parser.set_defaults(run=run)


# def load_predictions(path: pathlib.Path, predictor: ChemicalOntologyPredictor, console: Console) -> anndata.AnnData:
#     console.print(f"[bold blue]{'Loading':>12}[/] probability predictions from {str(path)!r}")
#     probas = anndata.read(path)
#     probas = probas[:, predictor.classes_.index]
#     classes = predictor.propagate(probas.X > 0.5)
#     return anndata.AnnData(X=classes, obs=probas.obs, var=probas.var, dtype=bool)


def load_catalog(path: pathlib.Path, console: Console) -> anndata.AnnData:
    console.print(f"[bold blue]{'Loading':>12}[/] compound catalog from {str(path)!r}")
    catalog = anndata.read(path)
    return catalog


# def build_results(
#     classes: anndata.AnnData,
#     catalog: anndata.AnnData,
#     distances: numpy.ndarray,
#     ranks: numpy.ndarray,
#     max_rank: int,
# ) -> pandas.DataFrame:
#     rows = []
#     for i, name in enumerate(classes.obs_names):
#         for j in ranks[i].argsort():
#             if ranks[i, j] > max_rank:
#                 break
#             rows.append([
#                 name,
#                 ranks[i, j],
#                 catalog.obs.index[j],
#                 catalog.obs.compound[j],
#                 distances[i, j],
#             ])
#     return pandas.DataFrame(
#         rows,
#         columns=["bgc_id", "rank", "index", "compound", "distance"]
#     )


# def build_table(results: pandas.DataFrame) -> rich.table.Table:
#     table = rich.table.Table("BGC", "Index", "Compound", "Distance")
#     for bgc_id, rows in results[results["rank"] == 1].groupby("bgc_id", sort=False):
#         for i, row in enumerate(rows.itertuples()):
#             table.add_row(
#                 row.bgc_id if i == 0 else "",
#                 row.index,
#                 rich.text.Text(row.compound, style="repr.tag_name"),
#                 rich.text.Text(format(row.distance, ".5f"), style="repr.number"),
#                 end_section=i==len(rows)-1,
#             )
#     return table


@requires("kennard_stone")
@requires("rdkit.Chem.rdMHFPFingerprint")
@requires("rdkit.DataStructs")
@requires("rdkit.RDLogger")
@requires("sklearn.metrics.pairwise")
def run(args: argparse.Namespace, console: Console) -> int:
    # disable rdkit logging
    rdkit.RDLogger.DisableLog('rdApp.warning')  
    mhfp_encoder = rdkit.Chem.rdMHFPFingerprint.MHFPEncoder(2048, args.seed)

    # load data
    console.print(f"[bold blue]{'Loading':>12}[/] training data")
    features = anndata.read(args.features)
    classes = anndata.read(args.classes)
    console.print(f"[bold green]{'Loaded':>12}[/] {features.n_obs} observations, {features.n_vars} features and {classes.n_vars} classes")
    # remove compounds with unknown structure
    features = features[~classes.obs.unknown_structure]
    classes = classes[~classes.obs.unknown_structure]
    console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} observations with known compounds")
    # remove similar BGCs based on nucleotide similarity
    # if args.similarity is not None:
    #     ani = anndata.read(args.similarity).obs
    #     ani = ani.loc[classes.obs_names].drop_duplicates("groups")
    #     classes = classes[ani.index]
    #     features = features[ani.index]
    #     console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} unique observations based on nucleotide similarity")
    # remove classes absent from training set
    classes = classes[:, (classes.X.sum(axis=0).A1 >= 5) & (classes.X.sum(axis=0).A1 <= classes.n_obs - 5)]
    console.print(f"[bold blue]{'Using':>12}[/] {classes.n_vars} nontautological classes")
    # prepare ontology and groups
    ontology = Ontology(classes.varp["parents"])
    groups = classes.obs["compound"].cat.codes

    #
    #good_classes = pandas.read_table("good_classes.tsv", index_col=0)

    # load catalog
    catalog = load_catalog(args.catalog, console)[:, classes.var.index]
    #catalog = catalog[:, catalog.var_names.isin(good_classes.index)]

    # start training
    ground_truth = classes.X.toarray()
    console.print(f"[bold blue]{'Splitting':>12}[/] data into {args.kfolds} folds")
    if args.sampling == "group":
        kfold = sklearn.model_selection.GroupShuffleSplit(n_splits=args.kfolds, random_state=args.seed)
    elif args.sampling == "random":
        kfold = sklearn.model_selection.KFold(n_splits=args.kfolds, random_state=args.seed, shuffle=True)
    elif args.sampling == "kennard-stone":
        kfold = kennard_stone.KFold(n_splits=args.kfolds, n_jobs=args.jobs, metric="cosine")
    else:
        raise ValueError(f"Invalid value for `--sampling`: {args.sampling!r}")
    splits = list(kfold.split(features.X.toarray(), ground_truth, groups))

    # define search metric (using GOGO similarity)
    # def metric(x: numpy.ndarray, y: numpy.ndarray) -> float:
    #     i = numpy.where(x)[0]
    #     j = numpy.where(y)[0]
    #     return 1.0 - ontology.similarity(i, j)
    class Metric:
        def __init__(self, x: numpy.ndarray):
            self._x = numpy.where(x)[0]
            self._sim1 = ontology.semantic_similarity[self._x]
            self._sim2 = ontology.semantic_similarity[:, self._x]
        def __call__(self, y: numpy.ndarray):
            _y = numpy.where(y)[0]
            sx = self._sim1[:, _y].max(initial=0, axis=0).sum()
            sy = self._sim2[_y].max(initial=0, axis=0).sum()
            return (sx+sy) / (self._x.shape[0] + _y.shape[0])

    # record ranks
    ranks_true = numpy.zeros(classes.n_obs)
    ranks_total = numpy.zeros(classes.n_obs)

    # train models on cross-validation folds
    console.print(f"[bold blue]{'Running':>12}[/] cross-validation evaluation")
    probas = numpy.zeros(classes.X.shape, dtype=float)
    for i, (train_indices, test_indices) in enumerate(splits):
        # extract fold observations
        train_X = features[train_indices]
        train_Y = classes[train_indices]
        # train fold
        model = ChemicalOntologyPredictor(
            ontology,
            n_jobs=args.jobs,
            model=args.model,
            alpha=args.alpha,
            variance=args.variance,
        )
        model.fit(train_X, train_Y)

        # obtain predictions
        test_X = features[test_indices, model.features_.index]
        probas = model.predict_probas(test_X)
        predictions = model.propagate(probas > 0.5)
        #predictions = predictions[:, model.classes_.index.isin(good_classes.index)]

        # compute distance
        console.print(f"[bold blue]{'Computing':>12}[/] pairwise distances and ranks")

        # compute top-K accuracy
        cat = catalog.X.toarray()
        inchikey_index = { row.inchikey:i for i,row in enumerate(catalog.obs.itertuples()) }
        top03_count = 0
        top10_count = 0
        top30_count = 0
        top50_count = 0
        top_total = 0
        rank_true = []
        rank_total = []
        for x in rich.progress.track(range(len(predictions)), console=console):
            # compute distances and ranks
            m = Metric(predictions[x])
            similarities = numpy.array([m(y) for y in cat])
            distances = 1.0 - similarities
            #distances = numpy.array([cosine(predictions[x], y) for y in cat])
            ranks = scipy.stats.rankdata(distances, method="dense")
            # check if true compound is in catalog
            inchikey = classes.obs.iloc[test_indices].inchikey[x]
            if inchikey in inchikey_index:
                # compute top-k accuracy
                rank = ranks[inchikey_index[inchikey]]
                top_rank = ranks.max()
                top03_count += rank <= 3
                top10_count += rank <= 10
                top30_count += rank <= 30
                top50_count += rank <= 50
                top_total += 1
                rank_true.append(rank)
                rank_total.append(top_rank)
            else:
                rank_true.append(None)
                rank_total.append(None)

        ranks_true[test_indices] = rank_true
        ranks_total[test_indices] = rank_total

        stats = [
            f"[bold magenta]Top03Accuracy=[/][bold cyan]{top03_count/top_total:05.1%}[/]",
            f"[bold magenta]Top10Accuracy=[/][bold cyan]{top10_count/top_total:05.1%}[/]",
            f"[bold magenta]Top30Accuracy=[/][bold cyan]{top30_count/top_total:05.1%}[/]",
            f"[bold magenta]Top50Accuracy=[/][bold cyan]{top50_count/top_total:05.1%}[/]",
            # f"[bold magenta]ClosestTop10=[/][bold cyan]{best_dist_total/len(predictions):5.3}[/]",
        ]
        console.print(f"[bold green]{'Finished':>12}[/] fold {i+1:2}:", *stats)

    # save results
    obs = classes.obs.copy()
    obs["rank_true"] = ranks_true
    obs["rank_total"] = ranks_total
    if args.output:
        console.print(f"[bold blue]{'Saving':>12}[/] search results to {str(args.output)!r}")
        obs.to_csv(args.output, sep="\t", index=False)
