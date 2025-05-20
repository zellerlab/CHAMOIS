import argparse
import collections
import errno
import math
import pathlib
import typing
from typing import List, Iterable, Set, Optional

import numpy
import rich.table
from rich.console import Console
from rich_argparse import ArgumentDefaultsRichHelpFormatter

from .._meta import requires
from ..model import ClusterSequence, Domain, Protein
from ..predictor import ChemicalOntologyPredictor
from ._common import (
    load_model,
    load_sequences,
    find_proteins,
    annotate_hmmer,
    build_observations,
    build_compositions,
    initialize_orf_finder,
)
from ._parser import (
    configure_group_predict_input,
    configure_group_gene_finding,
)

if typing.TYPE_CHECKING:
    from pandas import DataFrame


def configure_output(parser: argparse.ArgumentParser):
    params_output = parser.add_argument_group(
        'Output',
        'Mandatory and optional outputs.'
    )
    params_output.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="The path where to write the cluster contribution table in TSV format."
    )
    params_output.add_argument(
        "--render",
        action="store_true",
        help="Display the cluster contribution table in the console.",
    )
    return params_output

def configure_weights(parser: argparse.ArgumentParser):
    params_weights = parser.add_argument_group(
        'Weights',
        'Control which weights are displayed in the output.'
    )
    subparser = params_weights.add_mutually_exclusive_group()
    subparser.add_argument(
        "--nonzero",
        default=False,
        action="store_true",
        help="Display non-zero weights instead of only positive weights."
    )
    subparser.add_argument(
        "--min-weight",
        default=0.0,
        type=float,
        help="The minimum weight to filter the table with."
    )

def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=pathlib.Path,
        help="The path to an alternative model to extract weights from."
    )

    commands = parser.add_subparsers(required=True)
    parser_class = commands.add_parser(
        "class",
        formatter_class=ArgumentDefaultsRichHelpFormatter,
        help="Explain which domains contribute to a class prediction.",
    )
    parser_class.add_argument(
        "class_id",
        action="store",
        help="The class to explain",
    )
    configure_weights(parser_class)
    configure_output(parser_class)
    parser_class.set_defaults(run=run_class)

    parser_feature = commands.add_parser(
        "feature",
        formatter_class=ArgumentDefaultsRichHelpFormatter,
        help="Explain which features contribute to a class prediction.",
    )
    parser_feature.add_argument(
        "feature_id",
        action="store",
        help="The feature to explain"
    )
    configure_weights(parser_feature)
    configure_output(parser_feature)
    parser_feature.set_defaults(run=run_feature)

    parser_cluster = commands.add_parser(
        "cluster",
        formatter_class=ArgumentDefaultsRichHelpFormatter,
        help="Explain which genes of a cluster contribute to which predicted classes.",
    )
    parser_cluster.add_argument(
        "cluster_id",
        nargs="?",
        help="The cluster to explain",
    )
    configure_group_predict_input(parser_cluster)
    configure_group_gene_finding(parser_cluster)
    configure_output(parser_cluster)
    parser_cluster.set_defaults(run=run_cluster)


def write_table(table: "DataFrame", output: pathlib.Path):
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(output, index=False, sep="\t")


def get_feature_index(feature: str, predictor: ChemicalOntologyPredictor) -> int:
    # get by feature index (full Pfam accession, e.g PF13304.10)
    try:
        return predictor.features_.index.get_loc(feature)
    except KeyError:
        pass
    # get by feature accession (partial accession, e.g. PF13304)
    accessions = predictor.features_.index.str.rsplit(".", n=1).str[0]
    indices = numpy.where(accessions == feature)[0]
    if len(indices) == 1:
        return indices[0]
    # get by feature name (Pfam name, e.g. SBBP)
    names = predictor.features_["name"]
    indices = numpy.where(names == feature)[0]
    if len(indices) == 1:
        return indices[0]
    # failed to find feature
    raise KeyError(feature)


def run_feature(args: argparse.Namespace, console: Console) -> int:
    predictor = load_model(args.model, console)

    # Extract requested class index
    try:
        feature_index = get_feature_index(args.feature_id, predictor)
        name = predictor.features_["name"].iloc[feature_index]
        accession = predictor.features_.index[feature_index]
        console.print(f"[bold blue]{'Extracting':>12}[/] weights for feature [bold blue]{accession}[/] ([green]{name}[/])")
    except KeyError as e:
        console.print(f"[bold red]{'Failed':>12}[/] to find feature [bold blue]{args.feature_id}[/] in model")
        return 1

    # Extract positive weights
    weights = predictor.coef_[feature_index, :].toarray()[0]
    indices = numpy.where((weights != 0.0) if args.nonzero else (weights > args.min_weight))[0]
    selected_classes = predictor.classes_.iloc[indices].copy()
    selected_classes["weight"] = weights[indices]
    selected_classes.sort_values("weight", ascending=False, inplace=True)

    # Render the table
    table = rich.table.Table("ID", "Name", "Weight")
    for row in selected_classes.itertuples():
        table.add_row(
            rich.text.Text(row.Index, style="repr.tag_name"),
            row.name,
            rich.text.Text(format(row.weight, ".5f"), style="repr.number"),
        )
    console.print(table)

    # Write the table
    if args.output is not None:
        write_table(selected_classes, args.output)

    return 0


def get_class_index(class_: str, predictor: ChemicalOntologyPredictor) -> int:
    # get by feature index (full Pfam accession, e.g PF13304.10)
    try:
        return predictor.classes_.index.get_loc(class_)
    except KeyError:
        pass
    # get by feature name (Pfam name, e.g. SBBP)
    names = predictor.classes_["name"]
    indices = numpy.where(names == class_)[0]
    if len(indices) == 1:
        return indices[0]
    # failed to find feature
    raise KeyError(class_)


def run_class(args: argparse.Namespace, console: Console) -> int:
    predictor = load_model(args.model, console)

    # Extract requested class index
    try:
        class_index = get_class_index(args.class_id, predictor)
        name = predictor.classes_["name"].iloc[class_index]
        accession = predictor.classes_.index[class_index]
        console.print(f"[bold blue]{'Extracting':>12}[/] weights for class [bold blue]{args.class_id}[/] ([green]{name}[/])")
    except KeyError:
        console.print(f"[bold red]{'Failed':>12}[/] to find class [bold blue]{args.class_id}[/] in model")
        return 1

    # Extract positive weights
    weights = predictor.coef_[:, class_index].toarray().T[0]
    indices = numpy.where(weights != 0.0 if args.nonzero else weights > args.min_weight)[0]
    selected_classes = predictor.features_.iloc[indices].copy()
    selected_classes["weight"] = weights[indices]
    selected_classes.sort_values("weight", ascending=False, inplace=True)

    # Render the table
    table = rich.table.Table("Feature", "Kind", "Name", "Description", "Weight")
    table.add_row(
        rich.text.Text("Intercept", style="b i"),
        "",
        "",
        "",
        rich.text.Text(format(predictor.intercept_[class_index], ".5f"), style="repr.number"),
        end_section=True,
    )
    for row in selected_classes.itertuples():
        table.add_row(
            rich.text.Text(row.Index, style="repr.tag_name"),
            getattr(row, "kind", "N/A"),
            getattr(row, "name", "N/A"),
            getattr(row, "description", "N/A"),
            rich.text.Text(format(row.weight, ".5f"), style="repr.number"),
        )
    console.print(table)

    # Write the table
    if args.output is not None:
        write_table(selected_classes, args.output)

    return 0


def load_cluster(args: argparse.Namespace, console: Console) -> ClusterSequence:
    if args.cluster_id is not None:
        cluster = next(
            (
                record
                for record in load_sequences(args.input, console)
                if record.id == args.cluster_id
            ),
            None
        )
        if cluster is None:
            raise ValueError(f"no cluster named {args.cluster_id}")
    else:
        clusters = iter(load_sequences(args.input, console))
        cluster = next(clusters, None)
        if cluster is None:
            raise ValueError("no records found")
        if next(clusters, None) is not None:
            raise ValueError("more than one record found")
    return cluster


@requires("pandas")
def build_genetable(
    proteins: List[Protein],
    domains: List[Domain],
    model: ChemicalOntologyPredictor,
    probas: numpy.ndarray,
) -> "DataFrame":
    # group domains per proteins
    protein_domains = collections.defaultdict(list)
    for domain in domains:
        protein_domains[domain.protein.id].append(domain)

    # build cluster/gene table
    rows = []
    for class_id in model.classes_[probas[0, :] >= 0.5].index:
        j = model.classes_.index.get_loc(class_id)
        weights = []
        for prot in proteins:
            w = 0.0
            for domain in protein_domains[prot.id]:
                i = model.features_.index.get_loc(domain.accession)
                w += model.coef_[i, j]
            weights.append(w)
        rows.append([
            class_id,
            model.classes_["name"].loc[class_id],
            probas[0, j], # sum of 1 element
            *weights,
        ])
    return pandas.DataFrame(
        rows,
        columns=["class", "name", "probability", *[prot.id for prot in proteins]],
    )


def _format_weight(x: float) -> rich.text.Text:
    if x <= -2:
        return rich.text.Text(format(x, ".5f"), style="bold red")
    elif x < 0:
        return rich.text.Text(format(x, ".5f"), style="red")
    elif x >= 2:
        return rich.text.Text(format(x, ".5f"), style="bold green")
    elif x > 0:
        return rich.text.Text(format(x, ".5f"), style="green")
    else:
        return rich.text.Text("0.0", style="dim")

def format_genetable(table: "DataFrame", proteins: List[Protein]) -> rich.table.Table:
    # render the table
    console_table = rich.table.Table("Class", "Name", "Probability", *[prot.id for prot in proteins])
    for row in table.itertuples():
        console_table.add_row(
            rich.text.Text(row[1], style="repr.tag_name"),
            row[2],
            rich.text.Text(format(row[3], ".5f"), style="repr.number"),
            *map( _format_weight, row[4:] )
        )
    return console_table


def run_cluster(args: argparse.Namespace, console: Console) -> int:
    model = load_model(args.model, console)

    # get cluster
    try:
        cluster = load_cluster(args, console)
    except Exception as err:
        console.print(f"[bold red]{'Failed':>12}[/] to load cluster from {str(args.input)!r}: {err}")
        return getattr(err, "errno", 1)

    # extract genes
    orf_finder = initialize_orf_finder(args.cds, args.jobs, console)
    proteins = find_proteins([cluster], orf_finder, console)

    # label domains
    featurelist = set(model.features_[model.features_.kind == "Pfam"].index)
    domains = annotate_hmmer(args.hmm, proteins, args.jobs, console, featurelist)

    # make compositional data
    obs = build_observations([cluster], proteins)
    compositions = build_compositions(domains, obs, model.features_)

    # predict labels
    console.print(f"[bold blue]{'Predicting':>12}[/] chemical class probabilities")
    probas = model.predict_probas(compositions)

    # build gene table
    console.print(f"[bold blue]{'Build':>12}[/] gene contribution table")
    table = build_genetable(proteins, domains, model, probas)

    # Write the table
    if args.output is not None:
        write_table(table.set_index("class"), args.output)

    # Render the table
    if args.render:
        console_table = format_genetable(table, proteins)
        rich.print(console_table)

    return 0
