import argparse
import math
import pathlib
from typing import List, Iterable, Set, Optional

import anndata
import numpy
import pandas
import rich.table
import scipy.stats
from rich.console import Console
from rich_argparse import RichHelpFormatter
from scipy.spatial.distance import cdist

from ._common import load_model
from ..predictor import ChemicalOntologyPredictor


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=pathlib.Path,
        help="The path to an alternative model to extract weights from."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="The path where to write the table in TSV format."
    )

    subparser = parser.add_mutually_exclusive_group()
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

    commands = parser.add_subparsers(required=True)
    parser_class = commands.add_parser(
        "class",
        formatter_class=RichHelpFormatter,
        help="Explain a class prediction.",
    )
    parser_class.add_argument(
        "class_id",
        action="store",
        help="The class to explain",
    )
    parser_class.set_defaults(run=run_class)

    parser_feature = commands.add_parser(
        "feature",
        formatter_class=RichHelpFormatter,
        help="Explain a feature importance.",
    )
    parser_feature.add_argument(
        "feature_id",
        action="store",
        help="The feature to explain"
    )
    parser_feature.set_defaults(run=run_feature)


def write_table(table: pandas.DataFrame, output: pathlib.Path):
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
        name = predictor.features_["name"][feature_index]
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
    raise KeyError(feature)


def run_class(args: argparse.Namespace, console: Console) -> int:
    predictor = load_model(args.model, console)

    # Extract requested class index
    try:
        class_index = get_class_index(args.class_id, predictor)
        name = predictor.classes_["name"][class_index]
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
