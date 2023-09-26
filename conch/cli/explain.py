import argparse
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
        help="The path to an alternative model for predicting classes."
    )
    parser.add_argument(
        "--nonzero",
        default=False,
        action="store_true",
        help="Display non-zero weights instead of only positive weights."
    )

    commands = parser.add_subparsers(required=True)
    parser_class = commands.add_parser(
        "class",
        formatter_class=RichHelpFormatter,
        help="Explain a class prediction",
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
        help="Explain a feature importance",
    )
    parser_feature.add_argument(
        "feature_id",
        action="store",
        help="The feature to explain"
    )
    parser_feature.set_defaults(run=run_feature)


def run_feature(args: argparse.Namespace, console: Console) -> int:
    predictor = load_model(args.model, console)

    # Extract requested class index
    try:
        feature_index = predictor.features_.index.get_loc(args.feature_id)
        name = predictor.features_["name"][feature_index]
        console.print(f"[bold blue]{'Extracting':>12}[/] weights for class [bold blue]{args.feature_id}[/] ([green]{name}[/])")
    except KeyError:
        console.print(f"[bold red]{'Failed':>12}[/] to find class [bold blue]{args.feature_id}[/] in model")
        return 1

    # Extract positive weights
    weights = predictor.coef_[feature_index, :]
    indices = numpy.where(weights != 0.0 if args.nonzero else weights > 0)[0]
    selected_classes = predictor.classes_.iloc[indices].copy()
    selected_classes["weight"] = weights[indices]

    # Render the table
    table = rich.table.Table("ID", "Name", "Weight")
    for row in selected_classes.sort_values("weight", ascending=False).itertuples():
        table.add_row(
            rich.text.Text(row.Index, style="repr.tag_name"),
            row.name,
            rich.text.Text(format(row.weight, ".5f"), style="repr.number"),
        )
    console.print(table)

    return 0


def run_class(args: argparse.Namespace, console: Console) -> int:
    predictor = load_model(args.model, console)

    # Extract requested class index
    try:
        class_index = predictor.classes_.index.get_loc(args.class_id)
        name = predictor.classes_["name"][class_index]
        console.print(f"[bold blue]{'Extracting':>12}[/] weights for class [bold blue]{args.class_id}[/] ([green]{name}[/])")
    except KeyError:
        console.print(f"[bold red]{'Failed':>12}[/] to find class [bold blue]{args.class_id}[/] in model")
        return 1

    # Extract positive weights
    weights = predictor.coef_[:, class_index]
    indices = numpy.where(weights != 0.0 if args.nonzero else weights > 0)[0]
    selected_classes = predictor.features_.iloc[indices].copy()
    selected_classes["weight"] = weights[indices]

    # Render the table
    table = rich.table.Table("Feature", "Kind", "Name", "Weight")
    for row in selected_classes.sort_values("weight", ascending=False).itertuples():
        table.add_row(
            rich.text.Text(row.Index, style="repr.tag_name"),
            row.kind,
            row.name,
            rich.text.Text(format(row.weight, ".5f"), style="repr.number"),
        )
    console.print(table)

    return 0