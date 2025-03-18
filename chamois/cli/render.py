import argparse
import contextlib
import pathlib
from typing import List, Set, Iterable

import numpy
import rich.tree
import rich.panel
from rich.console import Console
from rich.table import Table

from ..ontology import AdjacencyMatrix
from ..predictor import ChemicalOntologyPredictor
from ._common import load_model


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        help="The input probabilites obtained from the predictor."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=pathlib.Path,
        help="The path to an alternative predictor with classes metadata.",
    )
    parser.add_argument(
        "-p",
        "--pager",
        action="store_true",
        help="Use the given pager to display results."
    )
    parser.add_argument(
        "-c",
        "--color",
        action="store_true",
        help="Use colored input in pager."
    )
    parser.set_defaults(run=run)


def all_superclasses( classes: Iterable[int], adjacency_matrix: AdjacencyMatrix ) -> Set[int]:
    superclasses = set()
    classes = set(classes)
    while classes:
        i = classes.pop()
        superclasses.add(i)
        classes.update(j.item() for j in adjacency_matrix.parents(i))
        superclasses.update(j.item() for j in adjacency_matrix.parents(i))
    return superclasses


def build_tree(
    model: ChemicalOntologyPredictor,
    bgc_probas: numpy.ndarray,
) -> None:
    # get probabilities and corresponding positive terms from ChemOnt
    bgc_labels = bgc_probas > 0.5
    terms = { j for j in range(len(model.classes_)) if bgc_probas[j] > 0.5 }
    whitelist = all_superclasses(terms, model.ontology.adjacency_matrix)
    # render a tree structure with rich
    def render(i, tree, whitelist):
        term_id = model.classes_.index[i]
        term_name = model.classes_.name[i]
        color = "cyan" if bgc_probas[i] >= 0.5 else "yellow"
        label = f"[bold blue]{term_id}[/] ([green]{term_name}[/]): [bold {color}]{bgc_probas[i]:.3f}[/]"
        subtree = tree.add(label, highlight=False)
        for j in model.ontology.adjacency_matrix.children(i):
            j = j.item()
            if j in whitelist:
                render(j, subtree, whitelist)
    roots = [
        i
        for i in range(len(model.classes_.index))
        if not len(model.ontology.adjacency_matrix.parents(i))
        and bgc_probas[i] > 0.5
    ]
    tree = rich.tree.Tree(".", hide_root=True)
    for root in roots:
        render(root, tree, whitelist=whitelist)
    return tree


def run(args: argparse.Namespace, console: Console) -> int:
    import anndata

    # load trained model
    model = load_model(args.model, console)

    # load predictions
    console.print(f"[bold blue]{'Loading':>12}[/] probability predictions from {str(args.input)!r}")
    predictions = anndata.read_h5ad(args.input)

    # build table with all tree to have output with consistent width
    table = Table.grid()
    for bgc_index in range(predictions.n_obs):
        tree = build_tree(model, predictions.X[bgc_index])
        panel = rich.panel.Panel(tree, title=f"[bold purple]{predictions.obs_names[bgc_index]}[/]")
        table.add_row(panel)

    # render output
    with console.pager(styles=args.color) if args.pager else contextlib.nullcontext():
        console.print(table)
