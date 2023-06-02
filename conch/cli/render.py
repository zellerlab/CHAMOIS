import argparse
import pathlib
from typing import List, Set, Iterable

import anndata
import numpy
import rich.tree
import rich.panel
from rich.console import Console

from ..treematrix import TreeMatrix
from ..predictor import ChemicalHierarchyPredictor
from .predict import load_model


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-m", "--model", type=pathlib.Path)
    parser.add_argument("-i", "--input", required=True, type=pathlib.Path)
    parser.set_defaults(run=run)


def all_superclasses( classes: Iterable[int], hierarchy: TreeMatrix ) -> Set[int]:
    superclasses = set()
    classes = set(classes)
    while classes:
        i = classes.pop()
        superclasses.add(i)
        classes.update(j.item() for j in hierarchy.parents(i))
        superclasses.update(j.item() for j in hierarchy.parents(i))
    return superclasses


def build_tree(
    model: ChemicalHierarchyPredictor, 
    bgc_index: int,
    bgc_probas: numpy.ndarray, 
) -> None:
    # get probabilities and corresponding positive terms from ChemOnt
    bgc_labels = bgc_probas > 0.5
    terms = { j for j in range(len(model.classes_)) if bgc_probas[j] > 0.5 }
    whitelist = all_superclasses(terms, model.hierarchy)
    # render a tree structure with rich
    def render(i, tree, whitelist):
        term_id = model.classes_.index[i]
        term_name = model.classes_.name[i]
        label = f"[bold blue]{term_id}[/] ([green]{term_name}[/]): [bold cyan]{bgc_probas[i]:.3f}[/]"
        subtree = tree.add(label, highlight=False)
        for j in model.hierarchy.children(i):
            j = j.item()
            if j in whitelist:
                render(j, subtree, whitelist)
    roots = [
        i 
        for i in range(len(model.classes_.index)) 
        if not len(model.hierarchy.parents(i)) 
        and bgc_probas[i] > 0.5
    ]
    tree = rich.tree.Tree(".", hide_root=True)
    for root in roots:
        render(root, tree, whitelist=whitelist)
    return tree


def run(args: argparse.Namespace, console: Console) -> int:
    model = load_model(args.model, console)

    # load predictions
    console.print(f"[bold blue]{'Loading':>12}[/] probability predictions from {str(args.input)!r}")
    predictions = anndata.read(args.input)

    # render tree
    for bgc_index in range(predictions.n_obs):
        tree = build_tree(model, bgc_index, predictions.X[bgc_index])
        panel = rich.panel.Panel(tree, title=predictions.obs_names[bgc_index])
        console.print(panel)
