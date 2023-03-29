import argparse
import pathlib

import anndata
import rich.progress
from rich.console import Console

from ..predictor import ChemicalHierarchyPredictor


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-f", "--features", required=True, type=pathlib.Path)
    parser.add_argument("-c", "--classes", required=True, type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    parser.add_argument("-e", "--epochs", type=int, default=200, help="The number of epochs to train the model for.")
    parser.add_argument("--report-period", type=int, default=20, help="Report evaluation metrics every N iterations.")
    parser.set_defaults(run=run)


def run(args: argparse.Namespace, console: Console) -> int:
    # load data
    console.print(f"[bold blue]{'Loading':>12}[/] training data")
    features = anndata.read(args.features)
    classes = anndata.read(args.classes)
    # remove compounds with unknown structure
    features = features[~classes.obs.unknown_structure]
    classes = classes[~classes.obs.unknown_structure]
    # remove features absent from training set
    features = features[:, features.X.sum(axis=0).A1 > 0]
    # remove clases absent from training set
    classes = classes[:, (classes.X.sum(axis=0).A1 >= 5) & (classes.X.sum(axis=0).A1 <= classes.n_obs - 5)]
    # prepare class hierarchy
    hierarchy = classes.varp["parents"].toarray()
    
    # start training
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(), 
        rich.progress.MofNCompleteColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[bold blue]{'Training':>12}[/]", total=None, start=False)
        def progress_callback(it) -> None:
            stats = [
                f"[bold magenta]lr=[/][bold cyan]{it.learning_rate:.2e}[/]",
                f"[bold magenta]loss=[/][bold cyan]{it.loss:.2f}[/]",
                f"[bold magenta]AUROC(µ)=[/][bold cyan]{it.micro_auroc:05.1%}[/]",
                f"[bold magenta]AUROC(M)=[/][bold cyan]{it.macro_auroc:05.1%}[/]",
            ]
            if it.epoch == 0:
                progress.start_task(task)
            progress.update(task, completed=it.epoch, total=it.total)
            if (it.epoch - 1) % args.report_period == 0:
                progress.console.print(f"[bold blue]{'Training':>12}[/] epoch {it.epoch} of {it.total} for {model.architecture.upper()} model:", *stats)
        progress.console.print(f"[bold blue]{'Pretraining':>12}[/] linear layer of the CRF")
        model = ChemicalHierarchyPredictor(epochs=args.epochs, devices=args.device or None)
        model.fit(features, classes, callback=progress_callback, hierarchy=hierarchy)

    # save result
    progress.console.print(f"[bold blue]{'Saving':>12}[/] trained model to {str(args.output)!r}")
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as dst:
        model.save(dst)
    progress.console.print(f"[bold green]{'Finished':>12}[/] training model")


