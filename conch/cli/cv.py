import argparse
import pathlib

import anndata
import rich.progress
import torch
import sklearn.model_selection
from rich.console import Console
from torchmetrics.functional.classification import multilabel_auroc, binary_auroc, binary_precision

from ..predictor import ChemicalHierarchyPredictor


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-f", "--features", required=True, type=pathlib.Path)
    parser.add_argument("-c", "--classes", required=True, type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    parser.add_argument("-e", "--epochs", type=int, default=200, help="The number of epochs to train the model for.")
    parser.add_argument("-k", "--kfolds", type=int, default=10, help="Number of cross-validation folds to run.")
    parser.add_argument("--report-period", type=int, default=20, help="Report evaluation metrics every N iterations.")
    parser.add_argument("--architecture", choices={"lr", "crf"}, default="crf", help="Model architecture to use for predicting classes")
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
    # prepare class hierarchy and groups
    hierarchy = classes.varp["parents"].toarray()
    groups = classes.obs["compound"].cat.codes

    # get devices from a CLI arguments
    if args.device:
        devices = args.device
    elif torch.cuda.is_available():
        devices = [torch.device("cuda")]
    else:
        devices = [torch.device("cpu")]

    # start training
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(),
        rich.progress.MofNCompleteColumn(),
        console=console
    ) as progress:
        kfold = sklearn.model_selection.GroupShuffleSplit(n_splits=args.kfolds, random_state=args.seed)
        probas = torch.zeros(classes.X.shape, dtype=torch.float, device=devices[0])
        for i, (train_indices, test_indices) in enumerate(kfold.split(features.X, classes.X, groups)):
            model = ChemicalHierarchyPredictor(epochs=args.epochs, devices=devices, architecture=args.architecture)
            # split data
            train_X = torch.tensor(features.X[train_indices].toarray(), dtype=torch.float, device=devices[0])
            train_Y = torch.tensor(classes.X[train_indices].toarray(), dtype=torch.float, device=devices[0])
            test_X = torch.tensor(features.X[test_indices].toarray(), dtype=torch.float, device=devices[0])
            test_Y = torch.tensor(classes.X[test_indices].toarray(), dtype=torch.float, device=devices[0])
            # train fold
            task = progress.add_task(f"[bold blue]{'Training':>12}[/]", total=args.epochs, start=False)
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
            model.fit(train_X, train_Y, callback=progress_callback, hierarchy=hierarchy)
            # test fold
            probas[test_indices] = model.predict_proba(test_X)
            # report test result
            loss = torch.nn.functional.binary_cross_entropy(probas[test_indices], test_Y)
            micro_auroc = multilabel_auroc(probas[test_indices], test_Y.to(torch.long), test_Y.shape[1], average="micro")
            macro_auroc = multilabel_auroc(probas[test_indices], test_Y.to(torch.long), test_Y.shape[1], average="macro")
            stats = [
                f"[bold magenta]loss(crf)=[/][bold cyan]{loss:.2f}[/]",
                f"[bold magenta]AUROC(µ)=[/][bold cyan]{micro_auroc:05.1%}[/]",
                f"[bold magenta]AUROC(M)=[/][bold cyan]{macro_auroc:05.1%}[/]",
            ]
            progress.console.print(f"[bold green]{'Finished':>12}[/] fold {i+1}:", *stats)

    # save predictions
    progress.console.print(f"[bold blue]{'Saving':>12}[/] cross-validation predictions to {str(args.output)!r}")
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    data = anndata.AnnData(
        X=probas.detach().cpu().numpy(),
        obs=classes.obs,
        var=classes.var,
    )
    data.write(args.output)
    progress.console.print(f"[bold green]{'Finished':>12}[/] cross-validating model")


