import itertools
import os
import sys

import anndata
import pandas
import rich.progress
import sklearn.linear_model
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from autochem.model import AutochemPredictor
from torch_treecrf import TreeMatrix

DEVICE = "cuda:0"
SEED = 42
torch.manual_seed(SEED)

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

feature_sets = {
    "pfam35": anndata.read("data/datasets/mibig3.1/pfam35.hdf5"),
    "kofam2023": anndata.read("data/datasets/mibig3.1/kofam2023.hdf5"),
    "pgap11": anndata.read("data/datasets/mibig3.1/pgap11.hdf5"),
    "smcogs6": anndata.read("data/datasets/mibig3.1/smcogs6.hdf5"),
}
predicted = {

}


criterion = torch.nn.BCELoss()
kfold = sklearn.model_selection.GroupShuffleSplit(n_splits=5, random_state=SEED)
os.makedirs("build", exist_ok=True)

for feature_set in filter(None, powerset(feature_sets)):
    rich.print(f"[bold blue]{'Starting':>12}[/] training with feature set: [bold magenta]{', '.join(feature_set)}[/]")

    with rich.progress.Progress(*rich.progress.Progress.get_default_columns(), rich.progress.MofNCompleteColumn(), transient=True) as progress:
        # create feature combination
        features = anndata.concat([ feature_sets[x] for x in feature_set ], axis=1)
        classes = anndata.read("data/datasets/mibig3.1/classes.hdf5")
        # remove compounds with unknown structure
        features = features[~classes.obs.unknown_structure]
        classes = classes[~classes.obs.unknown_structure]
        # remove features absent from training set
        features = features[:, features.X.sum(axis=0).A1 > 0]
        # remove classes absent from training set
        classes = classes[:, (classes.X.sum(axis=0).A1 >= 5) & (classes.X.sum(axis=0).A1 <= classes.n_obs - 5)]
        # prepare class hierarchy
        hierarchy = classes.varp["parents"].toarray()
        # create array for predicted probabilities
        predicted[feature_set] = torch.zeros(classes.shape, dtype=torch.float, device=DEVICE)

        for i, (train_indices, test_indices) in enumerate(kfold.split(features.X, classes.X, classes.obs.groups)):
            # split data
            train_X = torch.tensor(features.X[train_indices].toarray(), dtype=torch.float, device=DEVICE)
            train_Y = torch.tensor(classes.X[train_indices].toarray(), dtype=torch.float, device=DEVICE)
            test_X = torch.tensor(features.X[test_indices].toarray(), dtype=torch.float, device=DEVICE)
            test_Y = torch.tensor(classes.X[test_indices].toarray(), dtype=torch.float, device=DEVICE)

            # train fold
            task = progress.add_task(f"Fold {i+1}", total=None)
            def progress_callback(it) -> None:
                stats = [
                    f"[bold magenta]lr=[/][bold cyan]{it.learning_rate:.2e}[/]",
                    f"[bold magenta]loss(crf)=[/][bold cyan]{it.loss:.2f}[/]",
                    f"[bold magenta]AUROC(µ)=[/][bold cyan]{it.micro_auroc:05.1%}[/]",
                    f"[bold magenta]AUROC(M)=[/][bold cyan]{it.macro_auroc:05.1%}[/]",
                ]
                progress.update(task, advance=1, total=it.total)
                if (it.epoch - 1) % 10 == 0:
                    progress.console.print(f"[bold blue]{'Training':>12}[/] epoch {it.epoch} of {it.total}:", *stats)
            model = AutochemPredictor()
            model.fit(
                train_X, 
                train_Y,
                # test_X=test_X,
                # test_Y=test_Y,
                progress=progress_callback,
                hierarchy=hierarchy,
                epochs=200,
                # base_lr=1,
                # max_lr=100,
            )

            # test fold
            probas_lin = torch.sigmoid(model.model.linear(test_X))
            probas_crf = model.predict(test_X)
            rows = []
            for j in range( classes.n_vars ):
                if test_Y[:, j].sum() == 0 or test_Y[:, j].sum() == test_Y.shape[0]:
                    continue
                n_pos = classes.X[:, j].sum()
                auroc_lin = binary_auroc(probas_lin[:, j], test_Y[:, j].to(torch.long))
                auroc_crf = binary_auroc(probas_crf[:, j], test_Y[:, j].to(torch.long))
                rows.append((classes.var.index[j], classes.var.name[j], n_pos, auroc_lin.item(), auroc_crf.item()))

            # report test result
            loss = criterion(probas_crf, test_Y)
            micro_auroc = multilabel_auroc(probas_crf, test_Y.to(torch.long), test_Y.shape[1], average="micro")
            macro_auroc = multilabel_auroc(probas_crf, test_Y.to(torch.long), test_Y.shape[1], average="macro")
            stats = [
                f"[bold magenta]loss(crf)=[/][bold cyan]{loss:.2f}[/]",
                f"[bold magenta]AUROC(µ)=[/][bold cyan]{micro_auroc:05.1%}[/]",
                f"[bold magenta]AUROC(M)=[/][bold cyan]{macro_auroc:05.1%}[/]",
            ]
            progress.console.print(f"[bold green]{'Finished':>12}[/] fold {i+1}:", *stats)

            # record predictions on test fold
            predicted[feature_set][test_indices] = probas_crf

            # report fold statistics
            table = pandas.DataFrame(rows, columns=["id", "name", "positives", "auroc_lin", "auroc_crf"])
            table.sort_values("auroc_crf", inplace=True, ascending=False)
            # for row in table.head(10).itertuples():
            #     progress.console.print(f"- [bold cyan]{row.id}[/] ({row.name}): [bold magenta]positives=[/][bold cyan]{row.positives}[/] [bold magenta]AUROC(lr)=[/][bold cyan]{row.auroc_lin:.6f}[/] [bold magenta]AUROC(crf)=[/][bold cyan]{row.auroc_crf:.6f}[/] [bold magenta]Precision(crf)=[/][bold cyan]{row.precision_crf:.6f}[/]")
            # progress.console.print("...")
            # for row in table.tail(10).itertuples():
            #     progress.console.print(f"- [bold cyan]{row.id}[/] ({row.name}): [bold magenta]positives=[/][bold cyan]{row.positives}[/] [bold magenta]AUROC(lr)=[/][bold cyan]{row.auroc_lin:.6f}[/] [bold magenta]AUROC(crf)=[/][bold cyan]{row.auroc_crf:.6f}[/] [bold magenta]Precision(crf)=[/][bold cyan]{row.precision_crf:.6f}[/]")

            # save classifier
            table.to_csv(os.path.join("build", f"treecrf.{'+'.join(feature_set)}.fold{i+1}.tsv"), sep="\t")
            torch.save(model.model.state_dict(), os.path.join("build", f"treecrf.{'+'.join(feature_set)}.fold{i+1}.pt"))


    for feature_set, probas in predicted.items():
        y_true = classes.X.toarray().ravel()
        y_pred = probas.cpu().detach().numpy().ravel()
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label="{} ({:.3f})".format('+'.join(feature_set), auc))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join("build", "roc.svg"))
    plt.show()