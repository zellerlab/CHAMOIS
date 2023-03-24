import itertools
import os
import sys

import anndata
import pandas
import rich.progress
import sklearn.linear_model
import sklearn.preprocessing
import torch
import matplotlib.pyplot as plt
from torch_treecrf import TreeMatrix
from torchmetrics.functional.classification import multilabel_auroc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from autochem.model import AutochemPredictor

ARCHITECTURES = ["crf", "lr"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

# load data
features = anndata.read("data/datasets/mibig3.1/pfam35.hdf5")
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

# group by identical compounds
groups = classes.obs.compound.cat.codes

# cross-validation folds
with rich.progress.Progress(*rich.progress.Progress.get_default_columns(), rich.progress.MofNCompleteColumn(), transient=True) as progress:
    # create array for predicted probabilities
    probas = { x: torch.zeros(classes.shape, dtype=torch.float, device=DEVICE) for x in ARCHITECTURES }
    # use group
    kfold = sklearn.model_selection.GroupShuffleSplit(n_splits=5, random_state=SEED)
    for i, (train_indices, test_indices) in enumerate(kfold.split(features.X, classes.X, groups)):
        for arc in ARCHITECTURES:
            # train fold
            task = progress.add_task(f"Fold {i+1}", total=None)
            def progress_callback(it) -> None:
                stats = [
                    f"[bold magenta]lr=[/][bold cyan]{it.learning_rate:.2e}[/]",
                    f"[bold magenta]loss)=[/][bold cyan]{it.loss:.2f}[/]",
                    f"[bold magenta]AUROC(µ)=[/][bold cyan]{it.micro_auroc:05.1%}[/]",
                    f"[bold magenta]AUROC(M)=[/][bold cyan]{it.macro_auroc:05.1%}[/]",
                ]
                progress.update(task, advance=1, total=it.total)
                if (it.epoch - 1) % 10 == 0:
                    progress.console.print(f"[bold blue]{'Training':>12}[/] epoch {it.epoch} of {it.total} for {arc.upper()} model:", *stats)
            model = AutochemPredictor(architecture=arc)
            model.fit(
                features[train_indices],
                classes[train_indices],
                # test_X=test_X,
                # test_Y=test_Y,
                progress=progress_callback,
                hierarchy=hierarchy,
            )
            # test fold
            probas[arc][test_indices] = model.predict_proba(features[test_indices])

# Save result predictions
os.makedirs("build", exist_ok=True)
for arc in ARCHITECTURES:
    data = anndata.AnnData(X=probas[arc].detach().cpu().numpy(), var=classes.var, obs=classes.obs)
    data.write(os.path.join("build", f"{arc}.cv5_predictions.hdf5"))

