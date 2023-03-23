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
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from autochem.model import AutochemPredictor
from torch_treecrf import TreeMatrix

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
# standardize features
scaler = sklearn.preprocessing.StandardScaler()
features.X = scaler.fit_transform(features.X.toarray())

criterion = torch.nn.BCELoss()
kfold = sklearn.model_selection.GroupShuffleSplit(n_splits=5, random_state=SEED)
os.makedirs("build", exist_ok=True)


class LinearAutochem:

    def __init__(self):
        self.linear = None

    def predict_proba(self, X):
        """Predict probability estimates.
        """
        return torch.sigmoid(self.model(X))

    def fit(
        self, 
        X, 
        Y,
        test_X = None,
        test_Y = None,
        *,
        hierarchy = None,
        progress,
    ):
        # keep metadata from input if any
        if anndata is not None and isinstance(X, anndata.AnnData):
            self.feature_names_in_ = X.var_names
            X = X.X.toarray()
        if anndata is not None and isinstance(Y, anndata.AnnData):
            self.classes_ = Y.var_names
            Y = Y.X.toarray()

        # Prepare training data - no need for batching
        _X = torch.asarray(X, dtype=torch.float, device=self.data_device)
        _Y = torch.asarray(Y, dtype=torch.float, device=self.data_device)
        # Prepare validation data
        if test_X is not None:
            _test_X = torch.asarray(test_X, dtype=torch.float, device=self.data_device)
            _test_Y = torch.asarray(test_Y, dtype=torch.float, device=self.data_device)
            assert _test_X.shape[1] == _X.shape[1]
            assert _test_Y.shape[1] == _Y.shape[1]
            assert _test_X.shape[0] == _test_Y.shape[0]
        # Prepare hierarchy
        if not isinstance(hierarchy, TreeMatrix):
            hierarchy = TreeMatrix(hierarchy)

        # Initialize model with input dimensions
        self.model = torch.nn.Linear(_X.shape[1], _Y.shape[1], device=self.data_device, dtype=_X.dtype)

        # compute pos / neg weights for cross-entropy
        pos = _Y.count_nonzero(axis=0) + 1e-9
        neg = _Y.shape[1] - pos

        # Setup the optimization framework
        optimizer = torch.optim.ASGD(
            self.model.parameters(),
            weight_decay=0.0001,
            lr=self.base_lr
        )
        criterion = torch.nn.BCELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.data_device.type == "cuda")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            max_lr=self.max_lr,
            pct_start=self.warmup_percent,
            epochs=self.epochs,
            steps_per_epoch=1,
            base_momentum=0,
            cycle_momentum=False,
            anneal_strategy=self.anneal_strategy,
            div_factor=self.max_lr/self.base_lr,
            final_div_factor=self.max_lr/self.base_lr,
        )

        # Record the best model with the highest loss, so that it can
        # be recovered after the training iterations have completed
        best_model_state = None
        best_loss = math.inf

        for epoch in range(self.epochs):
            # run the model and update weights with new gradients
            self.model.train()
            optimizer.zero_grad()
            with self._autocast():
                probas = self.model(_X).exp()
                loss = criterion(probas, _Y)   
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # evaluate on validation set
            self.model.eval()
            if test_X is None:
                # probas = self.model(_X).detach()
                # loss = criterion(probas, _Y)
                micro_auroc = multilabel_auroc(probas, _Y.to(torch.float), _Y.shape[1], average="micro")
                macro_auroc = multilabel_auroc(probas, _Y.to(torch.float), _Y.shape[1], average="macro")
            else:
                probas = self.model(_test_X).exp().detach()
                loss = criterion(probas, _test_Y)
                micro_auroc = multilabel_auroc(probas, _test_Y.to(torch.float), _test_Y.shape[1], average="micro")
                macro_auroc = multilabel_auroc(probas, _test_Y.to(torch.float), _test_Y.shape[1], average="macro")
            
            # Report progress using the callback provided in arguments
            progress(
                self.TrainingIteration(
                    epoch+1,
                    self.epochs,
                    scheduler.get_last_lr()[0],
                    loss.item(),
                    micro_auroc,
                    macro_auroc
                )
            )
            # Record model
            best_model_state = self.model.state_dict()

        # After training is completed, recover the model with the
        # smallest loss using the locally stored model state.
        if best_model_state is None:
            raise RuntimeError("No best model found, training iterations were likely not successful")
        self.model.load_state_dict(best_model_state)


with rich.progress.Progress(*rich.progress.Progress.get_default_columns(), rich.progress.MofNCompleteColumn(), transient=True) as progress:
    
    # create array for predicted probabilities
    probas_crf = torch.zeros(classes.shape, dtype=torch.float, device=DEVICE)
    probas_lr = torch.zeros(classes.shape, dtype=torch.float, device=DEVICE)

    for i, (train_indices, test_indices) in enumerate(kfold.split(features.X, classes.X, classes.obs.groups)):

        # === CRF ===

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
            features[train_indices],
            classes[train_indices],
            # test_X=test_X,
            # test_Y=test_Y,
            progress=progress_callback,
            hierarchy=hierarchy,
        )

        # test fold
        probas_crf[test_indices] = model.predict_proba(features[test_indices])
  
        
        # === LR ===

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
        model = LinearAutochem()()
        model.fit(
            features[train_indices],
            classes[train_indices],
            # test_X=test_X,
            # test_Y=test_Y,
            progress=progress_callback,
            # hierarchy=hierarchy,
        )

        # test fold
        probas_lr[test_indices] = model.predict_proba(features[test_indices])
  


# Save results

predictions_lr = anndata.AnnData(
    X=probas_lr.detach().cpu().numpy(), 
    var=classes.var, 
    obs=classes.obs
)
predictions_lr.write(os.path.join("build", "lr.cv5_predictions.hdf5"))

predictions_crf = anndata.AnnData(
    X=probas_crf.detach().cpu().numpy(), 
    var=classes.var, 
    obs=classes.obs
)
predictions_lr.write(os.path.join("build", "crf.cv5_predictions.hdf5"))
        

# Plot results

plt.figure(1)

y_true = classes.X.toarray().ravel()
y_pred = probas_crf.cpu().detach().numpy().ravel()
auroc = roc_auc_score(y_true, y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, label="CRF ({:.3f})".format(auroc))

y_true = classes.X.toarray().ravel()
y_pred = probas_lr.cpu().detach().numpy().ravel()
auroc = roc_auc_score(y_true, y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, label="Linear Regression ({:.3f})".format(auroc))

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join("build", "crf_vs_lr.roc.svg"))


plt.figure(2)

y_true = classes.X.toarray().ravel()
y_pred = probas_crf.cpu().detach().numpy().ravel()
pr, rc, thresholds = roc_curve(y_true, y_pred)
aupr = auc(rc, pr)
plt.plot(rc, pr, label="CRF ({:.3f})".format(aupr))

y_true = classes.X.toarray().ravel()
y_pred = probas_lr.cpu().detach().numpy().ravel()
pr, rc, thresholds = roc_curve(y_true, y_pred)
aupr = auc(rc, pr)
plt.plot(fpr, tpr, label="Linear Regression ({:.3f})".format(aupr))

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig(os.path.join("build", "crf_vs_lr.pr.svg"))
