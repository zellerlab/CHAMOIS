import contextlib
import math
import typing
from typing import Tuple

import numpy
import torch.cuda.amp
import torch.nn
import sklearn.metrics
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import DataParallel, BCELoss, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch_treecrf import TreeCRF, TreeMatrix
from torchmetrics.functional.classification import multilabel_auroc


class AnnotatedDataSet:

    def __init__(self, X: Tensor, Y: Tensor) -> None:  # noqa: D107
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Length mismatch between features and labels")
        self.Y = Y
        self.X = X

    def __len__(self) -> int:  # noqa: D105
        return self.X.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:  # noqa: D105
        return self.X[index], self.Y[index]


class ClassyFireModel:
    
    def __init__(self):
        # use CPU device
        self.devices = [ torch.device("cpu") ]
        # cache the autocast context manager to use if we
        # are using CUDA devices
        if self.devices[0].type == 0:
            self._autocast: Callable[[], ContextManager[None]] = torch.cuda.amp.autocast
        else:
            self._autocast = contextlib.nullcontext

    def autocast(self) -> typing.ContextManager[None]:
        """Obtain a context manager for enabling mixed precision mode.
        """
        return self._autocast()

    @property
    def data_device(self):
        """`torch.device`: The device where to store the model data.
        """
        return self.devices[0]

    class TrainingIteration(typing.NamedTuple):
        """The statistics about a training iteration.
        """
        epoch: int
        total: int
        learning_rate: float
        loss: float
        micro_auroc: float
        macro_auroc: float

    def fit(
        self, 
        X, 
        Y,
        *,
        progress,
        hierarchy,
        test_X = None,
        test_Y = None,
        base_lr: float = 10,
        max_lr: float = 100,
        warmup_ratio: int = 100,
        batch_size: int = 32,
        anneal_strategy: str = "linear",
        epochs: int = 50,
    ):
        # Prepare training data
        _X = torch.Tensor(X)
        _Y = torch.Tensor(Y)
        data_loader = DataLoader(
            AnnotatedDataSet(_X, _Y),
            batch_size=batch_size,
        )

        # Prepare validation data
        if test_X is not None:
            _test_X = torch.asarray(test_X, dtype=torch.float)
            _test_Y = torch.asarray(test_Y, dtype=torch.float)
            assert _test_X.shape[1] == _X.shape[1]
            assert _test_Y.shape[1] == _Y.shape[1]
            assert _test_X.shape[0] == _test_Y.shape[0]

        # Initialize model with input dimensions
        # self.model = TreeCRF(_X.shape[1], hierarchy)
        self.model = torch.nn.Linear(_X.shape[1], _Y.shape[1])
        self.model.to(self.data_device)

        # compute pos / neg weights for cross-entropy
        pos = _Y.count_nonzero(axis=0) + 1e-9
        neg = _Y.shape[1] - pos

        # Setup the optimization framework
        penalty = torch.nn.L1Loss(size_average=False)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=base_lr)
        criterion = torch.nn.BCELoss() if isinstance(self.model, TreeCRF) else torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            max_lr=max_lr,
            pct_start=1/warmup_ratio,
            epochs=epochs,
            steps_per_epoch=len(data_loader),
            base_momentum=0,
            cycle_momentum=False,
            anneal_strategy=anneal_strategy,
            div_factor=max_lr/base_lr,
            final_div_factor=max_lr/base_lr,
        )

        # Record the best model with the highest loss, so that it can
        # be recovered after the training iterations have completed
        best_model_state = None
        best_loss = math.inf

        for epoch in range(epochs):
            # run the model and update weights with new gradients
            self.model.train()
            for index, (batch_X, batch_Y) in enumerate(data_loader):
                batch_X = batch_X.to(self.data_device)
                batch_Y = batch_Y.to(self.data_device)
                logits = self.model(batch_X)
                loss = criterion(logits, batch_Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # evaluate on validation set
            self.model.eval()
            if test_X is None:
                probas = self.model(_X).detach()
                loss = criterion(probas, _Y)
                micro_auroc = multilabel_auroc(probas, _Y, _Y.shape[1], average="micro")
                macro_auroc = multilabel_auroc(probas, _Y, _Y.shape[1], average="macro")
            else:
                probas = self.model(_test_X).detach()
                loss = criterion(probas, _test_Y)
                micro_auroc = multilabel_auroc(probas, _test_Y, _test_Y.shape[1], average="micro")
                macro_auroc = multilabel_auroc(probas, _test_Y, _test_Y.shape[1], average="macro")
            
            # Report progress using the callback provided in arguments
            progress(
                self.TrainingIteration(
                    epoch+1,
                    epochs,
                    #1e-1,
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


if __name__ == "__main__":

    import anndata
    import statistics
    import joblib
    import rich.progress
    import sklearn.linear_model

    # load data
    features = anndata.read("data/datasets/mibig3.1/pfam35.hdf5")
    classes = anndata.read("data/datasets/mibig3.1/classes.hdf5")

    # remove compounds with unknown structure
    features = features[~classes.obs.unknown_structure]
    classes = classes[~classes.obs.unknown_structure]
    # remove features absent from training set
    features = features[:, features.X.sum(axis=0).A1 > 0]
    # remove classes absent from training set
    classes = classes[:, classes.X.sum(axis=0).A1 > 0]

    #
    kfold = sklearn.model_selection.GroupKFold(n_splits=5)
    train_indices, test_indices = next(kfold.split(features.X, classes.X, groups=classes.obs["groups"]))
    # train_X, test_X, train_Y, test_Y = train_test_split(
    #     features.X.toarray(), 
    #     classes.X.toarray(), 
    #     test_size=0.25, 
    #     random_state=42,
    #     stratify=classes.obs["groups"],
    # )

    # build class hierarchy
    hierarchy = TreeMatrix(classes.varp["parents"].toarray())

    
    with rich.progress.Progress() as progress:

        task = progress.add_task("Training")
        def progress_callback(it) -> None:
            stats = [
                f"[bold magenta]lr=[/][bold cyan]{it.learning_rate:.2e}[/]",
                f"[bold magenta]loss=[/][bold cyan]{it.loss:.2f}[/]",
                f"[bold magenta]AUROC(µ)=[/][bold cyan]{it.micro_auroc:05.1%}[/]",
                f"[bold magenta]AUROC(M)=[/][bold cyan]{it.macro_auroc:05.1%}[/]",
            ]
            progress.update(task, advance=1, total=it.total)
            if (it.epoch - 1) % 1 == 0:
                progress.console.print(f"[bold blue]{'Training':>12}[/] epoch {it.epoch} of {it.total}:", *stats)

        model = ClassyFireModel()
        model.fit(
            features.X[train_indices].toarray(), 
            classes.X[train_indices].toarray(),
            test_X=features.X[test_indices].toarray(),
            test_Y=classes.X[test_indices].toarray(),
            progress=progress_callback,
            hierarchy=hierarchy
        )



