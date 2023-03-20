import contextlib
import math
import typing
from typing import Tuple

import numpy
import torch.cuda.amp
import torch.nn
import torchmetrics.classification
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from torch import Tensor
from torch.nn import DataParallel, BCELoss, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch_treecrf import TreeCRF, TreeCRFLayer, TreeMatrix
from torchmetrics.functional.classification import multilabel_auroc, binary_auroc, binary_precision


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


class NoCRF(torch.nn.Module):

    def __init__(self, n_features, hierarchy, device=None, dtype=None) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, len(hierarchy)).to(device=device, dtype=dtype)

    def forward(self, X):
        return torch.sigmoid(self.linear(X))


class AutochemPredictor:
    
    def __init__(self):
        # use CPU device
        self.devices = [ torch.device("cuda") ]
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

    def predict_probas(self, X):
        return self.model(torch.asarray(X, dtype=torch.float, device=self.data_device))

    def fit(
        self, 
        X, 
        Y,
        *,
        progress,
        hierarchy,
        test_X = None,
        test_Y = None,
        base_lr: float = 1.0,
        max_lr: float = 10.0,
        warmup_ratio: int = 10,
        anneal_strategy: str = "linear",
        epochs: int = 50,
    ):
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
        self.model = TreeCRF(_X.shape[1], hierarchy, device=self.data_device, dtype=_X.dtype)
        torch.nn.init.zeros_(self.model.crf.pairs)
        # self.model = NoCRF(_X.shape[1], hierarchy, device=self.data_device, dtype=_X.dtype)

        # compute pos / neg weights for cross-entropy
        pos = _Y.count_nonzero(axis=0) + 1e-9
        neg = _Y.shape[1] - pos

        # Setup the optimization framework
        optimizer = torch.optim.ASGD(
            self.model.parameters(),
            weight_decay=0.0001,
            lr=base_lr
        )
        criterion = torch.nn.BCELoss()
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            max_lr=max_lr,
            pct_start=1/warmup_ratio,
            epochs=epochs,
            steps_per_epoch=1,
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
            optimizer.zero_grad()
            with self._autocast():
                probas = self.model(_X)
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
                micro_auroc = multilabel_auroc(probas, _Y.to(torch.long), _Y.shape[1], average="micro")
                macro_auroc = multilabel_auroc(probas, _Y.to(torch.long), _Y.shape[1], average="macro")
            else:
                probas = self.model(_test_X).detach()
                loss = criterion(probas, _test_Y)
                micro_auroc = multilabel_auroc(probas, _test_Y.to(torch.long), _test_Y.shape[1], average="micro")
                macro_auroc = multilabel_auroc(probas, _test_Y.to(torch.long), _test_Y.shape[1], average="macro")
            
            # Report progress using the callback provided in arguments
            progress(
                self.TrainingIteration(
                    epoch+1,
                    epochs,
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

