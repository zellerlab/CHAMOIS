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

try:
    import anndata
except ImportError:
    anndata = None


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


class TreeCRF(torch.nn.Module):
    """A Tree-structured CRF for binary classification of labels.
    """

    def __init__(
        self, 
        n_features: int, 
        hierarchy: TreeMatrix,
        device=None,
        dtype=None,
    ):  
        super().__init__()
        self.linear = torch.nn.Linear(n_features, len(hierarchy), device=device, dtype=dtype)
        self.crf = TreeCRFLayer(hierarchy, device=device, dtype=dtype)
        torch.nn.init.zeros_(self.crf.pairs)

    def forward(self, X):
        emissions_pos = self.linear(X)
        emissions_all = torch.stack((-emissions_pos, emissions_pos), dim=2)
        return self.crf(emissions_all)[:, :, 1]


class AutochemPredictor:
    
    def __init__(
        self,
        base_lr: float = 1.0,
        max_lr: float = 10.0,
        warmup_percent: int = 0.1,
        anneal_strategy: str = "linear",
        epochs: int = 200,
    ):
        # record training parameters
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_percent = warmup_percent
        self.anneal_strategy = anneal_strategy
        self.epochs = epochs

        # record class / feature metadata
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.classes_ = None

        # use CPU device
        self.devices = [ 
            torch.device("cuda") 
            if torch.cuda.is_available() 
            else torch.device("cpu") 
        ]
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

    def predict_log_proba(self, X):
        """Predict logarithm of probability estimates.
        """
        if anndata is not None and isinstance(X, anndata.AnnData):
            X = X.X.toarray()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(dtype=torch.float, device=self.data_device)).detach()

    def predict_proba(self, X):
        """Predict probability estimates.
        """
        return self.predict_log_proba(X).exp()

    def fit(
        self, 
        X, 
        Y,
        test_X = None,
        test_Y = None,
        *,
        hierarchy,
        progress,
    ):
        # keep metadata from input if any
        if anndata is not None and isinstance(X, anndata.AnnData):
            self.feature_names_in_ = X.var_names
            X = X.X.toarray()
        if anndata is not None and isinstance(Y, anndata.AnnData):
            self.classes_ = Y.var_names
            Y = Y.X.toarray()
        # convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        if not isinstance(Y, torch.Tensor):
            Y = torch.Tensor(Y)
        # Prepare training data - no need for batching
        _X = X.to(dtype=torch.float, device=self.data_device)
        _Y = Y.to(dtype=torch.float, device=self.data_device)
        # Prepare validation data
        if test_X is not None:
            if not isinstance(test_X, torch.Tensor):
                test_X = torch.Tensor(test_X)
            if not isinstance(test_Y, torch.Tensor):
                test_Y = torch.Tensor(test_Y)   
            _test_X = test_X.to(dtype=torch.float, device=self.data_device)
            _test_Y = test_Y.to(dtype=torch.float, device=self.data_device)
            assert _test_X.shape[1] == _X.shape[1]
            assert _test_Y.shape[1] == _Y.shape[1]
            assert _test_X.shape[0] == _test_Y.shape[0]
        # Prepare hierarchy
        if not isinstance(hierarchy, TreeMatrix):
            hierarchy = TreeMatrix(hierarchy)

        # Initialize model with input dimensions
        self.model = TreeCRF(_X.shape[1], hierarchy, device=self.data_device, dtype=_X.dtype)
        # self.model = NoCRF(_X.shape[1], hierarchy, device=self.data_device, dtype=_X.dtype)

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
                micro_auroc = multilabel_auroc(probas, _Y.to(torch.long), _Y.shape[1], average="micro")
                macro_auroc = multilabel_auroc(probas, _Y.to(torch.long), _Y.shape[1], average="macro")
            else:
                self.model.eval()
                probas = self.model(_test_X).exp()
                loss = criterion(probas, _test_Y)
                micro_auroc = multilabel_auroc(probas, _test_Y.to(torch.long), _test_Y.shape[1], average="micro")
                macro_auroc = multilabel_auroc(probas, _test_Y.to(torch.long), _test_Y.shape[1], average="macro")
            
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

