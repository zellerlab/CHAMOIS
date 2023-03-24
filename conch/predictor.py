import contextlib
import json
import math
import typing
from typing import Literal, Tuple, Dict

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


class ChemicalHierarchyPredictor:
    """A model for predicting chemical hierarchy from BGC compositions.
    """

    class TrainingIteration(typing.NamedTuple):
        """The statistics about a training iteration.
        """
        epoch: int
        total: int
        learning_rate: float
        loss: float
        micro_auroc: float
        macro_auroc: float

    def __init__(
        self,
        architecture: Literal["crf", "lr"] = "crf",
        base_lr: float = 1.0,
        max_lr: float = 10.0,
        warmup_percent: int = 0.1,
        anneal_strategy: str = "linear",
        epochs: int = 200,
    ):
        # record model architecture
        self.architecture = architecture
        self.model = None
        self.output_function = None

        # record training parameters
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_percent = warmup_percent
        self.anneal_strategy = anneal_strategy
        self.epochs = epochs

        # record class / feature metadata
        self.features = None
        self.classes = None

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

    def __getstate__(self) -> Dict[str, Any]:
        state = {
            "architecture": self.architecture,
            "base_lr": self.base_lr,
            "max_lr": self.max_lr,
            "warmup_percent": self.warmup_percent,
            "anneal_strategy": self.anneal_strategy,
            "epochs": self.epochs,
            "model": None,
            "features": None,
            "classes": None,
        }
        if self.model is not None:
            state["model"] = self.model.state_dict()
        if self.features is not None:
            state["features"] = self.features
        if self.classes is not None:
            state["classes"] = self.classes``
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        model = state.pop("model")
        features = state.pop("features")
        classes = state.pop("classes")
        self.__init__(**state)
        if model is not None:
            self.model.load_state_dict(model)
        self.features = features
        self.classes = classes

    def _initialize_model(self, n_features, n_labels, hierarchy):
        if self.architecture == "crf":
            self.model = TreeCRF(n_features, hierarchy, device=self.data_device, dtype=torch.float)
            self.output_function = torch.exp
        elif self.architecture == "lr":
            self.model = torch.nn.Linear(n_features, n_labels, device=self.data_device, dtype=torch.float)
            self.output_function = torch.sigmoid
        else:
            raise ValueError(f"Invalid model architecture: {self.architecture!r}")

    @property
    def data_device(self):
        """`torch.device`: The device where to store the model data.
        """
        return self.devices[0]

    def predict_proba(self, X):
        """Predict probability estimates.
        """
        if anndata is not None and isinstance(X, anndata.AnnData):
            X = X.X.toarray()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        self.model.eval()
        with torch.no_grad():
            _X = X.to(dtype=torch.float, device=self.data_device)
            return self.output_function(self.model(_X)).detach()

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
        if self.architecture == "crf" and not isinstance(hierarchy, TreeMatrix):
            hierarchy = TreeMatrix(hierarchy)

        # Initialize model with input dimensions and label hierarchy
        self._initialize_model(_X.shape[1], _Y.shape[1], hierarchy)

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
                probas = self.output_function(self.model(_X))
                loss = criterion(probas, _Y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # evaluate on validation set
            self.model.eval()
            if test_X is None:
                micro_auroc = multilabel_auroc(probas, _Y.to(torch.long), _Y.shape[1], average="micro")
                macro_auroc = multilabel_auroc(probas, _Y.to(torch.long), _Y.shape[1], average="macro")
            else:
                self.model.eval()
                probas = self.output_function(self.model(_test_X))
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

    @classmethod
    def load(cls, file: BinaryIO) -> ChemicalHierarchyPredictor:
        predictor = cls()
        predictor.__setstate__(json.load(file))
        return predictor

    def save(self, file: BinaryIO) -> None:
        state = self.__getstate__()
        json.dump(state, file)