import pathlib
from typing import Optional

from rich.console import Console

from ..predictor import ChemicalHierarchyPredictor


def load_model(path: Optional[pathlib.Path], console: Console) -> ChemicalHierarchyPredictor:
    if path is not None:
        console.print(f"[bold blue]{'Loading':>12}[/] trained model from {str(path)!r}")
        with open(path, "rb") as src:
            return ChemicalHierarchyPredictor.load(src)
    else:
        console.print(f"[bold blue]{'Loading':>12}[/] embedded model")
        return ChemicalHierarchyPredictor.trained()