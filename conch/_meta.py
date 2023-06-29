import functools
import importlib
import typing
from typing import Union, Callable, TypeVar


if typing.TYPE_CHECKING:
    _F = TypeVar("_F", bound=Callable[..., Any])


class requires:
    """A decorator for functions that require optional dependencies.
    """

    module: Union["ModuleType", BaseException]

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

        try:
            self.module = importlib.import_module(module_name)
        except ImportError as err:
            self.module = err

    def __call__(self, func: "_F") -> "_F":

        if isinstance(self.module, ImportError):

            @functools.wraps(func)
            def newfunc(*args, **kwargs):  # type: ignore
                msg = f"calling {func.__qualname__} requires module {self.module.name}"
                raise RuntimeError(msg) from self.module
        else:

            newfunc = func
            basename = self.module_name.split(".")[-1]
            newfunc.__globals__[basename] = self.module

        return newfunc # type: ignore