import contextlib
import functools
import importlib
import io
import pathlib
import sys
import traceback
import types
import typing
from typing import (
    BinaryIO,
    Iterator,
    Union, 
    Callable, 
    TypeVar
)

try:
    from isal import igzip as gzip
except ImportError:
    import gzip

try:
    import lz4.frame
except ImportError as err:
    lz4 = err

try:
    import bz2
except ImportError as err:
    bz2 = err

try:
    import lzma
except ImportError as err:
    lzma = err


if typing.TYPE_CHECKING:
    _F = TypeVar("_F", bound=Callable[..., Any])


_BZ2_MAGIC = b"BZh"
_GZIP_MAGIC = b"\x1f\x8b"
_XZ_MAGIC = b"\xfd7zXZ"
_LZ4_MAGIC = b"\x04\x22\x4d\x18"


class RequiredModuleError(RuntimeError):
    """A required module is missing.
    """

    def __init__(self, module, name, module_name):
        self.module = module
        self.name = name
        self.module_name = module_name

    def __str__(self):
        return f"calling {self.module}.{self.name} requires {self.module_name} module"


class requires:
    """A decorator for functions that require optional dependencies.
    """

    module: Union["ModuleType", BaseException]

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.module = None

    def __call__(self, func: "_F") -> "_F":

        @functools.wraps(func)
        def newfunc(*args, **kwargs):  # type: ignore
            # attempt to import the required module
            if self.module is None:
                try:
                    self.module = importlib.import_module(self.module_name)
                except ImportError as err:
                    self.module = err

            # raise an error if module is missing
            if isinstance(self.module, ImportError):
                raise RequiredModuleError(func.__module__, func.__qualname__, self.module_name) from self.module

            # get back to the original function (traversing wrapper)
            wrapped = func
            while hasattr(wrapped, "__wrapped__"):
                wrapped = wrapped.__wrapped__

            # set the globals on the original function
            basename = self.module_name.split(".")[0]
            wrapped.__globals__[basename] = sys.modules[basename]

            # call the function with the given arguments
            return func(*args, **kwargs)

        return newfunc  # type: ignore


@contextlib.contextmanager
def zopen(path: Union[str, pathlib.Path]) -> Iterator[BinaryIO]:
    """Open a file with optional compression in binary mode.
    """
    with contextlib.ExitStack() as ctx:
        if isinstance(path, (str, pathlib.Path)):
            file = ctx.enter_context(open(path, "rb"))
        else:
            file = ctx.enter_context(io.BufferedReader(path))
        peek = file.peek()
        if peek.startswith(_GZIP_MAGIC):
            file = ctx.enter_context(gzip.open(file, mode="rb"))
        elif peek.startswith(_BZ2_MAGIC):
            if isinstance(bz2, ImportError):
                raise RuntimeError("File compression is LZMA but lzma is not available") from lz4
            file = ctx.enter_context(bz2.open(file, mode="rb"))
        elif peek.startswith(_XZ_MAGIC):
            if isinstance(lzma, ImportError):
                raise RuntimeError("File compression is LZMA but lzma is not available") from lz4
            file = ctx.enter_context(lzma.open(file, mode="rb"))
        elif peek.startswith(_LZ4_MAGIC):
            if isinstance(lz4, ImportError):
                raise RuntimeError("File compression is LZ4 but python-lz4 is not installed") from lz4
            file = ctx.enter_context(lz4.frame.open(file))
        yield file