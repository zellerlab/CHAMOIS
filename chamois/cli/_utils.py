import contextlib
import warnings
import typing

if typing.TYPE_CHECKING:
    ShowWarning = typing.Callable[
        [typing.Union[Warning, str], typing.Type[Warning], str, int, typing.Optional[typing.TextIO], typing.Optional[str]],
        None
    ]

@contextlib.contextmanager
def patch_showwarnings(new_showwarning: "ShowWarning") -> typing.Iterator[None]:
    """Make a context patching `warnings.showwarning` with the given function.
    """
    old_showwarning: "ShowWarning" = warnings.showwarning
    try:
        warnings.showwarning = new_showwarning  # type: ignore
        yield
    finally:
        warnings.showwarning = old_showwarning  # type: ignore
