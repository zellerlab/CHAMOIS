# PYTHON_ARGCOMPLETE_OK

import argparse
import functools
from typing import Optional, List, TextIO, Type

from rich.console import Console
from rich_argparse import ArgumentDefaultsRichHelpFormatter

try:
    import argcomplete
except ImportError:
    argcomplete = None

from .. import __version__, __package__ as _module
from . import train, predict, render, cv, cvi, cvsearch, annotate, search, compare, explain, validate
from ._utils import patch_showwarnings


def _showwarnings(
    console: Console,
    message: str,
    category: Type[Warning],
    filename: str,
    lineno: int,
    file: Optional[TextIO] = None,
    line: Optional[str] = None,
) -> None:
    for line in filter(str.strip, str(message).splitlines()):
        verb, *rest = line.strip()
        console.print(f"[bold yellow]{'Warning':>12}[/]", line)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_module,
        formatter_class=ArgumentDefaultsRichHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"{_module} {__version__}",
        help="Show the program version number and exit.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=None,
        type=int,
        help="The number of jobs to run in parallel sections."
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="The seed to use for initializing pseudo-random number generators."
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Disable the console output",
    )
    parser.add_argument(
        "--no-color",
        dest="color",
        action="store_false",
        help="Disable the console color",
    )
    parser.add_argument(
        "--no-markup",
        dest="markup",
        action="store_false",
        help="Disable the console markup",
    )

    commands = parser.add_subparsers(required=True, metavar="COMMAND")
    annotate.configure_parser(
        commands.add_parser(
            "annotate", 
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Annotate BGC records with CHAMOIS features."
        ))
    cv.configure_parser(
        commands.add_parser(
            "cv", 
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Evaluate predictor performance on a training set with cross-validation."
        )
    )
    cvi.configure_parser(
        commands.add_parser(
            "cvi",
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Evalute predictor performance on a training set with independent cross-validations."
        )
    )
    # cvsearch.configure_parser(
    #     commands.add_parser(
    #         "cvsearch", 
    #         formatter_class=ArgumentDefaultsRichHelpFormatter,
    #         help="Evaluate compound search on a training set with cross-validation."
    #     )
    # )
    predict.configure_parser(
        commands.add_parser(
            "predict", 
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Predict compound classes for BGC records."
        )
    )
    render.configure_parser(
        commands.add_parser(
            "render", 
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Render predicted class probabilities into a tree display."
        )
    )
    train.configure_parser(
        commands.add_parser(
            "train", 
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Train the CHAMOIS predictor on a training dataset."
        )
    )
    search.configure_parser(
        commands.add_parser(
            "search", 
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Search a catalog for compounds similar to predicted classes."
        )
    )
    compare.configure_parser(
        commands.add_parser(
            "compare", 
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Compare predicted classes to a query compound.",
        )
    )
    explain.configure_parser(
        commands.add_parser(
            "explain",
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Explain the model weights between query classes and features."
        )
    )
    validate.configure_parser(
        commands.add_parser(
            "validate",
            formatter_class=ArgumentDefaultsRichHelpFormatter,
            help="Evaluate model on a validation dataset."
        )
    )

    return parser

def run(argv: Optional[List[str]] = None, console: Optional[Console] = None) -> int:
    parser = build_parser()
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)

    if console is None:
        console = Console(
            legacy_windows=not args.markup, 
            no_color=not args.color, 
            quiet=args.quiet, 
            safe_box=not args.markup
        )

    with patch_showwarnings(functools.partial(_showwarnings, console)):
        try:
            return args.run(args, console)
        except Exception as err:
            console.print_exception()
            return getattr(err, "code", 1)
        else:
            return 0
