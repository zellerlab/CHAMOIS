# PYTHON_ARGCOMPLETE_OK

import argparse
from typing import Optional, List

from rich.console import Console
from rich_argparse import RichHelpFormatter

try:
    import argcomplete
except ImportError:
    argcomplete = None

from .. import __version__, __package__ as _module
from . import train, predict, render, cv, cvsearch, annotate, search, screen, explain, validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_module,
        formatter_class=RichHelpFormatter,
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

    commands = parser.add_subparsers(required=True, metavar="COMMAND")
    annotate.configure_parser(
        commands.add_parser(
            "annotate", 
            formatter_class=RichHelpFormatter,
            help="Annotate BGC records with CONCH features."
        ))
    cv.configure_parser(
        commands.add_parser(
            "cv", 
            formatter_class=RichHelpFormatter,
            help="Evaluate predictor performance on a training set with cross-validation."
        )
    )
    cvsearch.configure_parser(
        commands.add_parser(
            "cvsearch", 
            formatter_class=RichHelpFormatter,
            help="Evaluate compound search on a training set with cross-validation."
        )
    )
    predict.configure_parser(
        commands.add_parser(
            "predict", 
            formatter_class=RichHelpFormatter,
            help="Predict compound classes for BGC records."
        )
    )
    render.configure_parser(
        commands.add_parser(
            "render", 
            formatter_class=RichHelpFormatter,
            help="Render predicted class probabilities into a tree display."
        )
    )
    train.configure_parser(
        commands.add_parser(
            "train", 
            formatter_class=RichHelpFormatter,
            help="Train the CONCH predictor on a training dataset."
        )
    )
    search.configure_parser(
        commands.add_parser(
            "search", 
            formatter_class=RichHelpFormatter,
            help="Search a catalog for compounds similar to predicted classes."
        )
    )
    screen.configure_parser(
        commands.add_parser(
            "screen", 
            formatter_class=RichHelpFormatter,
            help="Search predicted classes for a particular compound.",
        )
    )
    explain.configure_parser(
        commands.add_parser(
            "explain",
            formatter_class=RichHelpFormatter,
            help="Explain the model weights between query classes and features."
        )
    )
    validate.configure_parser(
        commands.add_parser(
            "validate",
            formatter_class=RichHelpFormatter,
            help="Evaluate model on a validation dataset."
        )
    )

    return parser

def run(argv: Optional[List[str]] = None, console: Optional[Console] = None) -> int:
    if console is None:
        console = Console()

    parser = build_parser()
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)

    try:
        args.run(args, console)
    except Exception as err:
        console.print_exception()
        return getattr(err, "code", 1)
    else:
        return 0