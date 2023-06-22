import argparse
from typing import Optional

from rich.console import Console
from rich_argparse import RichHelpFormatter

from . import train, predict, render, cv, annotate, search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)

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

    commands = parser.add_subparsers(required=True)
    annotate.configure_parser(commands.add_parser("annotate"))
    cv.configure_parser(commands.add_parser("cv"))
    predict.configure_parser(commands.add_parser("predict"))
    render.configure_parser(commands.add_parser("render"))
    train.configure_parser(commands.add_parser("train"))
    search.configure_parser(commands.add_parser("search"))

    return parser

def run(console: Optional[Console] = None) -> int:
    if console is None:
        console = Console()

    parser = build_parser()
    args = parser.parse_args()

    try:
        args.run(args, console)
    except Exception as err:
        console.print_exception()
        return getattr(err, "code", 1)
    else:
        return 0