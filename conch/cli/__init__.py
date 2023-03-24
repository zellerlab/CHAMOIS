import argparse
from typing import Optional

from rich.console import Console

from . import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    parser_train = commands.add_parser("train")
    train.configure_parser(parser_train)
    return parser

def run(console: Optional[Console] = None) -> int:
    if console is None:
        console = Console()
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.run(args, console)                
        return 0
    except Exception as err:
        console.print_exception()
        return getattr(err, "code", 1)