import argparse
from typing import Optional

from rich.console import Console

from . import train, predict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("-j", "--jobs", default=None, type=int, help="The number of jobs to run in parallel sections.")

    commands = parser.add_subparsers(required=True)
    parser_train = commands.add_parser("train")
    train.configure_parser(parser_train)
    parser_predict = commands.add_parser("predict")
    predict.configure_parser(parser_predict)
    
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