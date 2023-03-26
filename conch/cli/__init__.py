import argparse
from typing import Optional

import torch
from rich.console import Console

from . import train, predict, render, cv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("-j", "--jobs", default=None, type=int, help="The number of jobs to run in parallel sections.")
    parser.add_argument("--seed", default=0, type=int, help="The seed to use for initializing pseudo-random number generators.")
    parser.add_argument(
        "-D", 
        "--device", 
        action="append",
        default=[], 
        type=torch.device, 
        help="The device to use for running the PyTorch model."
    )

    commands = parser.add_subparsers(required=True)
    train.configure_parser(commands.add_parser("train"))
    predict.configure_parser(commands.add_parser("predict"))
    render.configure_parser(commands.add_parser("render"))
    cv.configure_parser(commands.add_parser("cv"))

    return parser

def run(console: Optional[Console] = None) -> int:
    if console is None:
        console = Console()
    
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.jobs is not None:
            torch.set_num_threads(args.jobs)
        args.run(args, console)                
        return 0
    except Exception as err:
        console.print_exception()
        return getattr(err, "code", 1)