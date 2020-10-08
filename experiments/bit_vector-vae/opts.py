import argparse


def populate_experiment_params(
        arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    arg_parser.add_argument("--noinit", action="store_true")
    arg_parser.add_argument("--budget", type=int, default=0)

    return arg_parser
