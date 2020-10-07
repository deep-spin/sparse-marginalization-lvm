import argparse


def populate_experiment_params(
        arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    arg_parser.add_argument("--labeled_only", action="store_true")
    arg_parser.add_argument('--warm_start_path', type=str, default='',
                            help='Path for warm start')

    return arg_parser
