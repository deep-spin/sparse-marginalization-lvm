import argparse


def populate_experiment_params(
        arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    arg_parser.add_argument('--root', default='data/signal-game/',
                            help='data root folder')
    # 2-agents specific parameters
    arg_parser.add_argument('--tau_s', type=float, default=10.0,
                            help='Sender Gibbs temperature')
    arg_parser.add_argument('--game_size', type=int, default=2,
                            help='Number of images seen by an agent')
    arg_parser.add_argument('--same', type=int, default=0,
                            help='Use same concepts')
    arg_parser.add_argument('--embedding_size', type=int, default=50,
                            help='embedding size')
    arg_parser.add_argument('--hidden_size', type=int, default=20,
                            help='hidden size (number of filters informed sender)')
    arg_parser.add_argument('--batches_per_epoch', type=int, default=100,
                            help='Batches in a single training/validation epoch')

    arg_parser.add_argument('--loss_type', type=str, default='nll',
                            help='acc or nll')

    return arg_parser
