import argparse


def _populate_cl_params(
        arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    arg_parser.add_argument(
        '--root', default='data/signal-game/', help='data root folder')
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
    arg_parser.add_argument('--entropy_coeff', type=float, default=0.1,
                            help="""Sender and Receiver entropy
                        loss term coefficient (regularization)""")
    arg_parser.add_argument('--mode', type=str, default='sfe',
                            help="""Training mode: Gumbel-Softmax (gs) or
                        SFE (sfe). Default: sfe.""")
    arg_parser.add_argument('--normalizer', type=str, default='entmax',
                            help='softmax, sparsemax or entmax15')
    arg_parser.add_argument('--loss', type=str, default='nll',
                            help='acc or nll')
    arg_parser.add_argument('--gs_tau', type=float, default=1.0,
                            help='GS temperature')
    arg_parser.add_argument("--straight_through", action="store_true")
    arg_parser.add_argument("--labeled_only", action="store_true")
    arg_parser.add_argument(
        '--warm_start_path', type=str, default='',
        help='Path for warm start')
    arg_parser.add_argument(
        '--temperature_decay', type=float, default=1e-5,
        help='temperature decay constant for Gumbel-Softmax (default: 1e-5)')
    arg_parser.add_argument(
        '--temperature_update_freq', type=int, default=1000,
        help='temperature decay frequency for Gumbel-Softmax, in steps (default: 1000)')
    arg_parser.add_argument('--baseline_type', type=str, default='runavg',
                            help='runavg or sample')

    arg_parser.add_argument(
        '--random_seed', type=int, default=42,
        help='Set random seed')
    # trainer params
    arg_parser.add_argument(
        '--checkpoint_dir', type=str, default=None,
        help='Where the checkpoints are stored')
    arg_parser.add_argument(
        '--preemptable', default=False,
        action='store_true',
        help="""If the flag is set,
        Trainer would always try to initialise itself from a checkpoint""")

    arg_parser.add_argument(
        '--checkpoint_freq', type=int, default=0,
        help='How often the checkpoints are saved')
    arg_parser.add_argument(
        '--validation_freq', type=int, default=1,
        help='The validation would be run every `validation_freq` epochs')
    arg_parser.add_argument(
        '--n_epochs', type=int, default=10,
        help='Number of epochs to train (default: 10)')
    arg_parser.add_argument(
        '--load_from_checkpoint', type=str, default=None,
        help="""If the parameter is set, model,
                trainer, and optimizer states are loaded from the
                'checkpoint (default: None)""")
    # cuda setup
    arg_parser.add_argument(
        '--no_cuda', default=False, help='disable cuda',
        action='store_true')
    # dataset
    arg_parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Input batch size for training (default: 32)')

    # optimizer
    arg_parser.add_argument(
        '--optimizer', type=str, default='adam',
        help='Optimizer to use [adam, sgd, adagrad] (default: adam)')
    arg_parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate (default: 1e-3)')
    arg_parser.add_argument(
        '--weight_decay', type=float, default=1e-5,
        help='L2 regularization constant (default: 1e-5)')

    # Channel parameters
    arg_parser.add_argument(
        '--vocab_size', type=int, default=10,
        help='Number of symbols (terms) in the vocabulary (default: 10)')
    arg_parser.add_argument(
        '--max_len', type=int, default=1,
        help='Max length of the sequence (default: 1)')

    # Setting up tensorboard
    arg_parser.add_argument(
        '--tensorboard', default=False, help='enable tensorboard',
        action='store_true')
    arg_parser.add_argument(
        '--tensorboard_dir', type=str, default='runs/',
        help='Path for tensorboard log')

    return arg_parser
