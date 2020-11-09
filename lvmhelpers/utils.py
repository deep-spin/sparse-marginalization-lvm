import argparse
import torch.nn as nn


class DeterministicWrapper(nn.Module):
    """
    Simple wrapper that makes a deterministic agent.
    No sampling is run on top of the wrapped agent,
    it is passed as is.
    """
    def __init__(self, agent):
        super(DeterministicWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        out = self.agent(*args, **kwargs)
        return out


def populate_common_params(
        arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    # discrete latent variable training
    arg_parser.add_argument("--mode", type=str, default="sfe",
                            choices=[
                                "sfe", "nvil", "gs", "marg", "sumsample",
                                "topksparse", "sparsemap"],
                            help="""Method to train the discrete/structured
                            latent variable model""")
    arg_parser.add_argument("--entropy_coeff", type=float, default=1.0,
                            help="""Entropy loss term coefficient (regularization)""")
    arg_parser.add_argument("--latent_size", type=int, default=10,
                            help="Number of categories (default: 10)")

    # Marginalization
    arg_parser.add_argument("--normalizer", type=str, default="softmax",
                            choices=["softmax", "sparsemax"],
                            help="""Normalizer to use when parameterizing
                            a discrete distribution over categories""")

    # Structured Marginalization
    arg_parser.add_argument("--topksparse", type=int, default=10,
                            help="""k in top-k sparsemax. Not to be confused with the --topk option
                            of the sum&sample estimator""")
    arg_parser.add_argument("--noinit", action="store_true")
    arg_parser.add_argument("--budget", type=int, default=0)

    # Gumbel-Softmax
    arg_parser.add_argument("--gs_tau", type=float, default=1.0,
                            help="GS temperature")
    arg_parser.add_argument("--straight_through", action="store_true")
    arg_parser.add_argument("--temperature_decay", type=float, default=1e-5,
                            help="""temperature decay constant for
                            Gumbel-Softmax (default: 1e-5)""")
    arg_parser.add_argument("--temperature_update_freq", type=int, default=1000,
                            help="""temperature decay frequency for
                            Gumbel-Softmax, in steps (default: 1000)""")

    # SFE
    arg_parser.add_argument("--baseline_type", type=str, default="runavg",
                            choices=["runavg", "sample"],
                            help="""baseline to use in SFE. runavg is the running average
                             and sample is a self-critic baseline""")

    # sum and sample
    arg_parser.add_argument("--topk", type=int, default=1,
                            help="""number of classes summed over
                            for sum&sample gradient estimator""")

    # random seed
    arg_parser.add_argument("--random_seed", type=int, default=42,
                            help="Set random seed")

    # trainer params
    arg_parser.add_argument("--n_epochs", type=int, default=10,
                            help="Number of epochs to train (default: 10)")
    arg_parser.add_argument("--load_from_checkpoint", type=str, default=None,
                            help="""If the parameter is set, model,
                            trainer, and optimizer states are loaded from the
                            checkpoint (default: None)""")

    # cuda setup
    arg_parser.add_argument("--no_cuda", default=False, help="disable cuda",
                            action="store_true")

    # dataset
    arg_parser.add_argument("--batch_size", type=int, default=32,
                            help="Input batch size for training (default: 32)")

    # optimizer
    arg_parser.add_argument("--optimizer", type=str, default="adam",
                            choices=["adam", "sgd", "adagrad"],
                            help="Optimizer to use [adam, sgd, adagrad] (default: adam)")
    arg_parser.add_argument("--lr", type=float, default=1e-3,
                            help="Learning rate (default: 1e-3)")
    arg_parser.add_argument("--weight_decay", type=float, default=1e-5,
                            help="L2 regularization constant (default: 1e-5)")

    return arg_parser
