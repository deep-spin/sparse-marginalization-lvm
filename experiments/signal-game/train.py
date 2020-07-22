# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch.nn.functional as F

from lvmwrappers.explicit_wrappers import (
    ExplicitWrapper,
    ExplicitDeterministicWrapper,
    SymbolGameExplicit)
from lvmwrappers.reinforce_wrappers import (
    ReinforceWrapper,
    ReinforceDeterministicWrapper,
    SymbolGameReinforce)
from lvmwrappers.gumbel_wrappers import (
    GumbelSoftmaxWrapper,
    SymbolGameGS)
from features import ImageNetFeat, ImagenetLoader
from archs import InformedSender, Sender, Receiver


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', default='', help='data root folder')
    # 2-agents specific parameters
    parser.add_argument('--tau_s', type=float, default=10.0,
                        help='Sender Gibbs temperature')
    parser.add_argument('--game_size', type=int, default=2,
                        help='Number of images seen by an agent')
    parser.add_argument('--same', type=int, default=0,
                        help='Use same concepts')
    parser.add_argument('--embedding_size', type=int, default=50,
                        help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=20,
                        help='hidden size (number of filters informed sender)')
    parser.add_argument('--batches_per_epoch', type=int, default=100,
                        help='Batches in a single training/validation epoch')
    parser.add_argument('--inf_sen', type=int, default=0,
                        help='Use informed sender')
    parser.add_argument('--inf_rec', type=int, default=0,
                        help='Use informed receiver')
    parser.add_argument('--entropy_coeff', type=float, default=0.1,
                        help="""Sender and Receiver entropy
                        loss term coefficient (regularization)""")
    parser.add_argument('--mode', type=str, default='rf',
                        help="""Training mode: Gumbel-Softmax (gs) or
                        Reinforce (rf). Default: rf.""")
    parser.add_argument('--normalizer', type=str, default='entmax',
                        help='softmax, sparsemax or entmax15')
    parser.add_argument('--loss', type=str, default='nll',
                        help='acc or nll')
    parser.add_argument('--gs_tau', type=float, default=1.0,
                        help='GS temperature')

    return parser


def loss_acc(_sender_input, _message, _receiver_input, receiver_output, labels):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()
    return -acc, {'acc': acc}


def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float()
    return nll, {'acc': acc}


def get_game(opt):
    feat_size = 4096

    if opt.inf_sen:
        sender = InformedSender(opt.game_size, feat_size,
                                opt.embedding_size, opt.hidden_size, opt.vocab_size,
                                temp=opt.tau_s)
    else:
        sender = Sender(opt.game_size, feat_size,
                        opt.embedding_size, opt.hidden_size, opt.vocab_size,
                        temp=opt.tau_s)

    receiver = Receiver(opt.game_size, feat_size,
                        opt.embedding_size, opt.vocab_size,
                        reinforce=(opt.mode == 'rf' or opt.mode == 'explicit'))

    if opt.mode == 'rf':
        sender = ReinforceWrapper(sender)
        if opt.loss == 'acc':
            loss = loss_acc
            receiver = ReinforceWrapper(receiver)
        else:
            loss = loss_nll
            receiver = ReinforceDeterministicWrapper(receiver)
        game = SymbolGameReinforce(
            sender, receiver, loss,
            sender_entropy_coeff=opt.entropy_coeff,
            receiver_entropy_coeff=opt.entropy_coeff)
    elif opt.mode == 'gs':
        sender = GumbelSoftmaxWrapper(sender, temperature=opt.gs_tau)
        game = SymbolGameGS(sender, receiver, loss_nll)
    elif opt.mode == 'explicit':
        sender = ExplicitWrapper(sender, normalizer=opt.normalizer)
        receiver = ExplicitDeterministicWrapper(receiver)
        game = SymbolGameExplicit(
            sender, receiver, loss_nll,
            sender_entropy_coeff=opt.entropy_coeff)
    else:
        raise RuntimeError(f"Unknown training mode: {opt.mode}")

    return game


def main(params):

    parser = parse_arguments()
    opts = core.init(params=params, arg_parser=parser)

    data_folder = os.path.join(opts.root, "train/")
    dataset = ImageNetFeat(root=data_folder)

    train_loader = ImagenetLoader(dataset, batch_size=opts.batch_size, shuffle=True, opt=opts,
                                  batches_per_epoch=opts.batches_per_epoch, seed=None)
    validation_loader = ImagenetLoader(dataset, opt=opts, batch_size=opts.batch_size,
                                       batches_per_epoch=opts.batches_per_epoch,
                                       seed=7)
    game = get_game(opts)
    optimizer = core.build_optimizer(game.parameters())

    if opts.mode == 'gs':
        callbacks = [core.TemperatureUpdater(
            agent=game.sender, decay=0.9, minimum=0.1)]
    else:
        callbacks = []

    callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=validation_loader, callbacks=callbacks)

    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
