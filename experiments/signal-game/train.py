import os
import argparse

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from lvmwrappers.explicit_wrappers import ExplicitWrapper
from lvmwrappers.reinforce_wrappers import \
    ReinforceWrapper, ReinforceDeterministicWrapper
from lvmwrappers.gumbel_wrappers import GumbelSoftmaxWrapper
# from lvmwrappers.callbacks import TemperatureUpdater

from data import ImageNetFeat, ImagenetLoader
from archs import Sender, Receiver
from opts import _populate_cl_params


class SignalGame(pl.LightningModule):
    def __init__(
            self, sender, receiver, loss, lvm_method, opts,
            sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0):
        super(SignalGame, self).__init__()
        self.opts = opts

        self.lvm_method = lvm_method(
            sender,
            receiver,
            loss,
            encoder_entropy_coeff=sender_entropy_coeff,
            decoder_entropy_coeff=receiver_entropy_coeff)

    def forward(self, sender_input, receiver_input, labels):
        return self.lvm_method(sender_input, receiver_input, labels)

    def _step(self, batch):

        sender_input, receiver_input, labels = batch
        return self(sender_input, receiver_input, labels)

    def training_step(self, batch, batch_nb):

        return self._step(batch)

    def validation_step(self, batch, batch_nb):

        return self._step(batch)

    def test_step(self, batch, batch_nb):

        return self._step(batch)

    def _epoch_end(self, outputs):

        acc_mean = 0
        baseline_mean = 0
        loss_mean = 0
        sender_entropy_mean = 0
        receiver_entropy_mean = 0
        for output in outputs:
            acc_mean += output['log']['acc']
            baseline_mean += output['log']['baseline']
            loss_mean += output['log']['loss']
            sender_entropy_mean += output['log']['encoder_entropy']
            receiver_entropy_mean += output['log']['decoder_entropy']

        acc_mean /= len(outputs)
        baseline_mean /= len(outputs)
        loss_mean /= len(outputs)
        sender_entropy_mean /= len(outputs)
        receiver_entropy_mean /= len(outputs)

        return (
            acc_mean.item(),
            baseline_mean,
            loss_mean.item(),
            sender_entropy_mean.item(),
            receiver_entropy_mean.item()
            )

    def training_epoch_end(self, outputs):

        (train_acc_mean,
         train_baseline_mean,
         train_loss_mean,
         train_sender_entropy_mean,
         train_receiver_entropy_mean
         ) = self._epoch_end(outputs)

        tqdm_dict = {'train_acc': train_acc_mean}

        results = {
            'progress_bar': tqdm_dict,
            'log': {
                'train_acc': train_acc_mean,
                'train_baseline': train_baseline_mean,
                'train_loss': train_loss_mean,
                'train_sender_entropy': train_sender_entropy_mean,
                'train_receiver_entropy': train_receiver_entropy_mean
                }
            }

        return results

    def validation_epoch_end(self, outputs):

        (val_acc_mean,
         val_baseline_mean,
         val_loss_mean,
         val_sender_entropy_mean,
         val_receiver_entropy_mean
         ) = self._epoch_end(outputs)

        tqdm_dict = {'val_acc': val_acc_mean}

        results = {
            'progress_bar': tqdm_dict,
            'log': {
                'val_acc': val_acc_mean,
                'val_baseline': val_baseline_mean,
                'val_loss': val_loss_mean,
                'val_sender_entropy': val_sender_entropy_mean,
                'val_receiver_entropy': val_receiver_entropy_mean
                }
            }

        return results

    def test_epoch_end(self, outputs):

        (test_acc_mean,
         test_baseline_mean,
         test_loss_mean,
         test_sender_entropy_mean,
         test_receiver_entropy_mean
         ) = self._epoch_end(outputs)

        tqdm_dict = {'test_acc': test_acc_mean}

        results = {
            'progress_bar': tqdm_dict,
            'log': {
                'test_acc': test_acc_mean,
                'test_baseline': test_baseline_mean,
                'test_loss': test_loss_mean,
                'test_sender_entropy': test_sender_entropy_mean,
                'test_receiver_entropy': test_receiver_entropy_mean
                }
            }

        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.opts.lr)

    def train_dataloader(self):
        data_folder = os.path.join(self.opts.root, "train/")
        dataset = ImageNetFeat(root=data_folder)
        return ImagenetLoader(
            dataset,
            num_workers=4, shuffle=True, opt=self.opts,
            seed=self.opts.random_seed)

    def val_dataloader(self):
        # TODO: use specific indexes
        # fixed seed so it's always the same 1024 (32*32) pairs
        data_folder = os.path.join(self.opts.root, "train/")
        dataset = ImageNetFeat(root=data_folder)
        return ImagenetLoader(
            dataset,
            num_workers=4, opt=self.opts,
            seed=20200724)

    def test_dataloader(self):
        # TODO: use specific indexes
        # fixed seed so it's always the same 1024 (32*32) pairs
        data_folder = os.path.join(self.opts.root, "train/")
        dataset = ImageNetFeat(root=data_folder)
        return ImagenetLoader(
            dataset,
            num_workers=4, opt=self.opts,
            seed=20200725)


class ScoreFunctionEstimator(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(ScoreFunctionEstimator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff
        self.mean_baseline = 0.0
        self.n_points = 0.0

    def forward(self, encoder_input, decoder_input, labels):
        message, encoder_log_prob, encoder_entropy = \
            self.encoder(encoder_input)
        decoder_output, decoder_log_prob, decoder_entropy = \
            self.decoder(message, decoder_input)

        loss, logs = self.loss(
            encoder_input,
            message,
            decoder_input,
            decoder_output,
            labels)
        policy_loss = (
            (loss.detach() - self.mean_baseline) *
            (encoder_log_prob + decoder_log_prob)).mean()
        entropy_loss = -(
            encoder_entropy.mean() *
            self.encoder_entropy_coeff +
            decoder_entropy.mean() *
            self.decoder_entropy_coeff)

        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points

        full_loss = policy_loss + entropy_loss + loss.mean()

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['baseline'] = self.mean_baseline
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['decoder_entropy'] = decoder_entropy.mean()

        return {'loss': full_loss, 'log': logs}


class Gumbel(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(Gumbel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff

    def forward(self, encoder_input, decoder_input, labels):
        message = self.encoder(encoder_input)
        decoder_output = self.decoder(message, decoder_input)

        loss, logs = self.loss(
            encoder_input,
            message,
            decoder_input,
            decoder_output,
            labels)

        full_loss = loss.mean()

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['baseline'] = torch.zeros(1).to(loss.device)
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = torch.zeros(1).to(loss.device)
        logs['decoder_entropy'] = torch.zeros(1).to(loss.device)

        return {'loss': full_loss, 'log': logs}


class Marginalizer(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(Marginalizer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff

    def forward(self, encoder_input, decoder_input, labels):
        message, encoder_probs, encoder_entropy = self.encoder(encoder_input)
        batch_size, vocab_size = encoder_probs.shape

        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)
        if self.training:
            losses = torch.zeros_like(encoder_probs)
            logs_global = None

            for possible_message in range(vocab_size):
                if encoder_probs[:, possible_message].sum().item() != 0:
                    # if it's zero, all batch examples
                    # will be multiplied by zero anyway,
                    # so skip computations
                    possible_message_ = \
                        possible_message + \
                        torch.zeros(
                            batch_size, dtype=torch.long).to(encoder_probs.device)
                    decoder_output = self.decoder(
                        possible_message_, decoder_input)

                    loss_sum_term, logs = self.loss(
                        encoder_input,
                        message,
                        decoder_input,
                        decoder_output,
                        labels)

                    losses[:, possible_message] += loss_sum_term

                    if not logs_global:
                        logs_global = {k: 0.0 for k in logs.keys()}
                    for k, v in logs.items():
                        if hasattr(v, 'mean'):
                            # expectation of accuracy
                            logs_global[k] += (
                                encoder_probs[:, possible_message] * v).mean()

            for k, v in logs.items():
                if hasattr(v, 'mean'):
                    logs[k] = logs_global[k]

            # encoder_probs: [batch_size, vocab_size]
            # losses: [batch_size, vocab_size]
            # encoder_probs.unsqueeze(1): [batch_size, 1, vocab_size]
            # losses.unsqueeze(-1): [batch_size, vocab_size, 1]
            # entropy_loss: []
            # full_loss: []
            loss = encoder_probs.unsqueeze(1).bmm(losses.unsqueeze(-1)).squeeze()
            full_loss = loss.mean() + entropy_loss

        else:
            decoder_output = self.decoder(message, decoder_input)
            loss, logs = self.loss(
                encoder_input,
                message,
                decoder_input,
                decoder_output,
                labels)

            full_loss = loss.mean() + entropy_loss

            for k, v in logs.items():
                if hasattr(v, 'mean'):
                    logs[k] = v.mean()

        logs['baseline'] = torch.zeros(1).to(loss.device)
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['decoder_entropy'] = torch.zeros(1).to(loss.device)
        # TODO: nonzero for every epoch end
        logs['nonzeros'] = (encoder_probs != 0).sum(-1).to(torch.float).mean()
        return {'loss': full_loss, 'log': logs}


def loss_acc(_sender_input, _message, _receiver_input, receiver_output, labels):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    # receiver outputs are samples
    acc = (labels == receiver_output).float()
    return -acc, {'acc': acc}


def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    # receiver outputs are logits
    acc = (labels == receiver_output.argmax(dim=1)).float()
    return nll, {'acc': acc}


def get_game(opt):
    feat_size = 4096

    sender = Sender(
        opt.game_size,
        feat_size,
        opt.embedding_size,
        opt.hidden_size,
        opt.vocab_size,
        temp=opt.tau_s)

    receiver = Receiver(
        opt.game_size,
        feat_size,
        opt.embedding_size,
        opt.vocab_size,
        reinforce=(opt.mode == 'rf' or opt.mode == 'marg'))

    loss = loss_nll

    if opt.mode == 'rf':
        sender = ReinforceWrapper(sender)
        receiver = ReinforceDeterministicWrapper(receiver)
        if opt.loss == 'acc':
            loss = loss_acc
            receiver = ReinforceWrapper(receiver)
        lvm_method = ScoreFunctionEstimator
    elif opt.mode == 'gs':
        sender = GumbelSoftmaxWrapper(
            sender,
            temperature=opt.gs_tau,
            trainable_temperature=False,
            straight_through=False)
        lvm_method = Gumbel
    elif opt.mode == 'marg':
        sender = ExplicitWrapper(sender, normalizer=opt.normalizer)
        lvm_method = Marginalizer

    else:
        raise RuntimeError(f"Unknown training mode: {opt.mode}")

    game = SignalGame(
        sender, receiver, loss, lvm_method, opt,
        sender_entropy_coeff=opt.entropy_coeff,
        receiver_entropy_coeff=opt.entropy_coeff)
    return game


def main(params):

    parser = argparse.ArgumentParser()
    arg_parser = _populate_cl_params(parser)
    opts = arg_parser.parse_args(params)

    signal_game = get_game(opts)

    # if opts.mode == 'gs':
    #     callbacks = [TemperatureUpdater(
    #         agent=game.sender, decay=0.9, minimum=0.1)]
    # else:
    #     callbacks = []

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(
        progress_bar_refresh_rate=20,
        logger=tb_logger,
        max_steps=opts.batches_per_epoch*opts.n_epochs,
        limit_val_batches=1024/opts.batch_size,
        limit_test_batches=1024/opts.batch_size,
        val_check_interval=opts.batches_per_epoch)
    trainer.fit(signal_game)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
