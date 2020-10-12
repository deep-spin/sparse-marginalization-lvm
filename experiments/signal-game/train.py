import os
import argparse

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from lvmhelpers.marg import \
    ExplicitWrapper, Marginalizer
from lvmhelpers.sum_and_sample import \
    SumAndSampleWrapper, SumAndSample
from lvmhelpers.sfe import \
    SFEWrapper, SFEDeterministicWrapper, ScoreFunctionEstimator
from lvmhelpers.vimco import \
    VIMCOWrapper, VIMCO
from lvmhelpers.gumbel import \
    GumbelSoftmaxWrapper, Gumbel
from lvmhelpers.utils import DeterministicWrapper, populate_common_params

from data import ImageNetFeat, ImagenetLoader
from archs import Sender, Receiver
from opts import populate_experiment_params


class SignalGame(pl.LightningModule):
    def __init__(
            self,
            feat_size,
            embedding_size,
            hidden_size,
            game_size,
            tau_s,
            loss_type,
            root,
            same,
            mode,
            entropy_coeff,
            vocab_size,
            normalizer,
            gs_tau,
            temperature_decay,
            temperature_update_freq,
            straight_through,
            baseline_type,
            vimco_k,
            topk,
            random_seed,
            batch_size,
            lr,
            weight_decay,
            optimizer):
        super(SignalGame, self).__init__()

        self.save_hyperparameters()

        sender = Sender(
            self.hparams.game_size,
            self.hparams.feat_size,
            self.hparams.embedding_size,
            self.hparams.hidden_size,
            self.hparams.vocab_size,
            temp=self.hparams.tau_s)

        receiver = Receiver(
            self.hparams.game_size,
            self.hparams.feat_size,
            self.hparams.embedding_size,
            self.hparams.vocab_size,
            sfe=(
                self.hparams.mode == 'sfe' or
                self.hparams.mode == 'marg' or
                self.hparams.mode == 'sumsample'))

        loss_fun = loss_nll

        if self.hparams.mode == 'sfe':
            sender = SFEWrapper(sender, baseline_type=self.hparams.baseline_type)
            if self.hparams.loss_type == 'acc':
                loss_fun = loss_acc
                receiver = SFEWrapper(
                    receiver, baseline_type=self.hparams.baseline_type)
            else:
                receiver = SFEDeterministicWrapper(receiver)
            lvm_method = ScoreFunctionEstimator
        elif self.hparams.mode == 'vimco':
            sender = VIMCOWrapper(sender, k=self.hparams.vimco_k)
            receiver = DeterministicWrapper(receiver)
            lvm_method = VIMCO
        elif self.hparams.mode == 'gs':
            sender = GumbelSoftmaxWrapper(
                sender,
                temperature=self.hparams.gs_tau,
                straight_through=self.hparams.straight_through)
            receiver = DeterministicWrapper(receiver)
            lvm_method = Gumbel
        elif self.hparams.mode == 'marg':
            sender = ExplicitWrapper(sender, normalizer=self.hparams.normalizer)
            receiver = DeterministicWrapper(receiver)
            lvm_method = Marginalizer
        elif self.hparams.mode == 'sumsample':
            sender = SumAndSampleWrapper(sender, topk=self.hparams.topk)
            receiver = DeterministicWrapper(receiver)
            lvm_method = SumAndSample
        else:
            raise RuntimeError(f"Unknown training mode: {self.hparams.mode}")

        self.lvm_method = lvm_method(
            sender,
            receiver,
            loss_fun,
            encoder_entropy_coeff=self.hparams.entropy_coeff,
            decoder_entropy_coeff=self.hparams.entropy_coeff)

    def forward(self, sender_input, receiver_input, labels):
        return self.lvm_method(sender_input, receiver_input, labels)

    def training_step(self, batch, batch_nb):
        sender_input, receiver_input, labels = batch
        training_result = self(sender_input, receiver_input, labels)
        loss = training_result['loss']

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', training_result['log']['loss'], prog_bar=True)
        result.log('train_acc', training_result['log']['acc'], prog_bar=True)

        if 'nonzeros' in training_result['log'].keys():
            result.log(
                'train_nonzeros',
                training_result['log']['nonzeros'],
                prog_bar=True)

        # Update temperature if Gumbel
        if self.hparams.mode == 'gs':
            self.lvm_method.encoder.update_temperature(
                self.global_step,
                self.hparams.temperature_update_freq,
                self.hparams.temperature_decay)
            result.log('temperature', self.lvm_method.encoder.temperature)

        return result

    def validation_step(self, batch, batch_nb):
        sender_input, receiver_input, labels = batch
        validation_result = self(sender_input, receiver_input, labels)
        result = pl.EvalResult(checkpoint_on=validation_result['log']['loss'])
        result.log('val_loss', validation_result['log']['loss'], prog_bar=True)
        result.log('val_acc', validation_result['log']['acc'], prog_bar=True)

        if 'nonzeros' in validation_result['log'].keys():
            result.log(
                'val_nonzeros',
                validation_result['log']['nonzeros'],
                prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        sender_input, receiver_input, labels = batch
        test_result = self(sender_input, receiver_input, labels)
        result = pl.EvalResult()
        result.log('test_loss', test_result['log']['loss'])
        result.log('test_acc', test_result['log']['acc'])

        if 'nonzeros' in test_result['log'].keys():
            result.log(
                'test_nonzeros',
                test_result['log']['nonzeros'],
                prog_bar=True)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        data_folder = os.path.join(self.hparams.root, "train/")
        dataset = ImageNetFeat(root=data_folder)
        return ImagenetLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            game_size=self.hparams.game_size,
            same=self.hparams.same,
            shuffle=True,
            seed=self.hparams.random_seed,
            num_workers=4,
            pin_memory=True)

    def val_dataloader(self):
        # TODO: use specific indexes
        # fixed seed so it's always the same 1024 (32*32) pairs
        data_folder = os.path.join(self.hparams.root, "train/")
        dataset = ImageNetFeat(root=data_folder)
        return ImagenetLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            game_size=self.hparams.game_size,
            same=self.hparams.same,
            shuffle=False,
            seed=20200724,
            num_workers=4,
            pin_memory=True)

    def test_dataloader(self):
        # TODO: use specific indexes
        # fixed seed so it's always the same 1024 (32*32) pairs
        data_folder = os.path.join(self.hparams.root, "train/")
        dataset = ImageNetFeat(root=data_folder)
        return ImagenetLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            game_size=self.hparams.game_size,
            same=self.hparams.same,
            shuffle=False,
            seed=20200725,
            num_workers=4,
            pin_memory=True)


def loss_acc(_sender_input, _message, _receiver_input, receiver_output, labels):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    # receiver outputs are samples
    acc = (labels == receiver_output).float()
    return -acc, {'acc': acc}


def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels):
    """
    NLL loss - differentiable and can be used with both GS and SFE
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    # receiver outputs are logits
    acc = (labels == receiver_output.argmax(dim=-1)).float()
    return nll, {'acc': acc}


def get_model(opt):
    game = SignalGame(
        feat_size=4096,
        embedding_size=opt.embedding_size,
        hidden_size=opt.hidden_size,
        game_size=opt.game_size,
        tau_s=opt.tau_s,
        loss_type=opt.loss_type,
        root=opt.root,
        same=opt.same,
        mode=opt.mode,
        entropy_coeff=opt.entropy_coeff,
        vocab_size=opt.latent_size,
        normalizer=opt.normalizer,
        gs_tau=opt.gs_tau,
        temperature_decay=opt.temperature_decay,
        temperature_update_freq=opt.temperature_update_freq,
        straight_through=opt.straight_through,
        baseline_type=opt.baseline_type,
        vimco_k=opt.vimco_k,
        topk=opt.topk,
        random_seed=opt.random_seed,
        batch_size=opt.batch_size,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        optimizer=opt.optimizer)

    return game


def main(params):

    arg_parser = argparse.ArgumentParser()
    arg_parser = populate_experiment_params(arg_parser)
    arg_parser = populate_common_params(arg_parser)
    opts = arg_parser.parse_args(params)

    # fix seed
    pl.seed_everything(opts.random_seed)

    signal_game = get_model(opts)

    experiment_name = 'signal-game'
    model_name = '%s/%s' % (experiment_name, opts.mode)
    other_info = [
        "lr-{}".format(opts.lr),
    ]
    model_name = '%s/%s' % (model_name, '_'.join(other_info))

    tb_logger = pl_loggers.TensorBoardLogger(
        'logs/',
        name=model_name)

    trainer = pl.Trainer(
        progress_bar_refresh_rate=20,
        logger=tb_logger,
        max_steps=opts.batches_per_epoch*opts.n_epochs,
        limit_val_batches=1024/opts.batch_size,
        limit_test_batches=1024/opts.batch_size,
        val_check_interval=opts.batches_per_epoch,
        weights_save_path='checkpoints/',
        weights_summary='full',
        gpus=1 if torch.cuda.is_available() else 0,
        resume_from_checkpoint=opts.load_from_checkpoint,
        deterministic=True)
    trainer.fit(signal_game)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
