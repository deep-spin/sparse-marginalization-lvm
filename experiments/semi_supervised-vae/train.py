import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import torch
from torch.nn import CrossEntropyLoss

from lvmhelpers.marg import \
    ExplicitWrapper, Marginalizer
from lvmhelpers.sfe import \
    ReinforceWrapper, ReinforceDeterministicWrapper, ScoreFunctionEstimator
from lvmhelpers.gumbel import \
    GumbelSoftmaxWrapper, Gumbel
from lvmhelpers.utils import DeterministicWrapper

from opts import _populate_cl_params
from data import get_mnist_dataset_semisupervised, CycleConcatDataset
from archs import MLPEncoder, MLPDecoder, Classifier, MNISTVAE


class SSVAE(pl.LightningModule):
    def __init__(
            self,
            latent_dim,
            slen,
            n_classes,
            mode,
            gs_tau,
            normalizer,
            lr,
            weight_decay,
            batch_size,
            labeled_only):
        super(SSVAE, self).__init__()

        self.save_hyperparameters()

        inference_net = MLPEncoder(
            latent_dim=self.hparams.latent_dim,
            slen=self.hparams.slen,
            n_classes=self.hparams.n_classes)

        generative_net = MLPDecoder(
            latent_dim=self.hparams.latent_dim,
            slen=self.hparams.slen,
            n_classes=self.hparams.n_classes)

        gaussian_vae = MNISTVAE(
            inference_net,
            generative_net)

        classifier_net = Classifier(
            slen=self.hparams.slen,
            n_classes=self.hparams.n_classes)

        loss_fun = get_unsupervised_loss

        if self.hparams.mode == 'rf':
            classifier_net = ReinforceWrapper(classifier_net)
            gaussian_vae = ReinforceDeterministicWrapper(gaussian_vae)
            lvm_method = ScoreFunctionEstimator
        elif self.hparams.mode == 'gs':
            classifier_net = GumbelSoftmaxWrapper(
                classifier_net,
                temperature=self.hparams.gs_tau,
                trainable_temperature=False,
                straight_through=False)
            gaussian_vae = DeterministicWrapper(gaussian_vae)
            lvm_method = Gumbel
        elif self.hparams.mode == 'marg':
            classifier_net = ExplicitWrapper(classifier_net, normalizer=normalizer)
            gaussian_vae = DeterministicWrapper(gaussian_vae)
            lvm_method = Marginalizer
        else:
            raise RuntimeError(f"Unknown training mode: {self.hparams.mode}")
        self.lvm_method = lvm_method(
            classifier_net,
            gaussian_vae,
            loss_fun,
            encoder_entropy_coeff=1.0)

    def forward(self, classifier_input, vae_input, labels):
        return self.lvm_method(classifier_input, vae_input, labels)

    def training_step(self, batch, batch_nb):
        if not self.hparams.labeled_only:
            labeled_batch, unlabeled_batch = batch
            labeled_batch_image = labeled_batch['image']
            labeled_batch_labels = labeled_batch['label']
            unlabeled_batch_image = unlabeled_batch['image']
            unlabeled_batch_labels = unlabeled_batch['label']
        else:
            labeled_batch_image = batch['image']
            labeled_batch_labels = batch['label']

        vae = self.lvm_method.decoder
        if hasattr(vae, 'agent'):
            vae = vae.agent
        classifier = self.lvm_method.encoder

        supervised_loss = get_supervised_loss(
            vae, classifier, labeled_batch_image, labeled_batch_labels)

        if not self.hparams.labeled_only:
            unsupervised_output = self(
                unlabeled_batch_image,
                unlabeled_batch_image,
                unlabeled_batch_labels)
            unsupervised_loss = unsupervised_output['loss']

            loss = \
                supervised_loss + \
                unsupervised_loss * (self.num_unlabeled / self.num_labeled)
        else:
            loss = supervised_loss

        result = pl.TrainResult(minimize=loss)
        if not self.hparams.labeled_only:
            result.log('train_elbo', unsupervised_output['log']['loss'], prog_bar=True)
            result.log('train_acc', unsupervised_output['log']['acc'], prog_bar=True)
        return result

    def validation_step(self, batch, batch_nb):
        image = batch['image']
        true_labels = batch['label']
        validation_result = self(image, image, true_labels)
        result = pl.EvalResult(checkpoint_on=validation_result['log']['loss'])
        result.log('val_elbo', validation_result['log']['loss'], prog_bar=True)
        result.log('val_acc', validation_result['log']['acc'], prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        image = batch['image']
        true_labels = batch['label']
        test_result = self(image, image, true_labels)
        result = pl.EvalResult()
        result.log('test_elbo', test_result['log']['loss'])
        result.log('test_acc', test_result['log']['acc'])
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        train_labeled, train_unlabeled, _ = get_mnist_dataset_semisupervised(
            data_dir='data/semi_supervised-vae/mnist_data/',
            train_test_split_folder='data/semi_supervised-vae/test_train_splits/',
            n_labeled=5000,
            one_of_each=False)

        self.num_labeled = len(train_labeled)
        self.num_unlabeled = len(train_unlabeled)

        if not self.hparams.labeled_only:
            concat_dataset = CycleConcatDataset(
                train_labeled,
                train_unlabeled
            )
            loader = torch.utils.data.DataLoader(
                concat_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.num_labeled = len(train_labeled)
            self.num_unlabeled = len(train_labeled)
            loader = torch.utils.data.DataLoader(
                train_labeled,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

        return loader

    def val_dataloader(self):
        _, _, valid_set = get_mnist_dataset_semisupervised(
            data_dir='data/semi_supervised-vae/mnist_data/',
            train_test_split_folder='data/semi_supervised-vae/test_train_splits/',
            eval_test_set=False)
        return torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    def test_dataloader(self):
        _, _, test_set = get_mnist_dataset_semisupervised(
            data_dir='data/semi_supervised-vae/mnist_data/',
            train_test_split_folder='data/semi_supervised-vae/test_train_splits/',
            eval_test_set=True)
        return torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)


def get_reconstruction_loss(x_reconstructed, x):
    batch_size = x.shape[0]

    bce_loss = -x * torch.log(x_reconstructed + 1e-8) - \
        (1 - x) * torch.log(1 - x_reconstructed + 1e-8)

    return bce_loss.view(batch_size, -1).sum(dim=1)


def get_kl_divergence_loss(mean, logvar):
    batch_size = mean.shape[0]
    return ((
        mean**2 + logvar.exp() - 1 - logvar
        ) / 2).view(batch_size, -1).sum(dim=1)


def get_elbo_loss(image, vae_output):
    latent_means = vae_output['latent_means']
    latent_std = vae_output['latent_std']
    image_mean = vae_output['image_mean']
    reconstruction_loss = get_reconstruction_loss(
        image_mean, image)
    kl_divergence_loss = get_kl_divergence_loss(
        latent_means, 2 * torch.log(latent_std))
    return reconstruction_loss + kl_divergence_loss


def get_unsupervised_loss(
        _classifier_input,
        discrete_latent_z,
        _vae_input,
        vae_output,
        true_labels):
    vae_loss = get_elbo_loss(_classifier_input, vae_output)
    # classifier accuracy (for logging)
    if len(discrete_latent_z.size()) == 2:
        discrete_latent_z = discrete_latent_z.argmax(dim=-1)
    acc = (discrete_latent_z == true_labels).float()
    return vae_loss, {'acc': acc}


def get_supervised_loss(vae, classifier, labeled_image, true_labels):
    loss = CrossEntropyLoss(reduction='none')
    # get loss on a batch of labeled images
    vae_output = vae(true_labels, labeled_image)
    labeled_loss = get_elbo_loss(labeled_image, vae_output)
    # cross entropy term
    scores = classifier.agent.forward(labeled_image)
    cross_entropy = loss(scores, true_labels)

    return (labeled_loss + cross_entropy).mean()


def get_model(opt):

    model = SSVAE(
        latent_dim=8,
        slen=28,
        n_classes=10,
        mode=opt.mode,
        gs_tau=opt.gs_tau,
        normalizer=opt.normalizer,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        batch_size=opt.batch_size,
        labeled_only=opt.labeled_only)

    if len(opt.warm_start_path) != 0:
        model = model.load_from_checkpoint(
            opt.warm_start_path,
            latent_dim=8,
            slen=28,
            n_classes=10,
            mode=opt.mode,
            gs_tau=opt.gs_tau,
            normalizer=opt.normalizer,
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            batch_size=opt.batch_size,
            labeled_only=opt.labeled_only)

    return model


def main(params):

    parser = argparse.ArgumentParser()
    arg_parser = _populate_cl_params(parser)
    opts = arg_parser.parse_args(params)

    # fix seed
    pl.seed_everything(opts.random_seed)

    model = get_model(opts)

    # if opts.mode == 'gs':
    #     callbacks = [TemperatureUpdater(
    #         agent=game.sender, decay=0.9, minimum=0.1)]
    # else:
    #     callbacks = []

    experiment_name = 'ssvae'
    if not opts.labeled_only:
        model_name = '%s/%s' % (experiment_name, opts.mode)
    else:
        model_name = '%s/warm_start' % (experiment_name, )
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
        max_epochs=opts.n_epochs,
        weights_save_path='checkpoints/',
        weights_summary='full',
        gpus=1 if torch.cuda.is_available() else 0,
        resume_from_checkpoint=opts.load_from_checkpoint,
        deterministic=True)
    trainer.fit(model)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
