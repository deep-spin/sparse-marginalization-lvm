# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli


class ReinforceWrapper(nn.Module):
    """
    Reinforce Wrapper for an agent. Assumes that the during the forward,
    the wrapped agent returns log-probabilities over the potential outputs.
    During training, the wrapper
    transforms them into a tuple of (sample from the multinomial,
    log-prob of the sample, entropy for the multinomial).
    Eval-time the sample is replaced with argmax.

    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = ReinforceWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4])
    >>> (log_prob < 0).all().item()
    1
    >>> (entropy > 0).all().item()
    1
    """
    def __init__(self, agent, baseline_type):
        super(ReinforceWrapper, self).__init__()
        self.agent = agent
        self.baseline_type = baseline_type

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = logits.argmax(dim=-1)

        return sample, logits, entropy


class BitVectorReinforceWrapper(nn.Module):
    """
    Reinforce Wrapper for an agent. Assumes that the during the forward,
    the wrapped agent returns log-probabilities over the potential outputs.
    During training, the wrapper
    transforms them into a tuple of (sample from the multinomial,
    log-prob of the sample, entropy for the multinomial).
    Eval-time the sample is replaced with argmax.

    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = BitVectorReinforceWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4])
    >>> (log_prob < 0).all().item()
    1
    >>> (entropy > 0).all().item()
    1
    """
    def __init__(self, agent, baseline_type):
        super(BitVectorReinforceWrapper, self).__init__()
        self.agent = agent
        self.baseline_type = baseline_type

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        distr = Bernoulli(logits=logits)
        entropy = distr.entropy().sum(dim=1)
        sample = distr.sample()

        return sample, logits, entropy


class ReinforceDeterministicWrapper(nn.Module):
    """
    Simple wrapper that makes a deterministic agent (without sampling)
    compatible with Reinforce-based game, by
    adding zero log-probability and entropy values to the output.
    No sampling is run on top of the wrapped agent,
    it is passed as is.
    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = ReinforceDeterministicWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4, 3])
    >>> (log_prob == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    """
    def __init__(self, agent):
        super(ReinforceDeterministicWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        out = self.agent(*args, **kwargs)
        device = next(self.parameters()).device
        return out, torch.zeros(1).to(device), torch.zeros(1).to(device)


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
        discrete_latent_z, encoder_log_prob, encoder_entropy = \
            self.encoder(encoder_input)
        decoder_output, decoder_log_prob, decoder_entropy = \
            self.decoder(discrete_latent_z, decoder_input)

        loss, logs = self.loss(
            encoder_input,
            discrete_latent_z,
            decoder_input,
            decoder_output,
            labels)

        encoder_categorical_helper = Categorical(logits=encoder_log_prob)
        encoder_sample_log_probs = encoder_categorical_helper.log_prob(discrete_latent_z)
        if len(decoder_log_prob.size()) != 1:
            decoder_categorical_helper = Categorical(logits=decoder_log_prob)
            decoder_sample_log_probs = decoder_categorical_helper.log_prob(decoder_output)
        else:
            decoder_sample_log_probs = decoder_log_prob

        if self.encoder.baseline_type == 'runavg':
            baseline = self.mean_baseline
        elif self.encoder.baseline_type == 'sample':
            alt_z_sample = encoder_categorical_helper.sample().detach()
            decoder_output, _, _ = self.decoder(alt_z_sample, decoder_input)
            baseline, _ = self.loss(
                encoder_input,
                alt_z_sample,
                decoder_input,
                decoder_output,
                labels)

        policy_loss = (
            (loss.detach() - baseline) *
            (encoder_sample_log_probs + decoder_sample_log_probs)
            ).mean()
        entropy_loss = -(
            encoder_entropy.mean() *
            self.encoder_entropy_coeff +
            decoder_entropy.mean() *
            self.decoder_entropy_coeff)

        if self.training and self.encoder.baseline_type == 'runavg':
            self.n_points += 1.0
            self.mean_baseline += (
                loss.detach().mean() - self.mean_baseline) / self.n_points

        full_loss = policy_loss + entropy_loss + loss.mean()

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['baseline'] = self.mean_baseline
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['decoder_entropy'] = decoder_entropy.mean()

        return {'loss': full_loss, 'log': logs}


class BitVectorScoreFunctionEstimator(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(BitVectorScoreFunctionEstimator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff
        self.mean_baseline = 0.0
        self.n_points = 0.0

    def forward(self, encoder_input, decoder_input, labels):
        discrete_latent_z, encoder_log_prob, encoder_entropy = \
            self.encoder(encoder_input)
        decoder_output, decoder_log_prob, decoder_entropy = \
            self.decoder(discrete_latent_z, decoder_input)

        loss, logs = self.loss(
            encoder_input,
            discrete_latent_z,
            decoder_input,
            decoder_output,
            labels)

        encoder_categorical_helper = Bernoulli(logits=encoder_log_prob)
        encoder_sample_log_probs = \
            encoder_categorical_helper.log_prob(discrete_latent_z).sum(dim=1)

        if self.encoder.baseline_type == 'runavg':
            baseline = self.mean_baseline
        elif self.encoder.baseline_type == 'sample':
            alt_z_sample = encoder_categorical_helper.sample().detach()
            decoder_output, _, _ = self.decoder(alt_z_sample, decoder_input)
            baseline, _ = self.loss(
                encoder_input,
                alt_z_sample,
                decoder_input,
                decoder_output,
                labels)

        policy_loss = (loss.detach() - baseline) * encoder_sample_log_probs
        entropy_loss = - encoder_entropy * self.encoder_entropy_coeff

        full_loss = (policy_loss + entropy_loss + loss).mean()

        if self.training and self.encoder.baseline_type == 'runavg':
            self.n_points += 1.0
            self.mean_baseline += (
                loss.detach().mean() - self.mean_baseline) / self.n_points

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['baseline'] = self.mean_baseline
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['decoder_entropy'] = decoder_entropy.mean()

        return {'loss': full_loss, 'log': logs}
