# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributions import Categorical


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
    def __init__(self, agent):
        super(ReinforceWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = logits.argmax(dim=1)
        log_prob = distr.log_prob(sample)

        return sample, log_prob, entropy


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
            self.mean_baseline += (loss.detach().mean() -
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
