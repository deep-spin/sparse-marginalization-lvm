# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: remove these things
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical
from torch.distributions import Categorical


def gumbel_softmax_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        training: bool = True,
        straight_through: bool = False):

    size = logits.size()
    if not training:
        indexes = logits.argmax(dim=-1)
        one_hot = torch.zeros_like(logits).view(-1, size[-1])
        one_hot.scatter_(1, indexes.view(-1, 1), 1)
        one_hot = one_hot.view(*size)
        return one_hot

    sample = RelaxedOneHotCategorical(
        logits=logits, temperature=temperature).rsample()

    if straight_through:
        size = sample.size()
        indexes = sample.argmax(dim=-1)
        hard_sample = torch.zeros_like(sample).view(-1, size[-1])
        hard_sample.scatter_(1, indexes.view(-1, 1), 1)
        hard_sample = hard_sample.view(*size)

        sample = sample + (hard_sample - sample).detach()
    return sample


class GumbelSoftmaxLayer(nn.Module):
    def __init__(self,
                 temperature: float = 1.0,
                 trainable_temperature: bool = False,
                 straight_through: bool = False):
        super(GumbelSoftmaxLayer, self).__init__()
        self.straight_through = straight_through

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True)

    def forward(self, logits: torch.Tensor):
        return gumbel_softmax_sample(
            logits, self.temperature, self.training, self.straight_through)


class GumbelSoftmaxWrapper(nn.Module):
    """
    Gumbel-Softmax Wrapper for an agent that outputs a single symbol.
    Assumes that during the forward pass,
    the agent returns log-probabilities over the potential output symbols.
    During training, the wrapper
    transforms them into a sample from the Gumbel Softmax (GS) distribution;
    eval-time it returns greedy one-hot encoding
    of the same shape.

    >>> inp = torch.zeros((4, 10)).uniform_()
    >>> outp = GumbelSoftmaxWrapper(nn.Linear(10, 2))(inp)
    >>> torch.allclose(outp.sum(dim=-1), torch.ones_like(outp.sum(dim=-1)))
    True
    >>> outp = GumbelSoftmaxWrapper(nn.Linear(10, 2), straight_through=True)(inp)
    >>> torch.allclose(outp.sum(dim=-1), torch.ones_like(outp.sum(dim=-1)))
    True
    >>> (max_value, _), (min_value, _) = outp.max(dim=-1), outp.min(dim=-1)
    >>> (max_value == 1.0).all().item() == 1 and (min_value == 0.0).all().item() == 1
    True
    """

    def __init__(self,
                 agent,
                 temperature=1.0,
                 trainable_temperature=False,
                 straight_through=False):
        """
        :param agent: The agent to be wrapped. agent.forward() has to output
            log-probabilities over the vocabulary
        :param temperature: The temperature of the Gumbel Softmax distribution
        :param trainable_temperature: If set to True, the temperature becomes
            a trainable parameter of the model
        :params straight_through: Whether straigh-through Gumbel Softmax is used
        """
        super(GumbelSoftmaxWrapper, self).__init__()
        self.agent = agent
        self.straight_through = straight_through
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True)

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)
        sample = gumbel_softmax_sample(
            logits, self.temperature, self.training, self.straight_through)
        distr = Categorical(logits=logits)
        entropy = distr.entropy()
        return sample, logits, entropy

    def update_temperature(
            self, current_step, temperature_update_freq, temperature_decay):
        """
        use this at the end of each training step to anneal the temperature according
        to max(0.5, exp(-rt)) with r and t being the decay rate and training step,
        respectively.
        """
        if current_step % temperature_update_freq == 0:
            rt = temperature_decay * torch.tensor(current_step)
            self.temperature = torch.max(
                torch.tensor(0.5), torch.exp(-rt))


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
        discrete_latent_z, encoder_log_prob, encoder_entropy = self.encoder(encoder_input)
        decoder_output = self.decoder(discrete_latent_z, decoder_input)

        # entropy component of the final loss, we can
        # compute already but will only use it later on
        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)

        loss, logs = self.loss(
            encoder_input,
            discrete_latent_z,
            decoder_input,
            decoder_output,
            labels)

        full_loss = loss.mean() + entropy_loss

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['baseline'] = torch.zeros(1).to(loss.device)
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = torch.zeros(1).to(loss.device)
        logs['decoder_entropy'] = torch.zeros(1).to(loss.device)

        return {'loss': full_loss, 'log': logs}
