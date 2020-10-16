# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: remove these things
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli
from torch.distributions import Categorical, Bernoulli


def gumbel_softmax_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        straight_through: bool = False):

    size = logits.size()

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


def gumbel_softmax_bit_vector_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        straight_through: bool = False):

    sample = RelaxedBernoulli(
        logits=logits, temperature=temperature).rsample()

    if straight_through:
        hard_sample = (logits > 0).to(torch.float)
        sample = sample + (hard_sample - sample).detach()

    return sample


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

        self.distr_type = Categorical

    def forward(self, *args, **kwargs):
        scores = self.agent(*args, **kwargs)
        sample = gumbel_softmax_sample(
            scores, self.temperature, self.straight_through)
        distr = self.distr_type(logits=scores)
        entropy = distr.entropy()
        return sample, scores, entropy

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
        discrete_latent_z, encoder_scores, encoder_entropy = self.encoder(encoder_input)
        decoder_output = self.decoder(discrete_latent_z, decoder_input)

        # entropy component of the final loss, we can
        # compute already but will only use it later on
        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)

        argmax = encoder_scores.argmax(dim=-1)

        loss, logs = self.loss(
            encoder_input,
            argmax,
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
        logs['distr'] = self.encoder.distr_type(logits=encoder_scores)
        return {'loss': full_loss, 'log': logs}


class BitVectorGumbelSoftmaxWrapper(GumbelSoftmaxWrapper):
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
        super(BitVectorGumbelSoftmaxWrapper, self).__init__(
            agent,
            temperature,
            trainable_temperature,
            straight_through)
        self.distr_type = Bernoulli

    def forward(self, *args, **kwargs):
        scores = self.agent(*args, **kwargs)
        sample = gumbel_softmax_bit_vector_sample(
            scores, self.temperature, self.straight_through)
        distr = self.distr_type(logits=scores)
        entropy = distr.entropy().sum(dim=-1)
        return sample, scores, entropy


class BitVectorGumbel(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(BitVectorGumbel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff

    def forward(self, encoder_input, decoder_input, labels):
        discrete_latent_z, encoder_scores, encoder_entropy = self.encoder(encoder_input)
        decoder_output = self.decoder(discrete_latent_z, decoder_input)

        # entropy component of the final loss, we can
        # compute already but will only use it later on
        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)

        argmax = (encoder_scores > 0).to(torch.float)

        loss, logs = self.loss(
            encoder_input,
            argmax,
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
        logs['distr'] = self.encoder.distr_type(logits=encoder_scores)
        return {'loss': full_loss, 'log': logs}
