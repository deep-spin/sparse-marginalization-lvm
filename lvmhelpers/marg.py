import torch
import torch.nn as nn
from entmax import entmax15, sparsemax


def entropy(p: torch.Tensor):
    """Numerically stable computation of Shannon's entropy
    for probability distributions with zero-valued elements.

    Arguments:
        p {torch.Tensor} -- tensor of probabilities.
            Size: [batch_size, n_categories]

    Returns:
        {torch.Tensor} -- the entropy of p.
            Size: [batch_size]
    """
    nz = (p > 0).to(p.device)

    eps = torch.finfo(p.dtype).eps
    p_stable = p.clone().clamp(min=eps, max=1 - eps)

    out = torch.where(
        nz,
        p_stable * torch.log(p_stable),
        torch.tensor(0., device=p.device, dtype=torch.float))

    return -(out).sum(-1)


class ExplicitWrapper(nn.Module):
    """
    Explicit Marginalization Wrapper for a network.
    Assumes that the during the forward pass,
    the network returns scores over the potential output categories.
    The wrapper transforms them into a tuple of (sample from the Categorical,
    log-prob of the sample, entropy for the Categorical).
    """
    def __init__(self, agent, normalizer='entmax'):
        super(ExplicitWrapper, self).__init__()
        self.agent = agent

        normalizer_dict = {
            'softmax': torch.softmax,
            'sparsemax': sparsemax,
            'entmax': entmax15}
        self.normalizer = normalizer_dict[normalizer]

    def forward(self, *args, **kwargs):
        scores = self.agent(*args, **kwargs)
        distr = self.normalizer(scores, dim=-1)
        entropy_distr = entropy(distr)
        sample = scores.argmax(dim=-1)
        return sample, distr, entropy_distr


class Marginalizer(torch.nn.Module):
    """
    The training loop for the marginalization method to train discrete latent variables.
    Encoder needs to be ExplicitWrapper.
    Decoder needs to be utils.DeterministicWrapper.
    """
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
        discrete_latent_z, encoder_probs, encoder_entropy = self.encoder(encoder_input)
        batch_size, latent_size = encoder_probs.shape

        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)

        losses = torch.zeros_like(encoder_probs)
        logs_global = None

        for possible_discrete_latent_z in range(latent_size):
            if encoder_probs[:, possible_discrete_latent_z].sum().detach() != 0:
                # if it's zero, all batch examples
                # will be multiplied by zero anyway,
                # so skip computations
                possible_discrete_latent_z_ = \
                    possible_discrete_latent_z + \
                    torch.zeros(
                        batch_size, dtype=torch.long).to(encoder_probs.device)
                decoder_output = self.decoder(
                    possible_discrete_latent_z_, decoder_input)

                loss_sum_term, logs = self.loss(
                    encoder_input,
                    discrete_latent_z,
                    decoder_input,
                    decoder_output,
                    labels)

                losses[:, possible_discrete_latent_z] += loss_sum_term

                if not logs_global:
                    logs_global = {k: 0.0 for k in logs.keys()}
                for k, v in logs.items():
                    if hasattr(v, 'mean'):
                        # expectation of accuracy
                        logs_global[k] += (
                            encoder_probs[:, possible_discrete_latent_z] * v).mean()

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = logs_global[k]

        # encoder_probs: [batch_size, latent_size]
        # losses: [batch_size, latent_size]
        # encoder_probs.unsqueeze(1): [batch_size, 1, latent_size]
        # losses.unsqueeze(-1): [batch_size, latent_size, 1]
        # entropy_loss: []
        # full_loss: []
        loss = encoder_probs.unsqueeze(1).bmm(losses.unsqueeze(-1)).squeeze()
        full_loss = loss.mean() + entropy_loss.mean()

        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['support'] = (encoder_probs != 0).sum(-1).to(torch.float).mean()
        logs['distr'] = encoder_probs
        return {'loss': full_loss, 'log': logs}
