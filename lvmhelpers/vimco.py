import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli


class VIMCOWrapper(nn.Module):
    """
    VIMCO Wrapper for an agent. Assumes that the during the forward,
    the wrapped agent returns log-probabilities over the potential outputs.
    During training, the wrapper
    transforms them into a tuple of (sample from the multinomial,
    log-prob of the sample, entropy for the multinomial).
    Eval-time the sample is replaced with argmax.

    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = VIMCOWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4])
    >>> (log_prob < 0).all().item()
    1
    >>> (entropy > 0).all().item()
    1
    """
    def __init__(self, agent, k=5):
        super(VIMCOWrapper, self).__init__()
        self.agent = agent
        self.k = k

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample((self.k, ))
        else:
            sample = logits.argmax(dim=-1)

        return sample, logits, entropy


class VIMCO(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(VIMCO, self).__init__()
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
        batch_size, latent_size = encoder_log_prob.shape
        if self.training:
            K, _ = discrete_latent_z.shape
        else:
            discrete_latent_z = discrete_latent_z.unsqueeze(0)
            K = 1

        encoder_input_repeat = \
            encoder_input.repeat(
                K, *torch.ones(len(encoder_input.shape), dtype=torch.long).tolist()
                ).view(-1, *encoder_input.shape[1:])
        decoder_input_repeat = \
            decoder_input.repeat(
                K, *torch.ones(len(decoder_input.shape), dtype=torch.long).tolist()
                ).view(-1, *decoder_input.shape[1:])
        labels_repeat = \
            labels.repeat(
                K, *torch.ones(len(labels.shape), dtype=torch.long).tolist()
                ).view(-1, *labels.shape[1:])

        decoder_output = self.decoder(
            discrete_latent_z.reshape(-1), decoder_input_repeat)

        loss, logs = self.loss(
            encoder_input_repeat,
            discrete_latent_z.reshape(-1),
            decoder_input_repeat,
            decoder_output,
            labels_repeat)

        encoder_categorical_distr = Categorical(logits=encoder_log_prob)
        encoder_sample_log_probs = \
            encoder_categorical_distr.log_prob(discrete_latent_z)

        logp = - loss + torch.log(1 / torch.tensor(latent_size, dtype=torch.float))

        # Log ratios: log r(x,z)
        # [B, K]
        log_p = logp.view(batch_size, -1)
        log_q = encoder_sample_log_probs.transpose(0, 1)
        log_r = log_p - log_q

        # Log importance weights: log w
        # (all computed in log space)
        # [B, K]
        log_w = log_r - log_r.logsumexp(-1).unsqueeze(-1)

        # Importance weights: w
        w = log_w.exp()

        # Generative gradient surrogate ​ # The learning signal (L)
        # is just the importance sampling estimate of log likelihood.
        # Its gradient with respect to the generative parameters is all
        # we need to update the generative net. Note I detach log_q to
        # make sure I have a surrogate for the generative gradient only

        # [B]
        L = (log_p - log_q.detach()).logsumexp(-1) - np.log(K)
        gen_grad_surrogate = L

        # Proposal gradient surrogate

        # part 2 (the entropy part)
        # [B]
        inf_grad_surrogate_entropy = (- w.detach() * log_q).sum(-1)

        # part 1 (the SFE-looking part)
        # I will assume the original paper used c = 0
        # []
        c = 0

        # Average log ratio (keeping the kth term out)
        # [B, K]
        log_a = (log_r.sum(-1).unsqueeze(-1) - log_r) / (K - 1)

        # Here we make b, which is a sample specific baseline, thus
        # with shape [B, K]

        # Smart trick: make log_r [B,K,1], then make log_a [B,K,K] by
        # placing log_a - log_r in the diagonal, then sum the two
        # things, we end up with [B,K,K] where in the diagonal we have
        # log_a (the k-exclusive combination)
        # [B, K, K]
        # note how log_r and -log_r cancel out in the diagonal :)
        b = log_r.unsqueeze(-1) + torch.diag_embed(log_a - log_r)
        # [B, K]
        b = b.logsumexp(-1) - np.log(K)

        # [B, K]
        centred_L = (L.unsqueeze(-1) - b - c)

        inf_grad_surrogate_sfe = centred_L.detach() * log_q
        # [B]
        inf_grad_surrogate_sfe = inf_grad_surrogate_sfe.sum(-1)

        # sfe surrogate
        # Switch to minimisation mode
        # []
        full_loss = - (
            gen_grad_surrogate +
            inf_grad_surrogate_entropy * self.encoder_entropy_coeff +
            inf_grad_surrogate_sfe).mean(dim=0)

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['distr'] = encoder_categorical_distr

        return {'loss': full_loss, 'log': logs}


class BitVectorVIMCOWrapper(nn.Module):
    """
    VIMCO Wrapper for an agent. Assumes that the during the forward,
    the wrapped agent returns log-probabilities over the potential outputs.
    During training, the wrapper
    transforms them into a tuple of (sample from the multinomial,
    log-prob of the sample, entropy for the multinomial).
    Eval-time the sample is replaced with argmax.

    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = BitVectorVIMCOWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4])
    >>> (log_prob < 0).all().item()
    1
    >>> (entropy > 0).all().item()
    1
    """
    def __init__(self, agent, k=5):
        super(BitVectorVIMCOWrapper, self).__init__()
        self.agent = agent
        self.k = k

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        distr = Bernoulli(logits=logits)
        entropy = distr.entropy().sum(dim=1)

        if self.training:
            sample = distr.sample((self.k, ))
        else:
            sample = (logits > 0).to(torch.float)

        return sample, logits, entropy


class BitVectorVIMCO(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(BitVectorVIMCO, self).__init__()
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
        if self.training:
            K, batch_size, latent_size = discrete_latent_z.shape
        else:
            batch_size, latent_size = discrete_latent_z.shape
            discrete_latent_z = discrete_latent_z.unsqueeze(0)
            K = 1

        encoder_input_repeat = \
            encoder_input.repeat(
                K, *torch.ones(len(encoder_input.shape), dtype=torch.long).tolist()
                ).view(-1, *encoder_input.shape[1:])
        decoder_input_repeat = \
            decoder_input.repeat(
                K, *torch.ones(len(decoder_input.shape), dtype=torch.long).tolist()
                ).view(-1, *decoder_input.shape[1:])
        labels_repeat = \
            labels.repeat(
                K, *torch.ones(len(labels.shape), dtype=torch.long).tolist()
                ).view(-1, *labels.shape[1:])

        decoder_output = self.decoder(
            discrete_latent_z.reshape(-1, latent_size), decoder_input_repeat)

        loss, logs = self.loss(
            encoder_input_repeat,
            discrete_latent_z.reshape(-1, latent_size),
            decoder_input_repeat,
            decoder_output,
            labels_repeat)

        encoder_bernoull_distr = Bernoulli(logits=encoder_log_prob)
        encoder_sample_log_probs = \
            encoder_bernoull_distr.log_prob(discrete_latent_z).sum(dim=-1)

        logp = - loss + latent_size * torch.log(torch.tensor(0.5))

        # Log ratios: log r(x,z)
        # [B, K]
        log_p = logp.view(batch_size, -1)
        log_q = encoder_sample_log_probs.transpose(0, 1)
        log_r = log_p - log_q

        # Log importance weights: log w
        # (all computed in log space)
        # [B, K]
        log_w = log_r - log_r.logsumexp(-1).unsqueeze(-1)

        # Importance weights: w
        w = log_w.exp()

        # Generative gradient surrogate ​ # The learning signal (L)
        # is just the importance sampling estimate of log likelihood.
        # Its gradient with respect to the generative parameters is all
        # we need to update the generative net. Note I detach log_q to
        # make sure I have a surrogate for the generative gradient only

        # [B]
        L = (log_p - log_q.detach()).logsumexp(-1) - np.log(K)
        gen_grad_surrogate = L

        # Proposal gradient surrogate

        # part 2 (the entropy part)
        # [B]
        inf_grad_surrogate_entropy = (- w.detach() * log_q).sum(-1)

        # part 1 (the SFE-looking part)
        # I will assume the original paper used c = 0
        # []
        c = 0

        # Average log ratio (keeping the kth term out)
        # [B, K]
        log_a = (log_r.sum(-1).unsqueeze(-1) - log_r) / (K - 1)

        # Here we make b, which is a sample specific baseline, thus
        # with shape [B, K]

        # Smart trick: make log_r [B,K,1], then make log_a [B,K,K] by
        # placing log_a - log_r in the diagonal, then sum the two
        # things, we end up with [B,K,K] where in the diagonal we have
        # log_a (the k-exclusive combination)
        # [B, K, K]
        # note how log_r and -log_r cancel out in the diagonal :)
        b = log_r.unsqueeze(-1) + torch.diag_embed(log_a - log_r)
        # [B, K]
        b = b.logsumexp(-1) - np.log(K)

        # [B, K]
        centred_L = (L.unsqueeze(-1) - b - c)

        inf_grad_surrogate_sfe = centred_L.detach() * log_q
        # [B]
        inf_grad_surrogate_sfe = inf_grad_surrogate_sfe.sum(-1)

        # sfe surrogate
        # Switch to minimisation mode
        # []
        full_loss = - (
            gen_grad_surrogate +
            inf_grad_surrogate_entropy * self.encoder_entropy_coeff +
            inf_grad_surrogate_sfe).mean(dim=0)

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['distr'] = encoder_bernoull_distr

        return {'loss': full_loss, 'log': logs}
