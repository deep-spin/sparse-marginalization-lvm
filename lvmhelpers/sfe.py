import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli


class SFEWrapper(nn.Module):
    """
    SFE Wrapper for a network. Assumes that the during the forward pass,
    the network returns scores over the potential output categories.
    The wrapper transforms them into a tuple of (sample from the Categorical,
    log-prob of the sample, entropy for the Categorical).
    """
    def __init__(self, agent, baseline_type):
        """
        Arguments:
            agent -- The agent to be wrapped. agent.forward() has to output
                scores over the categories
            baseline_type {str} -- which baseline to use. Either 'runavg'
                or 'sample'.
        """
        super(SFEWrapper, self).__init__()
        self.agent = agent
        self.baseline_type = baseline_type

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- SFE sample.
                Size: [batch_size]
            scores {torch.Tensor} -- the output of the network.
                Important to compute the policy component of the SFE loss.
                Size: [batch_size, n_categories]
            entropy {torch.Tensor} -- the entropy of the Categorical distribution
                parameterized by the scores.
                Size: [batch_size]
        """
        scores = self.agent(*args, **kwargs)

        distr = Categorical(logits=scores)
        entropy = distr.entropy()

        sample = distr.sample()

        return sample, scores, entropy


class SFEDeterministicWrapper(nn.Module):
    """
    Simple wrapper that makes a deterministic agent (without sampling)
    compatible with SFE-based training, by
    adding zero log-probability and entropy values to the output.
    No sampling is run on top of the wrapped agent,
    it is passed as is.
    """
    def __init__(self, agent):
        super(SFEDeterministicWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        out = self.agent(*args, **kwargs)
        device = next(self.parameters()).device
        return out, torch.zeros(1).to(device), torch.zeros(1).to(device)


class ScoreFunctionEstimator(torch.nn.Module):
    """
    The training loop for the SFE method to train discrete latent variables.
    Encoder/Decoder needs to be either SFEWrapper or SFEDeterministicWrapper.
    """
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
        discrete_latent_z, encoder_scores, encoder_entropy = \
            self.encoder(encoder_input)
        decoder_output, decoder_scores, decoder_entropy = \
            self.decoder(discrete_latent_z, decoder_input)

        argmax = encoder_scores.argmax(dim=-1)

        loss, logs = self.loss(
            encoder_input,
            argmax,
            decoder_input,
            decoder_output,
            labels)

        encoder_categorical_helper = Categorical(logits=encoder_scores)
        encoder_sample_log_probs = encoder_categorical_helper.log_prob(discrete_latent_z)
        if len(decoder_scores.size()) != 1:
            decoder_categorical_helper = Categorical(logits=decoder_scores)
            decoder_sample_log_probs = decoder_categorical_helper.log_prob(decoder_output)
        else:
            decoder_sample_log_probs = decoder_scores

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
        logs['distr'] = encoder_categorical_helper.probs

        return {'loss': full_loss, 'log': logs}


class BitVectorSFEWrapper(nn.Module):
    """
    SFE Wrapper for a network that parameterizes
    independent Bernoulli distributions.
    Assumes that the during the forward pass,
    the network returns scores for the Bernoulli parameters.
    The wrapper transforms them into a tuple of (sample from the Bernoulli,
    log-prob of the sample, entropy for the independent Bernoulli).
    """
    def __init__(self, agent, baseline_type):
        """
        Arguments:
            agent -- The agent to be wrapped. agent.forward() has to output
                scores for each Bernoulli
            baseline_type {str} -- which baseline to use. Either 'runavg'
                or 'sample'.
        """
        super(BitVectorSFEWrapper, self).__init__()
        self.agent = agent
        self.baseline_type = baseline_type

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- SFE sample.
                Size: [batch_size, n_bits]
            scores {torch.Tensor} -- the output of the network.
                Important to compute the policy component of the SFE loss.
                Size: [batch_size, n_bits]
            entropy {torch.Tensor} -- the entropy of the independent Bernoulli
                parameterized by the scores.
                Size: [batch_size]
        """
        scores = self.agent(*args, **kwargs)

        distr = Bernoulli(logits=scores)
        entropy = distr.entropy().sum(dim=1)

        sample = distr.sample()

        return sample, scores, entropy


class BitVectorScoreFunctionEstimator(torch.nn.Module):
    """
    The training loop for the SFE method to train
    a bit-vector of independent latent variables.
    Encoder/Decoder needs to be either BitVectorSFEWrapper or SFEDeterministicWrapper.
    """
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
        discrete_latent_z, encoder_scores, encoder_entropy = \
            self.encoder(encoder_input)
        decoder_output, decoder_scores, decoder_entropy = \
            self.decoder(discrete_latent_z, decoder_input)

        argmax = (encoder_scores > 0).to(torch.float)

        loss, logs = self.loss(
            encoder_input,
            argmax,
            decoder_input,
            decoder_output,
            labels)

        encoder_bernoull_distr = Bernoulli(logits=encoder_scores)
        encoder_sample_log_probs = \
            encoder_bernoull_distr.log_prob(discrete_latent_z).sum(dim=1)
        if len(decoder_scores.size()) != 1:
            decoder_categorical_helper = Bernoulli(logits=decoder_scores)
            decoder_sample_log_probs = \
                decoder_categorical_helper.log_prob(decoder_output).sum(dim=1)
        else:
            decoder_sample_log_probs = decoder_scores

        if self.encoder.baseline_type == 'runavg':
            baseline = self.mean_baseline
        elif self.encoder.baseline_type == 'sample':
            alt_z_sample = encoder_bernoull_distr.sample().detach()
            decoder_output, _, _ = self.decoder(alt_z_sample, decoder_input)
            baseline, _ = self.loss(
                encoder_input,
                alt_z_sample,
                decoder_input,
                decoder_output,
                labels)

        policy_loss = (
            loss.detach() - baseline) * (
                encoder_sample_log_probs + decoder_sample_log_probs)
        entropy_loss = -(
            encoder_entropy *
            self.encoder_entropy_coeff +
            decoder_entropy *
            self.decoder_entropy_coeff)

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
        logs['distr'] = encoder_bernoull_distr

        return {'loss': full_loss, 'log': logs}
