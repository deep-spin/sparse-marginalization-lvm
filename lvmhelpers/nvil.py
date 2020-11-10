import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli


class BaselineNN(nn.Module):
    """
    Neural network that outputs the baseline for NVIL.
    """
    def __init__(self, input_size):
        super(BaselineNN, self).__init__()

        self.input_size = input_size
        # define the linear layers
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, image):

        # feed through neural network
        h = image.view(-1, self.input_size)

        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = self.fc4(h)
        # h: [batch_size]
        return h


class NVILWrapper(nn.Module):
    """
    NVIL Wrapper for a network. Assumes that the during the forward pass,
    the network returns scores over the potential output categories.
    The wrapper transforms them into a tuple of (sample from the Categorical,
    log-prob of the sample, entropy for the Categorical).
    """
    def __init__(self, agent, input_size):
        super(NVILWrapper, self).__init__()
        self.agent = agent
        self.baseline_nn = BaselineNN(input_size)

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- Categorical sample.
                Size: [batch_size]
            scores {torch.Tensor} -- the output of the network.
                Important to compute the policy component of the loss.
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


class NVIL(torch.nn.Module):
    """
    The training loop for the NVIL method to train discrete latent variables.
    Encoder needs to be NVILWrapper.
    Decoder needs to be utils.DeterministicWrapper.
    """
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(NVIL, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff

    def forward(self, encoder_input, decoder_input, labels):
        discrete_latent_z, encoder_scores, encoder_entropy = \
            self.encoder(encoder_input)
        decoder_output = \
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

        baseline = self.encoder.baseline_nn(encoder_input).squeeze()

        baseline = baseline.reshape(-1, loss.size(0)).mean(dim=0).squeeze()

        policy_loss = ((loss - baseline).detach() * encoder_sample_log_probs).mean()
        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)
        mse = ((loss.detach()-baseline)**2).mean()

        full_loss = policy_loss + loss.mean() + mse + entropy_loss

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['baseline'] = baseline
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['distr'] = encoder_categorical_helper.probs

        return {'loss': full_loss, 'log': logs}


class BitVectorNVILWrapper(nn.Module):
    """
    NVIL Wrapper for a network that parameterizes
    independent Bernoulli distributions.
    Assumes that the during the forward pass,
    the network returns scores for the Bernoulli parameters.
    The wrapper transforms them into a tuple of (sample from the Bernoulli,
    log-prob of the sample, entropy for the independent Bernoulli).
    """
    def __init__(self, agent, input_size):
        super(BitVectorNVILWrapper, self).__init__()
        self.agent = agent
        self.baseline_nn = BaselineNN(input_size)

    def forward(self, *args, **kwargs):
        scores = self.agent(*args, **kwargs)

        distr = Bernoulli(logits=scores)
        entropy = distr.entropy().sum(dim=1)

        sample = distr.sample()

        return sample, scores, entropy


class BitVectorNVIL(torch.nn.Module):
    """
    The training loop for the NVIL method to train
    a bit-vector of independent latent variables.
    Encoder needs to be BitVectorNVILWrapper.
    Decoder needs to be utils.DeterministicWrapper.
    """
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(BitVectorNVIL, self).__init__()
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
        decoder_output = self.decoder(discrete_latent_z, decoder_input)

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

        baseline = self.encoder.baseline_nn(encoder_input).squeeze()

        policy_loss = (loss - baseline).detach() * encoder_sample_log_probs
        entropy_loss = - encoder_entropy * self.encoder_entropy_coeff
        mse = (loss.detach()-baseline)**2

        full_loss = (policy_loss + loss + mse + entropy_loss).mean()

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['baseline'] = baseline
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['distr'] = encoder_bernoull_distr

        return {'loss': full_loss, 'log': logs}
