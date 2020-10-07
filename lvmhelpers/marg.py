import torch
import torch.nn as nn
from entmax import entmax15, sparsemax


def entropy(p):
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
        logits = self.agent(*args, **kwargs)
        distr = self.normalizer(logits, dim=-1)
        entropy_distr = entropy(distr)
        sample = logits.argmax(dim=-1)
        return sample, distr, entropy_distr


class Marginalizer(torch.nn.Module):
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
        message, encoder_probs, encoder_entropy = self.encoder(encoder_input)
        batch_size, vocab_size = encoder_probs.shape

        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)
        if self.training:
            losses = torch.zeros_like(encoder_probs)
            logs_global = None

            for possible_message in range(vocab_size):
                if encoder_probs[:, possible_message].sum().detach() != 0:
                    # if it's zero, all batch examples
                    # will be multiplied by zero anyway,
                    # so skip computations
                    possible_message_ = \
                        possible_message + \
                        torch.zeros(
                            batch_size, dtype=torch.long).to(encoder_probs.device)
                    decoder_output = self.decoder(
                        possible_message_, decoder_input)

                    loss_sum_term, logs = self.loss(
                        encoder_input,
                        message,
                        decoder_input,
                        decoder_output,
                        labels)

                    losses[:, possible_message] += loss_sum_term

                    if not logs_global:
                        logs_global = {k: 0.0 for k in logs.keys()}
                    for k, v in logs.items():
                        if hasattr(v, 'mean'):
                            # expectation of accuracy
                            logs_global[k] += (
                                encoder_probs[:, possible_message] * v).mean()

            for k, v in logs.items():
                if hasattr(v, 'mean'):
                    logs[k] = logs_global[k]

            # encoder_probs: [batch_size, vocab_size]
            # losses: [batch_size, vocab_size]
            # encoder_probs.unsqueeze(1): [batch_size, 1, vocab_size]
            # losses.unsqueeze(-1): [batch_size, vocab_size, 1]
            # entropy_loss: []
            # full_loss: []
            loss = encoder_probs.unsqueeze(1).bmm(losses.unsqueeze(-1)).squeeze()
            full_loss = loss.mean() + entropy_loss.mean()

        else:
            decoder_output = self.decoder(message, decoder_input)
            loss, logs = self.loss(
                encoder_input,
                message,
                decoder_input,
                decoder_output,
                labels)

            full_loss = loss.mean() + entropy_loss

            for k, v in logs.items():
                if hasattr(v, 'mean'):
                    logs[k] = v.mean()

        logs['baseline'] = torch.zeros(1).to(loss.device)
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['decoder_entropy'] = torch.zeros(1).to(loss.device)
        # TODO: nonzero for every epoch end
        logs['nonzeros'] = (encoder_probs != 0).sum(-1).to(torch.float).mean()
        return {'loss': full_loss, 'log': logs}
