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


class ExplicitDeterministicWrapper(nn.Module):
    """
    """
    def __init__(self, agent):
        super(ExplicitDeterministicWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        out = self.agent(*args, **kwargs)

        return out, torch.zeros(1).to(out.device), torch.zeros(1).to(out.device)


class SymbolGameExplicit(nn.Module):
    """
    A single-symbol Sender/Receiver game implemented with Explicit expectation.
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0):
        """
        :param sender: Sender agent. On forward, returns a tuple of (message, log-prob of the message, entropy).
        :param receiver: Receiver agent. On forward, accepts a message and the dedicated receiver input. Returns
            a tuple of (output, log-probs, entropy).
        :param loss: The loss function that accepts:
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs the end-to-end loss. Can be non-differentiable; if it is differentiable, this will be leveraged
        :param sender_entropy_coeff: The entropy regularization coefficient for Sender
        :param receiver_entropy_coeff: The entropy regularizatino coefficient for Receiver
        """
        super(SymbolGameExplicit, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.sender_entropy_coeff = sender_entropy_coeff

    def forward(self, sender_input, labels, receiver_input=None):
        message, sender_probs, sender_entropy = self.sender(sender_input)
        batch_size, vocab_size = sender_probs.shape

        entropy_loss = -(sender_entropy.mean() * self.sender_entropy_coeff)
        if self.training:
            losses = torch.zeros_like(sender_probs)
            rest_info_global = None
            for possible_message in range(vocab_size):
                if sender_probs[:, possible_message].sum().item() != 0:
                    # if it's zero, all batch examples will be multiplied by zero anyway, so skip computations
                    possible_message_ = possible_message + torch.zeros(batch_size, dtype=torch.long).to(sender_probs.device)
                    receiver_output, receiver_log_prob, receiver_entropy = self.receiver(possible_message_, receiver_input)
                    possible_message_loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
                    losses[:, possible_message] += possible_message_loss

                    if not rest_info_global:
                        rest_info_global = {k: 0.0 for k in rest_info.keys()}
                    for k, v in rest_info.items():
                        if hasattr(v, 'mean'):
                            # expectation of accuracy
                            rest_info_global[k] += (sender_probs[:, possible_message] * v).mean().item()

            for k, v in rest_info.items():
                if hasattr(v, 'mean'):
                    rest_info[k] = rest_info_global[k]

            # sender_probs: [batch_size, vocab_size]
            # losses: [batch_size, vocab_size]
            # sender_probs.unsqueeze(1): [batch_size, 1, vocab_size]
            # losses.unsqueeze(-1): [batch_size, vocab_size, 1]
            # entropy_loss: []
            # full_loss: [batch_size]
            full_loss = sender_probs.unsqueeze(1).bmm(losses.unsqueeze(-1)).squeeze() + entropy_loss
        else:
            receiver_output, receiver_log_prob, receiver_entropy = self.receiver(message, receiver_input)
            loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)

            full_loss = loss + entropy_loss

            for k, v in rest_info.items():
                if hasattr(v, 'mean'):
                    rest_info[k] = v.mean().item()

        rest_info['sender_entropy'] = sender_entropy.mean().item()
        rest_info['nonzeros'] = (sender_probs != 0).sum(-1).to(torch.float).mean().item()

        return full_loss.mean(), rest_info
