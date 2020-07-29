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
