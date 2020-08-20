import torch.nn as nn


class DeterministicWrapper(nn.Module):
    """
    Simple wrapper that makes a deterministic agent.
    No sampling is run on top of the wrapped agent,
    it is passed as is.
    """
    def __init__(self, agent):
        super(DeterministicWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        out = self.agent(*args, **kwargs)
        return out
