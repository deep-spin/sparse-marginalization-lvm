import torch


def transform(pic):
    return torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
