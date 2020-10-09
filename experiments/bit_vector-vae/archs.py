import torch


class MLP(torch.nn.Sequential):
    def __init__(self, dim_in, dim_hid, dim_out, n_layers):
        super().__init__()
        Nonlin = torch.nn.ReLU
        self.add_module("layer0", torch.nn.Linear(dim_in, dim_hid))
        self.add_module("act0", Nonlin())
        for i in range(1, n_layers + 1):
            self.add_module(f"layer{i}", torch.nn.Linear(dim_hid, dim_hid))
            self.add_module(f"act{i}", Nonlin())
        self.add_module(
            f"layer{n_layers+1}", torch.nn.Linear(dim_hid, dim_out)
        )


class CategoricalGenerator(torch.nn.Module):
    def __init__(self, gen, n_features, out_rank, n_classes):
        super().__init__()
        self.gen = gen
        self.n_features = n_features
        self.out_rank = out_rank
        self.out = torch.nn.Linear(out_rank, n_classes)

    def forward(self, Z, *args):
        X = self.gen(Z)
        X = X.reshape(-1, self.n_features, self.out_rank)
        return self.out(X)
