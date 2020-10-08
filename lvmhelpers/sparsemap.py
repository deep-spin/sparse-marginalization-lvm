import numpy as np
import torch

from lpsmap.ad3qp.factor_graph import PFactorGraph
from .pbernoulli import PFactorBernoulli
from .pbudget import PFactorBudget
from .psequence import PFactorSequenceBinary


class SparseMAP(torch.autograd.Function):

    @classmethod
    def run_sparsemap(cls, ctx, x):
        ctx.n = x.shape[0]
        ctx.fg = PFactorGraph()
        ctx.fg.set_verbosity(1)
        ctx.variables = [ctx.fg.create_binary_variable()
                         for _ in range(2 * ctx.n)]
        ctx.f = cls.make_factor(ctx)
        ctx.fg.declare_factor(ctx.f, ctx.variables)
        x_np = x.detach().cpu().numpy().astype(np.double)
        x_np = np.concatenate([x_np, np.zeros_like(x_np)])

        # initialize better
        if ctx.init:
            init = torch.rand(x_np.shape[0], dtype=torch.double)
            ctx.f.init_active_set_from_scores(init, [])

        _, _ = ctx.f.solve_qp(x_np, [], max_iter=ctx.max_iter)
        aset, p = ctx.f.get_sparse_solution()

        p = p[:len(aset)]
        aset = torch.tensor(aset, dtype=torch.float32, device=x.device)
        p = torch.tensor(p).to(x.device)
        ctx.mark_non_differentiable(aset)
        return p, aset

    @classmethod
    def jv(cls, ctx, dp):
        d_eta_u = np.empty(2 * ctx.n, dtype=np.float32)
        d_eta_v = np.empty(0, dtype=np.float32)
        ctx.f.dist_jacobian_vec(dp.cpu().numpy(), d_eta_u, d_eta_v)
        d_eta_u = torch.tensor(d_eta_u[:ctx.n], dtype=torch.float32, device=dp.device)
        return d_eta_u


class SequenceSparseMAP(torch.autograd.Function):

    @classmethod
    def run_sparsemap(cls, ctx, x, t):
        ctx.n = x.shape[0]
        ctx.fg = PFactorGraph()
        ctx.fg.set_verbosity(1)
        ctx.variables = [ctx.fg.create_binary_variable()
                         for _ in range(2 * ctx.n)]
        ctx.f = cls.make_factor(ctx)
        ctx.fg.declare_factor(ctx.f, ctx.variables)
        x_np = x.detach().cpu().numpy().astype(np.double)
        x_np = np.concatenate([x_np, np.zeros_like(x_np)])

        # edge potentials
        t_np = t.detach().cpu().numpy().astype(np.double)
        ctx.f.set_additional_log_potentials(t_np)

        # initialize better
        if ctx.init:
            init = torch.rand(x_np.shape[0], dtype=torch.double)
            additionals = torch.zeros_like(t_np)
            ctx.f.init_active_set_from_scores(init, additionals)

        _, _ = ctx.f.solve_qp(x_np, t_np, max_iter=ctx.max_iter)
        aset, p = ctx.f.get_sparse_solution()

        p = p[:len(aset)]
        aset = torch.tensor(aset, dtype=torch.float32, device=x.device)
        p = torch.tensor(p).to(x.device)
        ctx.mark_non_differentiable(aset)
        return p, aset

    @classmethod
    def jv(cls, ctx, dp):
        d_eta_u = np.empty(2 * ctx.n, dtype=np.float32)
        d_eta_v = np.empty(ctx.n-1, dtype=np.float32)
        ctx.f.dist_jacobian_vec(dp.cpu().numpy(), d_eta_u, d_eta_v)
        d_eta_u = torch.tensor(d_eta_u[:ctx.n], dtype=torch.float32, device=dp.device)
        d_eta_v = torch.tensor(d_eta_v, dtype=torch.float32, device=dp.device)
        return d_eta_u, d_eta_v


class BernSparseMAP(SparseMAP):

    @classmethod
    def make_factor(cls, ctx):
        f = PFactorBernoulli()
        f.initialize(ctx.n)
        return f

    @classmethod
    def forward(cls, ctx, x, max_iter, init):
        ctx.max_iter = max_iter
        ctx.init = init
        return cls.run_sparsemap(ctx, x)

    @classmethod
    def backward(cls, ctx, dp, daset):
        return cls.jv(ctx, dp), None, None, None


class BudgetSparseMAP(SparseMAP):

    @classmethod
    def make_factor(cls, ctx):
        f = PFactorBudget()
        f.initialize(ctx.n, ctx.budget)
        return f

    @classmethod
    def forward(cls, ctx, x, budget, max_iter, init):
        ctx.n = x.shape[0]
        ctx.init = init
        ctx.max_iter = max_iter
        ctx.budget = budget
        return cls.run_sparsemap(ctx, x)

    @classmethod
    def backward(cls, ctx, dp, daset):
        return cls.jv(ctx, dp), None, None, None, None


class SequenceBinarySparseMAP(SequenceSparseMAP):

    @classmethod
    def make_factor(cls, ctx):
        f = PFactorSequenceBinary()
        f.initialize(ctx.n)
        return f

    @classmethod
    def forward(cls, ctx, x, t, max_iter, init):
        ctx.n = x.shape[0]
        ctx.init = init
        ctx.max_iter = max_iter
        return cls.run_sparsemap(ctx, x, t)

    @classmethod
    def backward(cls, ctx, dp, daset):
        d_eta_u, d_eta_v = cls.jv(ctx, dp)
        return d_eta_u, d_eta_v, None, None


def bernoulli_smap(x, max_iter=100, init=True):
    return BernSparseMAP.apply(x, max_iter, init)


def budget_smap(x, budget=5, max_iter=100, init=True):
    return BudgetSparseMAP.apply(x, budget, max_iter, init)


def sequence_smap(x, t, max_iter=100, init=True):
    return SequenceBinarySparseMAP.apply(x, t, max_iter, init)


def main():

    torch.manual_seed(42)

    x = torch.randn(5, requires_grad=True)
    print(x)
    p, aset = bernoulli_smap(x)
    print(p)
    print(aset)

    p, aset = budget_smap(x, budget=3)
    print(p)
    print(aset)

    print(torch.autograd.grad(p[0], x))
    print(torch.autograd.grad(p[1], x))


if __name__ == '__main__':
    main()
