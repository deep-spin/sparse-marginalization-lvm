import torch
import torch.nn as nn
from entmax import entmax15, sparsemax

# from .sparsemap import bernoulli_smap, budget_smap
from .pbinary_topk import batched_topk


def entropy(p):
    nz = (p > 0).to(p.device)

    eps = torch.finfo(p.dtype).eps
    p_stable = p.clone().clamp(min=eps, max=1 - eps)

    out = torch.where(
        nz,
        p_stable * torch.log(p_stable),
        torch.tensor(0., device=p.device, dtype=torch.float))

    return -(out).sum(-1)


class TopKSparsemaxWrapper(nn.Module):
    """
    """
    def __init__(self, agent, k=10):
        super(TopKSparsemaxWrapper, self).__init__()
        self.agent = agent

        self.k = k

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)
        batch_size, latent_size = logits.shape
        # get the top-k bit-vectors
        bit_vector_z = torch.empty((batch_size, self.k, latent_size), dtype=torch.float32)
        batched_topk(logits.detach().cpu().numpy(), bit_vector_z.numpy(), self.k)
        bit_vector_z = bit_vector_z.to(logits.device)

        # rank the top-k using sparsemax
        scores = torch.einsum("bkj,bj->bk", bit_vector_z, logits)
        distr = sparsemax(scores, dim=-1)

        # get the entropy
        distr_flat = distr.view(-1)
        mask = distr_flat > 0
        distr_flat = distr_flat[mask]
        entropy_distr = - distr_flat @ torch.log(distr_flat)

        if not self.training:
            # get the argmax sample
            sample_idx = scores.argmax(dim=-1)
            sample_idx = \
                sample_idx + \
                torch.arange(0, batch_size*self.k, self.k).to(sample_idx.device)
            sample = bit_vector_z.view(-1, latent_size)[sample_idx]
        else:
            sample = bit_vector_z

        return sample, distr, entropy_distr / batch_size


class TopKSparsemaxMarg(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(TopKSparsemaxMarg, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff

    def forward(self, encoder_input, decoder_input, labels):
        bit_vector_z, encoder_probs, encoder_entropy = self.encoder(encoder_input)
        batch_size = bit_vector_z.shape[0]
        k = self.encoder.k
        latent_size = bit_vector_z.shape[-1]

        entropy_loss = -(encoder_entropy * self.encoder_entropy_coeff)
        if self.training:
            # bit_vector_z: [batch_size, k, latent_size]
            # bit_vector_z_flat: [batch_size * k, latent_size]
            bit_vector_z_flat = bit_vector_z.view(-1, latent_size)
            # encoder_input: [batch_size, input_size]
            # encoder_input_rep: [batch_size, k, input_size]
            # encoder_input_rep_flat: [batch_size * k, input_size]
            encoder_input_rep = encoder_input.unsqueeze(1).repeat((1, k, 1))
            encoder_input_rep_flat = encoder_input_rep.view(-1, encoder_input.shape[-1])

            # encoder_probs: [batch_size, k]
            # encoder_probs_flat: [batch_size * k]
            encoder_probs_flat = encoder_probs.view(-1)

            # removing components that would end up being zero-ed out
            mask = encoder_probs_flat > 0
            # encoder_input_rep_flat: [<=batch_size * k, input_size]
            encoder_input_rep_flat = encoder_input_rep_flat[mask]
            # encoder_probs_flat: [<=batch_size * k]
            encoder_probs_flat = encoder_probs_flat[mask]
            # bit_vector_z_flat: [<=batch_size * k, latent_size]
            bit_vector_z_flat = bit_vector_z_flat[mask]

            # decoder_output: [<=batch_size * k, input_size, out_classes]
            decoder_output = self.decoder(bit_vector_z_flat, decoder_input)

            # loss_components: [<=batch_size * k]
            loss_components, logs = self.loss(
                encoder_input,
                bit_vector_z_flat,
                decoder_input,
                decoder_output,
                labels)

            # loss: []
            loss = (encoder_probs_flat @ loss_components) / batch_size

        else:
            decoder_output = self.decoder(bit_vector_z, decoder_input)
            loss, logs = self.loss(
                encoder_input,
                bit_vector_z,
                decoder_input,
                decoder_output,
                labels)

        full_loss = loss.mean() + entropy_loss

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['loss'] = loss.mean().detach()
        logs['encoder_entropy'] = encoder_entropy.detach()
        logs['support'] = (encoder_probs > 0).sum(dim=-1).detach()
        return {'loss': full_loss, 'log': logs}


class SparseMAPWrapper(nn.Module):
    """
    """
    def __init__(self, agent, normalizer='entmax'):
        super(SparseMAPWrapper, self).__init__()
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


class SparseMAPMarg(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(SparseMAPMarg, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff

    def forward(self, encoder_input, decoder_input, labels):
        bit_vector_z, encoder_probs, encoder_entropy = self.encoder(encoder_input)
        batch_size, latent_size = encoder_probs.shape

        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)
        if self.training:
            losses = torch.zeros_like(encoder_probs)
            logs_global = None

            for possible_bit_vector_z in range(latent_size):
                if encoder_probs[:, possible_bit_vector_z].sum().detach() != 0:
                    # if it's zero, all batch examples
                    # will be multiplied by zero anyway,
                    # so skip computations
                    possible_bit_vector_z_ = \
                        possible_bit_vector_z + \
                        torch.zeros(
                            batch_size, dtype=torch.long).to(encoder_probs.device)
                    decoder_output = self.decoder(
                        possible_bit_vector_z_, decoder_input)

                    loss_sum_term, logs = self.loss(
                        encoder_input,
                        bit_vector_z,
                        decoder_input,
                        decoder_output,
                        labels)

                    losses[:, possible_bit_vector_z] += loss_sum_term

                    if not logs_global:
                        logs_global = {k: 0.0 for k in logs.keys()}
                    for k, v in logs.items():
                        if hasattr(v, 'mean'):
                            # expectation of accuracy
                            logs_global[k] += (
                                encoder_probs[:, possible_bit_vector_z] * v).mean()

            for k, v in logs.items():
                if hasattr(v, 'mean'):
                    logs[k] = logs_global[k]

            # encoder_probs: [batch_size, latent_size]
            # losses: [batch_size, latent_size]
            # encoder_probs.unsqueeze(1): [batch_size, 1, latent_size]
            # losses.unsqueeze(-1): [batch_size, latent_size, 1]
            # entropy_loss: []
            # full_loss: []
            loss = encoder_probs.unsqueeze(1).bmm(losses.unsqueeze(-1)).squeeze()
            full_loss = loss.mean() + entropy_loss.mean()

        else:
            decoder_output = self.decoder(bit_vector_z, decoder_input)
            loss, logs = self.loss(
                encoder_input,
                bit_vector_z,
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
