import torch
import torch.nn as nn
from entmax import sparsemax

from .sparsemap import bernoulli_smap, budget_smap
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

            # decoder_input: [batch_size, input_size]
            # decoder_input_rep: [batch_size, k, input_size]
            # decoder_input_rep_flat: [batch_size * k, input_size]
            decoder_input_rep = decoder_input.unsqueeze(1).repeat((1, k, 1))
            decoder_input_rep_flat = decoder_input_rep.view(-1, decoder_input.shape[-1])

            # TODO: this label format is specific to VAE...
            # labels: [batch_size, input_size]
            # labels_rep: [batch_size, k, input_size]
            # labels_rep_flat: [batch_size * k, input_size]
            labels_rep = labels.unsqueeze(1).repeat((1, k, 1))
            labels_rep_flat = labels_rep.view(-1, labels.shape[-1])

            # encoder_probs: [batch_size, k]
            # encoder_probs_flat: [batch_size * k]
            encoder_probs_flat = encoder_probs.view(-1)

            # removing components that would end up being zero-ed out
            mask = encoder_probs_flat > 0
            # encoder_input_rep_flat: [<=batch_size * k, input_size]
            encoder_input_rep_flat = encoder_input_rep_flat[mask]
            # decoder_input_rep_flat: [<=batch_size * k, input_size]
            decoder_input_rep_flat = decoder_input_rep_flat[mask]
            # labels_rep_flat: [<=batch_size * k, input_size]
            labels_rep_flat = labels_rep_flat[mask]
            # encoder_probs_flat: [<=batch_size * k]
            encoder_probs_flat = encoder_probs_flat[mask]
            # bit_vector_z_flat: [<=batch_size * k, latent_size]
            bit_vector_z_flat = bit_vector_z_flat[mask]

            # decoder_output: [<=batch_size * k, input_size, out_classes]
            decoder_output = self.decoder(bit_vector_z_flat, decoder_input)

            # loss_components: [<=batch_size * k]
            loss_components, logs = self.loss(
                encoder_input_rep_flat,
                bit_vector_z_flat,
                decoder_input_rep_flat,
                decoder_output,
                labels_rep_flat)

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
        logs['support'] = (encoder_probs > 0).sum(dim=-1)
        return {'loss': full_loss, 'log': logs}


class SparseMAPWrapper(nn.Module):
    """
    """
    def __init__(self, agent, budget=0, init=False, max_iter=300):
        super(SparseMAPWrapper, self).__init__()
        self.agent = agent
        self.budget = budget
        self.init = init
        self.max_iter = max_iter

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)
        batch_size, latent_size = logits.shape

        distr = []
        sample = []
        idxs = []

        support = []
        for k in range(batch_size):
            zl = logits[k]
            if self.budget > 0:
                distri, samplei = budget_smap(
                    zl, budget=self.budget, init=self.init, max_iter=self.max_iter)
            else:
                distri, samplei = bernoulli_smap(
                    zl, init=self.init, max_iter=self.max_iter)
            samplei = samplei[distri > 0]
            distri = distri[distri > 0]
            supp = len(distri)
            assert supp > 0
            sample.append(samplei)
            distr.append(distri)
            idxs.extend(supp * [k])
            support.append(supp)

        sample = torch.cat(sample)
        distr = torch.cat(distr)
        entropy_distr = -distr @ torch.log(distr)

        return sample, distr, entropy_distr / batch_size, idxs, support


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
        bit_vector_z, encoder_probs, encoder_entropy, idxs, support = \
            self.encoder(encoder_input)
        batch_size = encoder_input.shape[0]

        entropy_loss = -(encoder_entropy * self.encoder_entropy_coeff)

        decoder_output = self.decoder(bit_vector_z)

        loss_components, logs = self.loss(
            encoder_input[idxs],
            bit_vector_z,
            decoder_input[idxs],
            decoder_output,
            labels[idxs])

        loss = (encoder_probs @ loss_components) / batch_size
        full_loss = loss + entropy_loss

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['baseline'] = torch.zeros(1).to(loss.device)
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['decoder_entropy'] = torch.zeros(1).to(loss.device)
        logs['support'] = torch.tensor(support).to(loss.device)
        return {'loss': full_loss, 'log': logs}
