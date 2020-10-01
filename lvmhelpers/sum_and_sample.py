"""
Most of the code here is inspired by or copied from
https://github.com/Runjing-Liu120/RaoBlackwellizedSGD/
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def get_concentrated_mask(class_weights, topk):
    """
    Returns a logical mask indicating the categories with the top k largest
    probabilities, as well as the catogories corresponding to those with the
    top k largest probabilities.

    Parameters
    ----------
    class_weights : torch.Tensor
        Array of class weights, with each row corresponding to a datapoint,
        each column corresponding to the probability of the datapoint
        belonging to that category
    topk : int
        the k in top-k

    Returns
    -------
    mask_topk : torch.Tensor
        Boolean array, same dimension as class_weights,
        with entry 1 if the corresponding class weight is
        in the topk for that observation
    topk_domain: torch.LongTensor
        Array specifying the indices of class_weights that correspond to
        the topk observations
    """

    mask_topk = torch.zeros(class_weights.shape).to(class_weights.device)

    seq_tensor = torch.LongTensor([i for i in range(class_weights.shape[0])])

    if topk > 0:
        _, topk_domain = torch.topk(class_weights, topk)

        for i in range(topk):
            mask_topk[seq_tensor, topk_domain[:, i]] = 1
    else:
        topk_domain = None

    return mask_topk, topk_domain, seq_tensor


class SumAndSampleWrapper(nn.Module):
    """
    """
    def __init__(self, agent, topk=10):
        super(SumAndSampleWrapper, self).__init__()
        self.agent = agent
        self.topk = topk

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        sample = logits.argmax(dim=1)

        return sample, logits, entropy


class SumAndSample(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(SumAndSample, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff

    def forward(self, encoder_input, decoder_input, labels):
        discrete_latent_z, encoder_log_prob, encoder_entropy = \
            self.encoder(encoder_input)
        batch_size, vocab_size = encoder_log_prob.shape

        # entropy component of the final loss, we can
        # compute already but will only use it later on
        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)

        if self.training:
            encoder_prob = torch.exp(encoder_log_prob.detach())
            # this is the indicator C_k
            concentrated_mask, topk_domain, seq_tensor = \
                get_concentrated_mask(encoder_prob, self.encoder.topk)
            concentrated_mask = concentrated_mask.float().detach()

            ############################
            # compute the summed term
            summed_term = 0.0

            for ii in range(self.encoder.topk):
                # get categories to be summed
                possible_z = topk_domain[:, ii]

                decoder_output = self.decoder(
                    possible_z, decoder_input)

                loss_sum_term, logs = self.loss(
                    encoder_input,
                    possible_z,
                    decoder_input,
                    decoder_output,
                    labels)

                # get log class probabilities
                encoder_log_prob_i = encoder_log_prob[seq_tensor, possible_z]
                # compute gradient estimate
                grad_estimate_loss = \
                    loss_sum_term.detach() * encoder_log_prob_i + loss_sum_term
                # sum
                summed_weights = encoder_prob[seq_tensor, possible_z].squeeze()
                summed_term = summed_term + (grad_estimate_loss * summed_weights)

                # only compute argmax for training log
                if ii == 0:
                    # save this log in a different variable
                    train_logs = logs
                    for k, v in train_logs.items():
                        if hasattr(v, 'mean'):
                            train_logs[k] = v.mean()

            ############################
            # compute sampled term
            sampled_weight = torch.sum(
                encoder_prob * (1 - concentrated_mask),
                dim=1,
                keepdim=True)

            if not(self.encoder.topk == encoder_prob.shape[1]):
                # if we didn't sum everything
                # we sample from the remaining terms

                # class weights conditioned on being in the diffuse set
                conditional_encoder_prob = (encoder_prob + 1e-12) * \
                    (1 - concentrated_mask) / (sampled_weight + 1e-12)

                # sample from conditional distribution
                cat_rv = Categorical(probs=conditional_encoder_prob)
                conditional_z_sample = cat_rv.sample().detach()

                decoder_output = self.decoder(
                    conditional_z_sample, decoder_input)

                loss_sum_term, _ = self.loss(
                    encoder_input,
                    conditional_z_sample,
                    decoder_input,
                    decoder_output,
                    labels)

                # get log class probabilities
                encoder_log_prob_i = encoder_log_prob[seq_tensor, conditional_z_sample]
                # compute gradient estimate
                grad_estimate_loss_sample = \
                    loss_sum_term.detach() * encoder_log_prob_i + loss_sum_term
            else:
                grad_estimate_loss_sample = 0.0

            loss = grad_estimate_loss_sample * sampled_weight.squeeze() + summed_term

            # restore the log of argmax
            logs = train_logs
        else:
            decoder_output = self.decoder(discrete_latent_z, decoder_input)
            loss, logs = self.loss(
                encoder_input,
                discrete_latent_z,
                decoder_input,
                decoder_output,
                labels)

            for k, v in logs.items():
                if hasattr(v, 'mean'):
                    logs[k] = v.mean()

        full_loss = loss.mean() + entropy_loss.mean()

        logs['baseline'] = torch.zeros(1).to(loss.device)
        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['decoder_entropy'] = torch.zeros(1).to(loss.device)
        return {'loss': full_loss, 'log': logs}
