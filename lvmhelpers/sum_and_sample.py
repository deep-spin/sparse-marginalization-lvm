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
    Sum&Sample Wrapper for a network. Assumes that the during the forward pass,
    the network returns scores over the potential output categories.
    The wrapper transforms them into a tuple of (sample from the Categorical,
    log-prob of the sample, entropy for the Categorical).

    See: https://arxiv.org/abs/1810.04777
    """
    def __init__(self, agent, topk=10, baseline_type=None):
        super(SumAndSampleWrapper, self).__init__()
        self.agent = agent
        self.topk = topk
        self.baseline_type = baseline_type

    def forward(self, *args, **kwargs):
        scores = self.agent(*args, **kwargs)

        distr = Categorical(logits=scores)
        entropy = distr.entropy()

        sample = scores.argmax(dim=-1)

        return sample, scores, entropy


class SumAndSample(torch.nn.Module):
    """
    The training loop for the Sum&Sample method to train discrete latent variables.
    Encoder needs to be SumAndSampleWrapper.
    Decoder needs to be utils.DeterministicWrapper.
    """
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
        self.mean_baseline = 0.0
        self.n_points = 0.0

    def forward(self, encoder_input, decoder_input, labels):
        discrete_latent_z, encoder_scores, encoder_entropy = \
            self.encoder(encoder_input)
        batch_size, _ = encoder_scores.shape

        # encoder_log_prob: [batch_size, latent_size]
        encoder_log_prob = torch.log_softmax(encoder_scores, dim=-1)
        # entropy component of the final loss, we can
        # compute already but will only use it later on
        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)

        if self.training:
            # encoder_prob: [batch_size, latent_size]
            encoder_prob = torch.softmax(encoder_scores.detach(), dim=-1)
            # this is the indicator C_k
            # concentrated_mask: [batch_size, latent_size]
            # topk_domain: [batch_size, self.encoder.topk]
            # seq_tensor: [batch_size]
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

                if self.encoder.baseline_type == 'runavg':
                    baseline = self.mean_baseline
                elif self.encoder.baseline_type == 'sample':
                    alt_z_sample = Categorical(logits=encoder_log_prob).sample().detach()
                    decoder_output = self.decoder(alt_z_sample, decoder_input)
                    baseline, _ = self.loss(
                        encoder_input,
                        alt_z_sample,
                        decoder_input,
                        decoder_output,
                        labels)
                else:
                    baseline = 0.

                # get log class probabilities
                encoder_log_prob_i = encoder_log_prob[seq_tensor, possible_z]
                # compute gradient estimate
                grad_estimate_loss = \
                    (loss_sum_term.detach() - baseline) * encoder_log_prob_i + \
                    loss_sum_term
                # sum
                summed_weights = encoder_prob[seq_tensor, possible_z].squeeze()
                summed_term = summed_term + (grad_estimate_loss * summed_weights)

                if self.training and self.encoder.baseline_type == 'runavg':
                    self.n_points += 1.0
                    self.mean_baseline += (
                        loss_sum_term.detach().mean() - self.mean_baseline
                        ) / self.n_points

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

                if self.encoder.baseline_type == 'runavg':
                    baseline = self.mean_baseline
                if self.encoder.baseline_type == 'sample':
                    alt_z_sample = Categorical(logits=encoder_log_prob).sample().detach()
                    decoder_output = self.decoder(alt_z_sample, decoder_input)
                    baseline, _ = self.loss(
                        encoder_input,
                        alt_z_sample,
                        decoder_input,
                        decoder_output,
                        labels)
                else:
                    baseline = 0.

                # get log class probabilities
                encoder_log_prob_i = encoder_log_prob[seq_tensor, conditional_z_sample]
                # compute gradient estimate
                grad_estimate_loss_sample = \
                    (loss_sum_term.detach() - baseline) * encoder_log_prob_i + \
                    loss_sum_term

                if self.training and self.encoder.baseline_type == 'runavg':
                    self.n_points += 1.0
                    self.mean_baseline += (
                        loss_sum_term.detach().mean() - self.mean_baseline
                        ) / self.n_points
            else:
                grad_estimate_loss_sample = 0.0

            loss = grad_estimate_loss_sample * sampled_weight.squeeze() + summed_term

            # restore the log of argmax
            logs = train_logs

        with torch.no_grad():
            decoder_output = self.decoder(discrete_latent_z, decoder_input)
            map_loss, map_logs = self.loss(
                encoder_input,
                discrete_latent_z,
                decoder_input,
                decoder_output,
                labels)

        for k, v in map_logs.items():
            if hasattr(v, 'mean'):
                map_logs[k] = v.mean()

        if not self.training:
            loss, logs = map_loss, map_logs

        full_loss = loss.mean() + entropy_loss

        logs['loss'] = map_loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['distr'] = encoder_prob

        return {'loss': full_loss, 'log': logs}
