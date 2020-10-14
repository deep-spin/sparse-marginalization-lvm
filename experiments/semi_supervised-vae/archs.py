import torch
import torch.nn as nn
import torch.nn.functional as F


def get_one_hot_encoding_from_int(z, n_classes):
    """
    Convert categorical variable to one-hot enoding

    Parameters
    ----------
    z : torch.LongTensor
        Tensor with integers corresponding to categories
    n_classes : Int
        The total number of categories

    Returns
    ----------
    z_one_hot : torch.Tensor
        One hot encoding of z
    """

    z_one_hot = torch.zeros(len(z), n_classes).to(z.device)
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot


class MLPEncoder(nn.Module):
    def __init__(
            self,
            latent_dim=5,
            slen=28,
            n_classes=10):
        # the encoder returns the mean and variance of the latent parameters
        # given the image and its class (one hot encoded)

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.slen = slen
        self.n_classes = n_classes

        # define the linear layers
        self.fc1 = nn.Linear(self.n_pixels + self.n_classes, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, latent_dim * 2)

    def forward(self, image, one_hot_label):
        # label should be one hot encoded
        assert one_hot_label.shape[1] == self.n_classes
        assert image.shape[0] == one_hot_label.shape[0]

        # feed through neural network
        h = image.view(-1, self.n_pixels)
        h = torch.cat((h, one_hot_label), dim=1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        # get means, std, and class weights
        indx1 = self.latent_dim
        indx2 = 2 * self.latent_dim

        latent_means = h[:, 0:indx1]
        latent_std = torch.exp(h[:, indx1:indx2])

        return latent_means, latent_std


class Classifier(nn.Module):
    def __init__(
            self,
            slen=28,
            n_classes=10):

        super(Classifier, self).__init__()

        self.slen = slen
        self.n_pixels = slen ** 2
        self.n_classes = n_classes

        self.fc1 = nn.Linear(self.n_pixels, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, n_classes)

    def forward(self, image):
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)

        return h


class MLPDecoder(nn.Module):
    def __init__(
            self,
            latent_dim=5,
            slen=28,
            n_classes=10):

        # This takes the latent parameters and returns the
        # mean and variance for the image reconstruction

        super(MLPDecoder, self).__init__()

        # image/model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.slen = slen

        self.fc1 = nn.Linear(latent_dim + n_classes, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_pixels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_params, one_hot_label):
        assert latent_params.shape[1] == self.latent_dim
        # label should be one hot encoded
        assert one_hot_label.shape[1] == self.n_classes
        assert latent_params.shape[0] == one_hot_label.shape[0]

        h = torch.cat((latent_params, one_hot_label), dim=1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        h = h.view(-1, self.slen, self.slen)

        image_mean = self.sigmoid(h)

        return image_mean


class MNISTVAE(nn.Module):

    def __init__(self, encoder, decoder):
        super(MNISTVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert self.encoder.latent_dim == self.decoder.latent_dim
        assert self.encoder.n_classes == self.decoder.n_classes
        assert self.encoder.slen == self.decoder.slen

        # save some parameters
        self.latent_dim = self.encoder.latent_dim
        self.n_classes = self.encoder.n_classes
        self.slen = self.encoder.slen

    def get_one_hot_encoding_from_label(self, label):
        return get_one_hot_encoding_from_int(label, self.n_classes)

    def forward(self, discrete_latent_z, image):

        if len(discrete_latent_z.size()) != 2:
            one_hot_label = torch.zeros(
                len(discrete_latent_z), self.n_classes).to(image.device)
            one_hot_label.scatter_(
                1, discrete_latent_z.view(-1, 1), 1)
            one_hot_label = one_hot_label.view(
                len(discrete_latent_z), self.n_classes)
        else:
            one_hot_label = discrete_latent_z

        assert one_hot_label.shape[0] == image.shape[0]
        assert one_hot_label.shape[1] == self.n_classes

        # pass through encoder
        latent_means, latent_std = self.encoder(image, one_hot_label)

        # sample latent dimension
        latent_samples = torch.randn(
            latent_means.shape).to(latent_means.device) * \
            latent_std + latent_means

        assert one_hot_label.shape[0] == latent_samples.shape[0]
        assert one_hot_label.shape[1] == self.n_classes

        # pass through decoder
        image_mean = self.decoder(latent_samples, one_hot_label)

        output = {
            'latent_means': latent_means,
            'latent_std': latent_std,
            'latent_samples': latent_samples,
            'image_mean': image_mean
        }
        return output
