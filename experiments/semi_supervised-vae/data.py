import os

import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def load_mnist_data(data_dir, train=True):
    if not os.path.exists(data_dir):
        print('creaing folder: ', data_dir)
        os.mkdir(data_dir)

    def trans(x):
        return transforms.ToTensor()(x).bernoulli()

    data = MNIST(
        root=data_dir, train=train,
        transform=trans, download=True)

    return data


class MNISTDataSet(Dataset):

    def __init__(
            self,
            data_dir,
            propn_sample=1.0,
            indices=None,
            train_set=True):

        super(MNISTDataSet, self).__init__()

        # Load MNIST dataset
        # This is the full dataset
        self.mnist_data_set = load_mnist_data(
            data_dir=data_dir,
            train=train_set)

        n_image_full = len(self.mnist_data_set.targets)

        # we may wish to subset
        if indices is None:
            self.num_images = round(n_image_full * propn_sample)
            self.sample_indx = np.random.choice(
                n_image_full, self.num_images,
                replace=False)
        else:
            self.num_images = len(indices)
            self.sample_indx = indices

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return {'image': self.mnist_data_set[self.sample_indx[idx]][0].squeeze(),
                'label': self.mnist_data_set[self.sample_indx[idx]][1]}


class CycleConcatDataset(Dataset):
    '''Dataset wrapping multiple train datasets
    Parameters
    ----------
    *datasets : sequence of torch.utils.data.Dataset
        Datasets to be concatenated and cycled
    '''
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        result = []
        for dataset in self.datasets:
            cycled_i = i % len(dataset)
            result.append(dataset[cycled_i])

        return tuple(result)

    def __len__(self):
        return max(len(d) for d in self.datasets)


def get_mnist_dataset_semisupervised(
        data_dir,
        train_test_split_folder,
        eval_test_set=False,
        n_labeled=5000,
        one_of_each=False):

    labeled_indx = np.load(train_test_split_folder + 'labeled_train_indx.npy')
    unlabeled_indx = np.load(train_test_split_folder + 'unlabeled_train_indx.npy')

    assert (n_labeled <= labeled_indx.shape[0])

    if one_of_each:
        train_set = MNISTDataSet(
            data_dir=data_dir, train_set=True)

        one_of_each_indx = []
        for digit in range(10):
            digit_mask = (train_set.mnist_data_set.targets[labeled_indx] == digit)
            digit_indx = np.where(digit_mask)[0]
            indx_choice = np.random.choice(digit_indx)
            one_of_each_indx.append(indx_choice)
        one_of_each_indx = np.array(one_of_each_indx)

        labeled_new = labeled_indx[one_of_each_indx]
        labeled_rest = np.delete(labeled_indx, one_of_each_indx)

        labeled_indx = labeled_new
        unlabeled_indx = np.hstack([unlabeled_indx, labeled_rest])

    elif labeled_indx.shape[0] != n_labeled:
        labeled_new_idxs = np.random.choice(
            len(labeled_indx), n_labeled, replace=False)
        labeled_new = labeled_indx[labeled_new_idxs]
        labeled_rest = np.delete(labeled_indx, labeled_new_idxs)

        labeled_indx = labeled_new
        unlabeled_indx = np.hstack([unlabeled_indx, labeled_rest])

    train_set_labeled = MNISTDataSet(
        data_dir=data_dir,
        indices=labeled_indx,
        train_set=True)
    train_set_unlabeled = MNISTDataSet(
        data_dir=data_dir,
        indices=unlabeled_indx,
        train_set=True)

    if eval_test_set:
        # get test set as usual
        test_set = MNISTDataSet(
            data_dir=data_dir,
            train_set=False)
    else:
        validation_indx = np.load(
            train_test_split_folder + 'validation_indx.npy')
        test_set = MNISTDataSet(
            data_dir=data_dir,
            indices=validation_indx,
            train_set=True)

    return train_set_labeled, train_set_unlabeled, test_set
