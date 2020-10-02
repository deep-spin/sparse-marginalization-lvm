import pickle
import torch.utils.data as data
import torch.nn.parallel
import os
import torch
import numpy as np


class _BatchIterator:
    def __init__(self, loader, seed=None):
        self.loader = loader
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        batch_data = self.get_batch()
        return batch_data

    def get_batch(self):
        loader = self.loader
        bsz = loader.bsz
        game_size = loader.game_size
        same = loader.same

        C = len(self.loader.dataset.obj2id.keys())  # number of concepts
        images_indexes_sender = np.zeros((bsz, game_size))

        for b in range(bsz):
            if same:
                # randomly sample a concept
                concepts = self.random_state.choice(C, 1)
                c = concepts[0]
                ims = loader.dataset.obj2id[c]["ims"]
                idxs_sender = self.random_state.choice(
                    ims, game_size, replace=False)
                images_indexes_sender[b, :] = idxs_sender
            else:
                idxs_sender = []
                # randomly sample k concepts
                concepts = self.random_state.choice(
                    C, game_size, replace=False)
                for i, c in enumerate(concepts):
                    ims = loader.dataset.obj2id[c]["ims"]
                    idx = self.random_state.choice(ims, 2, replace=False)
                    idxs_sender.append(idx[0])

                images_indexes_sender[b, :] = np.array(idxs_sender)

        images_vectors_sender = []

        for i in range(game_size):
            x, _ = loader.dataset[images_indexes_sender[:, i]]
            images_vectors_sender.append(x)

        images_vectors_sender = torch.stack(images_vectors_sender).contiguous()
        y = torch.zeros(bsz).long()

        images_vectors_receiver = torch.zeros_like(images_vectors_sender)
        for i in range(bsz):
            permutation = torch.randperm(game_size)

            images_vectors_receiver[:, i,
                                    :] = images_vectors_sender[permutation, i, :]
            y[i] = permutation.argmin()
        return images_vectors_sender, images_vectors_receiver, y


class ImagenetLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed')
        self.bsz = kwargs.pop('batch_size')
        self.game_size = kwargs.pop('game_size')
        self.same = kwargs.pop('same')

        super(ImagenetLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _BatchIterator(self, seed=seed)


class ImageNetFeat(data.Dataset):
    def __init__(self, root, train=True):
        import h5py

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        # FC features
        fc_file = os.path.join(root, 'ours_images_single_sm0.h5')

        fc = h5py.File(fc_file, 'r')
        # There should be only 1 key
        key = list(fc.keys())[0]
        # Get the data
        data = torch.FloatTensor(list(fc[key]))

        # normalise data
        img_norm = torch.norm(data, p=2, dim=1, keepdim=True)
        normed_data = data / img_norm

        objects_file = os.path.join(root,
                                    'ours_images_single_sm0.objects')
        with open(objects_file, "rb") as f:
            labels = pickle.load(f)
        objects_file = os.path.join(root,
                                    'ours_images_paths_sm0.objects')
        with open(objects_file, "rb") as f:
            paths = pickle.load(f)

        self.create_obj2id(labels)
        self.data_tensor = normed_data
        self.labels = labels
        self.paths = paths

    def __getitem__(self, index):
        return self.data_tensor[index], index

    def __len__(self):
        return self.data_tensor.size(0)

    def create_obj2id(self, labels):
        self.obj2id = {}
        keys = {}
        idx_label = -1
        for i in range(labels.shape[0]):
            if not labels[i] in keys.keys():
                idx_label += 1
                keys[labels[i]] = idx_label
                self.obj2id[idx_label] = {}
                self.obj2id[idx_label]['labels'] = labels[i]
                self.obj2id[idx_label]['ims'] = []
            self.obj2id[idx_label]['ims'].append(i)
