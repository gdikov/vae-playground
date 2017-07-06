from __future__ import division

import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataIterator(object):
    def __init__(self, seed, data_dim, latent_dim):
        np.random.seed(seed)
        self.data_dim = data_dim
        self.latent_dim = latent_dim

    @staticmethod
    def adjust_data_size(data, batch_size, fill_to=None):
        if isinstance(data, dict):
            # interpret the data as dict with keys `data` and `target` and apply augmentation operations equally on both
            data_size = data['data'].shape[0]
        else:
            data_size = data.shape[0]
        if fill_to is not None:
            additional_samples = max(fill_to - data_size, 0)
            data_size = fill_to
        else:
            additional_samples = 0
        n_batches = data_size / batch_size

        if n_batches - int(n_batches) > 0:
            additional_samples += int(np.ceil(n_batches) * batch_size - data_size)
            logger.warning("The dataset cannot be iterated in epochs completely with the current batch size "
                           "and it will be automatically augmented with {} more samples.".format(additional_samples))
            additional_samples_indices = np.random.choice(data_size, size=additional_samples)
            if isinstance(data, dict):
                altered_data = {'data': np.concatenate([data['data'],
                                                        data['data'][additional_samples_indices]], axis=0),
                                'target': np.concatenate([data['target'],
                                                          data['target'][additional_samples_indices]], axis=0)}
            else:
                altered_data = np.concatenate([data, data[additional_samples_indices]], axis=0)
        else:
            altered_data = data
        n_batches = altered_data['data'].shape[0] // batch_size
        return altered_data, n_batches

    def iter_data_inference(self, data, n_batches, **kwargs):
        raise NotImplementedError

    def iter_data_generation(self, data, n_batches,  **kwargs):
        raise NotImplementedError

    def iter_data_training(self, data, n_batches,  **kwargs):
        raise NotImplementedError

    def iter(self, data, batch_size, mode='training', **kwargs):
        if isinstance(data, tuple):
            logger.debug("Iterating over {} datasets.".format(len(data)))
            altered_data, n_batches = [], []
            max_data_size = max([len(d['data']) for d in data])
            for dataset in data:
                altered_i, n_batches_i = self.adjust_data_size(dataset, batch_size=batch_size, fill_to=max_data_size)
                altered_data.append(altered_i)
                n_batches.append(n_batches_i)
            altered_data = tuple(altered_data)
            n_batches = min(n_batches)
        else:
            altered_data, n_batches = self.adjust_data_size(data, batch_size=batch_size)
        iterator = getattr(self, 'iter_data_{}'.format(mode))
        return iterator(altered_data, n_batches, **kwargs), n_batches


class AVBDataIterator(DataIterator):
    def __init__(self, data_dim, latent_dim, seed=7):
        super(AVBDataIterator, self).__init__(seed=seed, data_dim=data_dim, latent_dim=latent_dim)

    def iter_data_training(self, data, n_batches, **kwargs):
        shuffle = kwargs.get('shuffle', True)
        data_size = data.shape[0]
        while True:
            indices_new_order = np.arange(data_size)
            if shuffle:
                np.random.shuffle(indices_new_order)
            batches_indices = np.split(indices_new_order, n_batches)
            # run for 1 epoch
            for batch_indices in batches_indices:
                yield data[batch_indices].astype(np.float32)

    def iter_data_inference(self, data, n_batches, **kwargs):
        data_size = data.shape[0]
        while True:
            for batch_indices in np.split(np.arange(data_size), n_batches):
                yield data[batch_indices].astype(np.float32)

    def iter_data_generation(self, data, n_batches, **kwargs):
        data_size = data.shape[0]
        while True:
            for batch_indices in np.split(np.arange(data_size), n_batches):
                yield data[batch_indices].astype(np.float32)

VAEDataIterator = AVBDataIterator


class ConjointVAEDataIterator(DataIterator):
    def __init__(self, data_dim, latent_dim, seed=7):
        super(ConjointVAEDataIterator, self).__init__(seed=seed, data_dim=data_dim, latent_dim=latent_dim)

    def iter_data_training(self, data, n_batches, **kwargs):
        group_by_target = kwargs.get('group_by_target', True)
        shuffle = kwargs.get('shuffle', True)
        unique_labels = np.unique(data[0]['target'])
        if not (all([len(d['data']) == len(d['target']) for d in data])
                and all([len(d['data']) == len(data[0]['data']) for d in data[1:]])
                and np.all([np.unique(d['target']) == unique_labels for d in data[1:]])):
            raise ValueError("Datasets are not equal in size or not well formed or not homogeneous in labels.")

        data_size = len(data[0]['data'])
        batch_size = data_size // n_batches

        if group_by_target:
            indices_groups = []
            for d in data:
                # group the targets of each dataset by label
                sorted_indices = np.argsort(d['target'])
                sorted_labels = d['target'][sorted_indices]
                first_uniques = np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1]))
                count_uniques = np.diff(np.nonzero(first_uniques)[0])
                indices_groups.append(np.split(sorted_indices, np.cumsum(count_uniques)))
            while True:
                for i in range(n_batches):
                    if shuffle:
                        # sample batch_size many samples from each dataset such that the labels are the same
                        random_groups = np.random.randint(low=0, high=unique_labels.size, size=batch_size)
                        sample_ids = np.squeeze([[np.random.choice(dataset_groups[random_group_id], size=1)
                                                 for dataset_groups in indices_groups]
                                                 for random_group_id in random_groups]).T
                    else:
                        sample_ids = np.squeeze([np.random.choice(dataset_groups[i % len(unique_labels)],
                                                                  size=batch_size)
                                                 for dataset_groups in indices_groups])
                    batch = [data[i]['data'][ids].astype(np.float32) for i, ids in enumerate(sample_ids)]
                    yield batch
        else:
            raise NotImplementedError

    def iter_data_inference(self, data, n_batches, **kwargs):
        kwargs['shuffle'] = False
        return self.iter_data_training(data, n_batches, **kwargs)

    def iter_data_generation(self, data, n_batches, **kwargs):
        raise NotImplementedError
