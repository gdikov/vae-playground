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
                altered_data = {key: np.concatenate([data[key], data[key][additional_samples_indices]], axis=0)
                                for key in data.keys()}
            else:
                altered_data = np.concatenate([data, data[additional_samples_indices]], axis=0)
        else:
            altered_data = data
        n_batches = (data_size + additional_samples) // batch_size
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
            batches_indices = np.split(np.arange(data_size), n_batches)
            for batch_indices in batches_indices:
                yield data[batch_indices].astype(np.float32)


class ConjointAVBDataIterator(DataIterator):
    def __init__(self, data_dim, latent_dim, seed=7):
        super(ConjointAVBDataIterator, self).__init__(seed=seed, data_dim=data_dim, latent_dim=latent_dim)

    def iter_data_training(self, data, n_batches, **kwargs):
        shuffle = kwargs.get('shuffle', True)
        unique_labels = np.unique(data[0]['target'])
        mode = kwargs.get('grouping_mode', 'by_pairs')

        if not (all([len(d['data']) == len(d['target']) for d in data])
                and all([len(d['data']) == len(data[0]['data']) for d in data[1:]])
                and np.all([np.unique(d['target']) == unique_labels for d in data[1:]])):
            raise ValueError("Datasets are not equal in size or not well formed or not homogeneous in labels.")

        data_size = len(data[0]['data'])
        batch_size = data_size // n_batches

        def pairwise_iter():
            while True:
                for i in range(n_batches):
                    if shuffle:
                        # shuffling is only pair-id-wise
                        sample_ids = np.random.choice(data_size, size=batch_size)
                    else:
                        if n_batches * batch_size < data_size and i == n_batches - 1:
                            # shouldn't happen to do this because of the size augmentation beforehand
                            # nevertheless keep this safety code here.
                            sample_ids = np.concatenate([np.arange(i * batch_size, data_size),
                                                         np.arange(0, data_size - batch_size * n_batches)])
                        else:
                            sample_ids = np.arange(i * batch_size, (i + 1) * batch_size)
                    batch = [d['data'][sample_ids].astype(np.float32) for d in data]
                    yield batch

        def targetwise_iter(key):
            if not shuffle:
                logger.warning("Iteration with grouping by target will enable shuffling within a group. Only groups are"
                               "iterated in a sequential order. If this is your intention, ignore this warning, "
                               "otherwise you might want to run it with `group_mode='by_pairs'` instead.")
            indices_groups = []
            for d in data:
                # group the targets of each dataset by label
                sorted_indices = np.argsort(d[key])
                sorted_labels = d[key][sorted_indices]
                first_uniques = np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1]))
                count_uniques = np.diff(np.nonzero(first_uniques)[0])
                indices_groups.append(np.split(sorted_indices, np.cumsum(count_uniques)))
            while True:
                for i in range(n_batches):
                    if shuffle:
                        # sample batch_size many samples from each dataset such that the labels are the same
                        # however, samples may have different labels within the batch.
                        random_groups = np.random.randint(low=0, high=unique_labels.size, size=batch_size)
                        sample_ids = np.squeeze([[np.random.choice(dataset_groups[random_group_id], size=1)
                                                  for dataset_groups in indices_groups]
                                                 for random_group_id in random_groups]).T
                    else:
                        # here all samples have the same label (target) but are randomly sampled within the group.
                        group_id = int(i / (n_batches / len(unique_labels)))
                        if np.all([indices_groups[0][group_id].size == dataset_groups[group_id].size
                                   for dataset_groups in indices_groups[1:]]):
                            random_samples_from_group = np.random.choice(indices_groups[0][group_id].size,
                                                                         size=batch_size)
                            sample_ids = np.squeeze([dataset_groups[group_id][random_samples_from_group]
                                                     for dataset_groups in indices_groups])
                        else:
                            # the data groups are unequally sized so some samples in the larger dataset will be missed
                            smaller_group = np.asscalar(np.argmin([dataset_groups[group_id].size
                                                                   for dataset_groups in indices_groups]))
                            random_samples_from_group = np.random.choice(indices_groups[smaller_group][group_id],
                                                                         size=batch_size)
                            sample_ids = [random_samples_from_group] * len(indices_groups)
                    batch = [data[j]['data'][ids].astype(np.float32) for j, ids in enumerate(sample_ids)]
                    yield batch

        if mode == 'by_pairs':
            batch_gen = pairwise_iter()
        elif mode == 'by_targets':
            batch_gen = targetwise_iter(key='target')
        else:
            raise ValueError('Grouping mode can only be `by_pairs` or `by_targets`.')
        while True:
            yield next(batch_gen)

    def iter_data_inference(self, data, n_batches, **kwargs):
        group_by_target = kwargs.get('group_by_target', False)
        kwargs['shuffle'] = False
        if group_by_target:
            return self.iter_data_training(data, n_batches, **kwargs)
        else:
            # assume that the two datasets are identical in terms of size, labels and sample order
            # and differ only in non-target information, e.g. background, texture etc.
            assert all([d['data'].shape[0] == data[0]['data'].shape[0] for d in data[1:]]), \
                "Datasets of unequal size cannot be iterated without grouping"
            data_size = data[0]['data'].shape[0]
            batches_indices = np.split(np.arange(data_size), n_batches)
            for batch_indices in batches_indices:
                yield [d['data'][batch_indices].astype(np.float32) for d in data]

    def iter_data_generation(self, data, n_batches, **kwargs):
        data_size = data.shape[0]
        batches_indices = np.split(np.arange(data_size), n_batches)
        for batch_indices in batches_indices:
            yield data[batch_indices].astype(np.float32)


ConjointVAEDataIterator = ConjointAVBDataIterator
VAEDataIterator = AVBDataIterator
