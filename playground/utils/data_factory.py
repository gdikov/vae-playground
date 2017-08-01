from __future__ import division
from __future__ import absolute_import

import logging
import os
import numpy as np

from skimage import color
from skimage.io import imread
from skimage.transform import resize

from .datasets import load_mnist

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config("global_config.yaml")

DATA_DIR = config['data_dir']
TEXTURES_DIR = os.path.join(DATA_DIR, 'textures')

np.random.seed(config['seed'])


class PatternFactory(object):
    def __init__(self):
        self.styles = np.array(('trippy', 'horizontal', 'vertical', 'checkerboard',
                                'mandelbrot', 'black', 'image'))

    @staticmethod
    def render_trippy(shape, as_gray=False):
        h, w = shape
        x, y = np.arange(-w // 2, w // 2) * 4 * 2 * np.pi / w, np.arange(-h // 2, h // 2) * 4 * 2 * np.pi / h
        r = np.sin(np.outer(x, y) * y)  # np.outer generates the complete matrix for the red channel
        g = np.cos(y) * np.cos(np.outer(x, y))  # green
        b = np.cos(np.outer(x, y) + np.pi)  # blue
        pattern = (np.array((r, g, b)) * 128 + 127).transpose(1, 2, 0).astype(np.uint8)
        if as_gray:
            pattern = color.rgb2gray(pattern)
        tag = 'trippy'
        return {'image': pattern, 'tag': tag}

    @staticmethod
    def render_strippy(shape, orientation='horizontal', width=1, height=1, as_gray=False):
        if not as_gray:
            raise NotImplementedError
        if orientation == 'vertical_or_horizontal':
            orientation = np.random.choice(['horizontal', 'vertical'])
        if orientation == 'random':
            orientation = np.random.choice(['horizontal', 'vertical', 'checker'])
        if orientation == 'horizontal':
            assert shape[0] // 2 // height > 0, "Invalid shape and block height."
            repeats_to_fit = int(np.ceil(shape[0] / 2 / height))
            col = np.array([[1] * height, [0] * height] * repeats_to_fit).ravel()[None, :]
            pattern = np.repeat(col, shape[1], axis=0).T
            tag = 'horizontal'
        elif orientation == 'vertical':
            repeats_to_fit = int(np.ceil(shape[1] / 2 / width))
            row = np.array([[1] * width, [0] * width] * repeats_to_fit).ravel()[None, :]
            pattern = np.repeat(row, shape[0], axis=0)
            tag = 'vertical'
        elif orientation == 'diagonal':
            tag = 'diagonal'
            raise NotImplementedError
        elif orientation == 'checker':
            assert shape[0] / 2 / height > 0 and shape[1] / 2 / width > 0, "Invalid shape and block width/height."
            # checkerboard style:
            repeats = int(np.ceil(shape[1] / 2 / width))
            row_1 = np.array([[1] * width, [0] * width] * repeats).ravel()[None, :]
            row_2 = np.array([[0] * width, [1] * width] * repeats).ravel()[None, :]
            two_rows = np.repeat(np.vstack([row_1, row_2]), height, axis=0)
            pattern = np.tile(two_rows.T, int(np.ceil(shape[0] / 2 / height)))
            tag = 'checker'
        else:
            raise ValueError("Unknown orientation")

        pattern = pattern[:shape[0], :shape[1]]
        return {'image': pattern, 'tag': tag}

    @staticmethod
    def render_mandelbrot(shape, as_gray=False):
        xmin, xmax, ymin, ymax, height, width, maxiter = -1.5, 0.5, -1.5, 1.5, 200, 200, 1000

        def mandelbrot(c, maxiter):
            output = np.zeros(c.shape)
            z = np.zeros(c.shape, np.complex64)
            for it in range(maxiter):
                notdone = np.less(z.real * z.real + z.imag * z.imag, 4.0)
                output[notdone] = it
                z[notdone] = z[notdone] ** 2 + c[notdone]
            output[output == maxiter - 1] = 0
            return output

        r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
        r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
        c = r1 + r2[:, None] * 1j
        pattern = mandelbrot(c, maxiter).T
        if as_gray:
            pattern = color.rgb2gray(pattern)
            pattern = (pattern - pattern.min()) / pattern.max()
        pattern = resize(pattern, shape, order=3)
        tag = 'mandelbrot'
        return {'image': pattern, 'tag': tag}

    @staticmethod
    def render_black(shape, as_gray=False):
        if as_gray:
            pattern = np.zeros(shape)
        else:
            pattern = np.zeros((3,) + shape)
        return {'image': pattern, 'tag': 'black'}

    @staticmethod
    def render_image(shape, image_filename, as_gray=False):
        pattern = imread(os.path.join(TEXTURES_DIR, image_filename), as_grey=as_gray)
        pattern = resize(pattern, shape, order=3)
        tag = os.path.splitext(os.path.basename(image_filename))[0]
        return {'image': pattern, 'tag': tag}

    def get_styles(self):
        return self.styles


class CustomMNIST(object):
    """
    Factory for customized MNIST digits. Background will have different structure/style.
    """

    def __init__(self):
        self.data = load_mnist(one_hot=False, binarised=False, rotated=False,
                               background=None, large_set=True)
        self.unique_labels = np.arange(10)
        # group the targets of each dataset by label
        sorted_indices = np.argsort(self.data['target'])
        sorted_labels = self.data['target'][sorted_indices]
        first_uniques = np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1]))
        count_uniques = np.diff(np.nonzero(first_uniques)[0])
        self.grouped_data = [self.data['data'][ids] for ids in np.split(sorted_indices, np.cumsum(count_uniques))]

        self.pattern_generator = PatternFactory()
        self.styles = self.pattern_generator.get_styles()
        self.cache = {}

    def _generate_style(self, style, **kwargs):
        self.cache = {}
        as_gray = kwargs.get('as_gray', True)
        if style in self.cache.keys():
            return self.cache[style]
        if style == 'trippy':
            background = self.pattern_generator.render_trippy((28, 28), as_gray=as_gray)
            self.cache[style] = background
        elif style == 'strippy':
            orientation = kwargs.get('orientation', 'random')
            background = self.pattern_generator.render_strippy((28, 28), orientation=orientation, as_gray=as_gray)
            self.cache[style] = background
        elif style == 'horizontal':
            background = self.pattern_generator.render_strippy((28, 28), orientation='horizontal', as_gray=as_gray)
            self.cache[style] = background
        elif style == 'vertical':
            background = self.pattern_generator.render_strippy((28, 28), orientation='vertical', as_gray=as_gray)
            self.cache[style] = background
        elif style == 'checkerboard':
            background = self.pattern_generator.render_strippy((28, 28), orientation='checkerboard', as_gray=as_gray)
            self.cache[style] = background
        elif style == 'mandelbrot':
            background = self.pattern_generator.render_mandelbrot((28, 28), as_gray=as_gray)
            self.cache[style] = background
        elif style == 'black':
            background = self.pattern_generator.render_black((28, 28), as_gray=as_gray)
            self.cache[style] = background
        else:
            raise ValueError("Unknown style id")
        return background

    def generate(self, n_samples, style_distribution=None, label_distribution=None, **kwargs):
        """
        Generate samples with different background styles.

        Args:
            n_samples: int, total number of samples to be generated
            style_distribution: list, array of percentage generations for each style (uniform if None)
            label_distribution: list, array of percentage generations for each label (uniform if None)

        Returns:
            A tuple of a dict with `data`, `target` and `style` keys and a verbal descriptions of the style encoding.
        """

        generated_data = {'data': [], 'target': [], 'tag': []}
        if label_distribution is None:
            label_distribution = np.ones(self.unique_labels.size) / self.unique_labels.size
        if style_distribution is None:
            style_distribution = np.ones(self.styles.size) / self.styles.size
        s_ids = np.random.choice(self.styles, size=n_samples, replace=True, p=style_distribution)
        l_ids = np.random.choice(self.unique_labels.size, size=n_samples, replace=True, p=label_distribution)
        for s_id, l_id in zip(s_ids, l_ids):
            new_sample_background = self._generate_style(style=s_id, **kwargs)
            random_id = np.random.choice(len(self.grouped_data[l_id]))
            newly_composed_digit = self._compose_new(new_sample_background['image'],
                                                     self.grouped_data[l_id][random_id])
            generated_data['data'].append(newly_composed_digit)
            generated_data['target'].append(l_id)
            generated_data['tag'].append(new_sample_background['tag'])
        generated_data['data'] = np.asarray(generated_data['data'])
        generated_data['target'] = np.asarray(generated_data['target'])
        generated_data['tag'] = np.asarray(generated_data['tag'])
        return generated_data

    def augment(self, style_distribution, **kwargs):
        """
        Generate backgrounds for MNIST images, without changing their order
        
        Args:
            style_distribution: list, array of percentage generations for each style (uniform if None)

        Returns:
            A tuple of a dict with `data`, `target` and `style` keys and a verbal descriptions of the style encoding.
        """

        augmented_data = {'data': [], 'target': [], 'tag': []}
        data_size = self.data['target'].shape[0]
        s_ids = np.random.choice(self.styles, size=data_size, replace=True, p=style_distribution)
        classes, counts = np.unique(s_ids, return_counts=True)
        logger.info("Augmenting MNIST dataset with background classes {} (distribution: {})".format(classes, counts))
        for id_ in range(data_size):
            s_id = s_ids[id_]
            new_sample_background = self._generate_style(style=s_id, **kwargs)
            newly_composed_digit = self._compose_new(new_sample_background['image'], self.data['data'][id_])
            augmented_data['data'].append(newly_composed_digit)
            augmented_data['target'].append(self.data['target'][id_])
            augmented_data['tag'].append(new_sample_background['tag'])
        augmented_data['data'] = np.asarray(augmented_data['data'])
        augmented_data['target'] = np.asarray(augmented_data['target'])
        augmented_data['tag'] = np.asarray(augmented_data['tag'])
        return augmented_data

    @staticmethod
    def _compose_new(new_background, old_sample):
        assert new_background.ravel().shape == old_sample.shape
        old_sample = old_sample.reshape(new_background.shape)
        mask_digit = old_sample > 0
        new_sample = old_sample.copy()
        new_sample[~mask_digit] = new_background[~mask_digit]
        return new_sample

    @staticmethod
    def save_dataset(new_data, tag):
        assert isinstance(new_data, dict), "Provide a dict data containing at least the keys `data` and `target`."
        new_data_path = os.path.join(DATA_DIR, 'MNIST_Custom_Variations')
        if not os.path.exists(new_data_path):
            os.makedirs(new_data_path)
        if not new_data['data'].shape[-1] == 28**2:
            new_data['data'] = new_data['data'].reshape(-1, 28**2)
            logger.debug("Implicit reshape to {}".format(new_data['data'].shape))
        np.savez(os.path.join(new_data_path, tag), **new_data)
        logger.info("Saved {} dataset at location: {}".format(tag, new_data_path))
        return new_data_path

    def load_dataset(self, tag, generate_if_none=True, default_distribution=None):
        if tag.endswith('.npz'):
            dataset_composition = os.path.splitext(tag)[0].split('_')
        else:
            dataset_composition = tag.split('_')
            tag += '.npz'
        dataset_path = os.path.join(DATA_DIR, 'MNIST_Custom_Variations', tag)
        if not os.path.exists(dataset_path):
            if generate_if_none:
                if default_distribution is None:
                    styles = {k: v for v, k in enumerate(self.styles)}
                    style_dist = np.zeros(len(list(styles.keys())))
                    for s in dataset_composition:
                        style_dist[styles[s]] = 1 / len(dataset_composition)
                else:
                    style_dist = default_distribution
                custom_data = self.augment(style_distribution=style_dist)
                self.save_dataset(custom_data, tag)
                return custom_data
            else:
                raise FileNotFoundError("No dataset with tag {} was found. Enable "
                                        "`generate_if_none` to build one.".format(tag))
        else:
            return np.load(dataset_path)
