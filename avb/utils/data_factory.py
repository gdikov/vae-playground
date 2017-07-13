from __future__ import division
from __future__ import absolute_import

import os
import numpy as np

from skimage import color
from skimage.io import imread
from skimage.transform import resize

from .datasets import load_mnist

TEXTURES_DIR = os.path.join('..', '..', 'data', 'textures')


class PatternFactory(object):
    def __init__(self):
        self.styles = np.array(('trippy', 'strippy', 'mandelbrot', 'image'))

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
        return pattern

    @staticmethod
    def render_strippy(shape, orientation='horizontal', width=1, height=1, as_gray=False):
        if not as_gray:
            raise NotImplementedError
        if orientation == 'random':
            orientation = np.random.choice(['horizontal', 'vertical', 'checker'])
        if orientation == 'horizontal':
            assert shape[0] // 2 // height > 0, "Invalid shape and block height."
            repeats_to_fit = int(np.ceil(shape[0] / 2 / height))
            col = np.array([[1] * height, [0] * height] * repeats_to_fit).ravel()[None, :]
            pattern = np.repeat(col, shape[1], axis=0).T
        elif orientation == 'vertical':
            repeats_to_fit = int(np.ceil(shape[1] / 2 / width))
            row = np.array([[1] * width, [0] * width] * repeats_to_fit).ravel()[None, :]
            pattern = np.repeat(row, shape[0], axis=0)
        elif orientation == 'diagonal':
            raise NotImplementedError
        elif orientation == 'checker':
            assert shape[0] / 2 / height > 0 and shape[1] / 2 / width > 0, "Invalid shape and block width/height."
            # checkerboard style:
            repeats = int(np.ceil(shape[1] / 2 / width))
            row_1 = np.array([[1]*width, [0]*width]*repeats).ravel()[None, :]
            row_2 = np.array([[0]*width, [1]*width]*repeats).ravel()[None, :]
            two_rows = np.repeat(np.vstack([row_1, row_2]), height, axis=0)
            pattern = np.tile(two_rows.T, int(np.ceil(shape[0] / 2 / height)))
        else:
            raise ValueError("Unknown orientation")

        pattern = pattern[:shape[0], :shape[1]]
        return pattern

    @staticmethod
    def render_mandelbrot(shape, as_gray=False):
        xmin, xmax, ymin, ymax, (height, width), maxiter = -1.5, 0.5, -1.5, 1.5, shape, 10000

        def mandelbrot_numpy(c, maxiter):
            output = np.zeros(c.shape)
            z = np.zeros(c.shape, np.complex64)
            for it in range(maxiter):
                notdone = np.less(z.real * z.real + z.imag * z.imag, 4.0)
                output[notdone] = it
                z[notdone] = z[notdone] ** 2 + c[notdone]
            output[output == maxiter - 1] = 0
            return output

        r1 = np.linspace(-2, -2, width, dtype=np.float32)
        r2 = np.linspace(-2, -2, height, dtype=np.float32)
        c = r1 + r2[:, None] * 1j
        pattern = mandelbrot_numpy(c, maxiter).T
        if as_gray:
            pattern = color.rgb2gray(pattern)
        return pattern

    @staticmethod
    def render_image(shape, image_filename, as_gray=False):
        pattern = imread(os.path.join(TEXTURES_DIR, image_filename), as_grey=as_gray)
        pattern = resize(pattern, shape, order=2)
        return pattern

    def get_styles(self):
        return self.styles


class CustomMNIST(object):
    """
    Factory for customized MNIST digits. Background will have different structure/style.
    """
    def __init__(self):
        self.data = load_mnist(one_hot=False, binarised=False, rotated=False, background=None, large_set=True)
        self.unique_labels = np.arange(10)
        # group the targets of each dataset by label
        sorted_indices = np.argsort(self.data['target'])
        sorted_labels = self.data['target'][sorted_indices]
        first_uniques = np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1]))
        count_uniques = np.diff(np.nonzero(first_uniques)[0])
        self.grouped_data = [self.data['data'][ids] for ids in np.split(sorted_indices, np.cumsum(count_uniques))]

        self.pattern_generator = PatternFactory()
        self.styles = self.pattern_generator.get_styles()

    def _generate_style(self, style):
        if style == 0:
            background = self.pattern_generator.render_trippy((28, 28), as_gray=True)
        elif style == 1:
            background = self.pattern_generator.render_strippy((28, 28), orientation='random', as_gray=True)
        elif style == 2:
            background = self.pattern_generator.render_mandelbrot((28, 28), as_gray=True)
        elif style == 3:
            raise NotImplementedError
        else:
            raise ValueError("Unknown style id")
        return background

    def generate(self, n_samples, style_distribution=None, label_distribution=None):
        """
        Generate samples with different background styles.

        Args:
            n_samples: int, total number of samples to be generated
            style_distribution: list, array of percentage generations for each style (uniform if None)
            label_distribution: list, array of percentage generations for each label (uniform if None)

        Returns:
            A tuple of a dict with `data`, `target` and `style` keys and a verbal descriptions of the style encoding.
        """

        def compose_new(new_background, old_sample):
            assert new_background.ravel().shape == old_sample.shape
            old_sample = old_sample.reshape(new_background.shape)
            mask_digit = old_sample > 0
            new_sample = old_sample.copy()
            new_sample[~mask_digit] = new_background[~mask_digit]
            return new_sample

        augmented_data = {'data': [], 'target': self.data['target']}
        if style_distribution is None:
            style_distribution = np.ones(self.styles.size) / self.styles.size
        if label_distribution is None:
            label_distribution = np.ones(self.unique_labels.size) / self.unique_labels.size

        s_ids = np.random.choice(self.styles.size, size=n_samples, replace=True, p=style_distribution)
        l_ids = np.random.choice(self.unique_labels.size, size=n_samples, replace=True, p=label_distribution)
        for s_id, l_id in zip(s_ids, l_ids):
            new_sample_background = self._generate_style(style=s_id)
            random_id = np.random.choice(len(self.grouped_data[l_id]))
            newly_composed_digit = compose_new(new_sample_background, self.grouped_data[l_id][random_id])
            augmented_data['data'].append(newly_composed_digit)

        return augmented_data
