import gzip
import logging
import struct
import scipy.io

import numpy as np
import os

try:
    from urllib.request import urlretrieve
except ImportError:
    # this works under Python 2
    from urllib import urlretrieve
from skimage import color

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config("global_config.yaml")

PROJECT_DATA_DIR = config['data_dir']
np.random.seed(config['seed'])


def load_usps(local_data_path=None):
    """
    Load the USPS dataset from local file or download it if not available.

    Args:
        local_data_path: path to the USPS dataset. Assumes unpacked files and original filenames.

    Returns:
        A dict with `data` and `target` keys with the USPS data grayscale images.
    """

    usps_path = os.path.join(PROJECT_DATA_DIR, "USPS")
    if local_data_path is None and not os.path.exists(usps_path):
        logger.info("Path to locally stored data not provided. Proceeding with downloading the USPS dataset.")
        url = 'http://www.cs.nyu.edu/~roweis/data/usps_all.mat'
        if not os.path.exists(usps_path):
            os.makedirs(usps_path)
        file_name = os.path.join(usps_path, 'usps_all.mat')
        urlretrieve(url, file_name)
        usps_imgs, usps_labels = _load_usps_from_file(usps_path)
        usps = {'data': usps_imgs, 'target': usps_labels}
    else:
        local_data_path = local_data_path or usps_path
        logger.info("Loading USPS dataset from {}".format(local_data_path))
        if os.path.exists(local_data_path):
            usps_imgs, usps_labels = _load_usps_from_file(local_data_path)
            usps = {'data': usps_imgs, 'target': usps_labels}
        else:
            logger.error("Path to locally stored USPS dataset does not exist.")
            raise ValueError

    return usps


def _load_usps_from_file(data_dir=None):
    """
    Load the binary files from disk.

    Args:
        data_dir: path to folder containing the USPS dataset blobs in a .mat file.

    Returns:
        A numpy array with the images and a numpy array with the corresponding labels.
    """
    # The files are assumed to have these names and should be found in 'path'

    train_file = os.path.join(data_dir, 'usps_all.mat')

    train_dict = scipy.io.loadmat(train_file)
    images = train_dict['data']
    images = np.reshape(images, [256, 1100*10])
    labels = np.repeat(np.eye(10), 1100, axis=0)

    return images, labels


def load_svhn(local_data_path=None, one_hot=True, grayscale=True, use_extra_set=False):
    """
    Load the SVHN dataset from local file or download it if not available.

    Args:
        local_data_path: path to the SVHN dataset. Assumes unpacked files and original filenames.
        one_hot: bool whether tha data targets should be converted to one hot encoded labels
        grayscale: bool, whether the images should be converted to a grayscale images.
        use_extra_set: bool, whether to use the additional dataset with easier examples

    Returns:
        A dict with `data` and `target` keys with the SVHN data converted to grayscale images.
    """

    def convert_to_one_hot(raw_target):
        n_uniques = len(np.unique(raw_target))
        one_hot_target = np.zeros((raw_target.shape[0], n_uniques))
        one_hot_target[np.arange(raw_target.shape[0]), raw_target.astype(np.int)] = 1
        return one_hot_target

    def convert_to_grayscale(raw_data):
        # converts the h x w x 3 images to h x w images with grayscale values
        # the conversion used is Y = 0.2125 R + 0.7154 G + 0.0721 B
        gray_data = color.rgb2gray(raw_data)
        return gray_data

    svhn_path = os.path.join(PROJECT_DATA_DIR, "SVHN")
    if local_data_path is None and not os.path.exists(svhn_path):
        logger.info("Path to locally stored data not provided. Proceeding with downloading the SVHN dataset.")
        url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
        if not os.path.exists(svhn_path):
            os.makedirs(svhn_path)
        file_name = os.path.join(svhn_path, 'train_32x32.mat')
        urlretrieve(url, file_name)
        file_name1 = os.path.join(svhn_path, 'extra_32x32.mat')
        url1 = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
        urlretrieve(url1, file_name1)
        svhn_imgs, svhn_labels = _load_svhn_from_file(svhn_path, use_extra_set)
        if one_hot:
            svhn_labels = convert_to_one_hot(svhn_labels)
        svhn = {'data': svhn_imgs, 'target': svhn_labels}
    else:
        local_data_path = local_data_path or svhn_path
        logger.info("Loading SVHN dataset from {}".format(local_data_path))
        if os.path.exists(local_data_path):
            svhn_imgs, svhn_labels = _load_svhn_from_file(local_data_path, use_extra_set)
            svhn_imgs = svhn_imgs.transpose((3, 0, 1, 2))
            if one_hot:
                svhn_labels = convert_to_one_hot(svhn_labels)
            svhn = {'data': svhn_imgs, 'target': svhn_labels}
        else:
            logger.error("Path to locally stored SVHN dataset does not exist.")
            raise ValueError

    if grayscale:
        svhn['data'] = convert_to_grayscale(svhn['data'])

    return svhn


def _load_svhn_from_file(data_dir=None, extra_set=False):
    """
    Load the binary files from disk.

    Args:
        data_dir: path to folder containing the SVHN dataset blobs in a .mat file.

    Returns:
        A numpy array with the images and a numpy array with the corresponding labels.
    """
    # The files are assumed to have these names and should be found in 'path'

    train_file = os.path.join(data_dir, 'train_32x32.mat')
    extra_file = os.path.join(data_dir, 'extra_32x32.mat')

    train_dict = scipy.io.loadmat(train_file)
    images = np.asarray(train_dict['X'])
    labels = np.asarray(train_dict['y'])

    if extra_set:
        train_dict = scipy.io.loadmat(extra_file)
        e_images = np.asarray(train_dict['X'])
        e_labels = np.asarray(train_dict['y'])
        images = np.concatenate((images, e_images), axis=0)
        labels = np.concatenate((labels, e_labels), axis=0)

    labels = labels.flatten()
    # labels mapping maps 1-9 digits to 1-9 label, however 0 has label 10 (for some reason), hence the reassignment.
    labels[labels == 10] = 0

    return images, labels


def load_mnist(local_data_path=None, one_hot=True, binarised=True):
    """
    Load the MNIST dataset from local file or download it if not available.
    
    Args:
        local_data_path: path to the MNIST dataset. Assumes unpacked files and original filenames. 
        one_hot: bool whether tha data targets should be converted to one hot encoded labels
        binarised: bool, whether the images should be ceiled/floored to 1 and 0 respectively.

    Returns:
        A dict with `data` and `target` keys with the MNIST data converted to [0, 1] floats. 
    """

    def convert_to_one_hot(raw_target):
        n_uniques = len(np.unique(raw_target))
        one_hot_target = np.zeros((raw_target.shape[0], n_uniques))
        one_hot_target[np.arange(raw_target.shape[0]), raw_target.astype(np.int)] = 1
        return one_hot_target

    def binarise(raw_data, mode='sampling', **kwargs):
        if mode == 'sampling':
            return np.random.binomial(1, p=raw_data).astype(np.int32)
        elif mode == 'threshold':
            threshold = kwargs.get('threshold', 0.5)
            return (raw_data > threshold).astype(np.int32)

    # For the binarised MNIST dataset one can use the original version from:
    #       http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat
    #       http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat
    #       http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat
    #
    # However this dataset does not contain label information and hence it is better go binarise it manually.

    mnist_path = os.path.join(PROJECT_DATA_DIR, "MNIST")
    if local_data_path is None and not os.path.exists(mnist_path):
        logger.info("Path to locally stored data not provided. Proceeding with downloading the MNIST dataset.")
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(mnist_path, one_hot=one_hot)
        for filename in os.listdir(mnist_path):
            if filename.endswith('.gz'):
                unzipped = gzip.open(os.path.join(mnist_path, filename), 'rb').read()
                with open(os.path.join(mnist_path, filename[:-3]), 'wb') as f:
                    f.write(unzipped)
        mnist_data = np.concatenate((mnist.train.images, mnist.test.images, mnist.validation.images))
        mnist_labels = np.concatenate((mnist.train.labels, mnist.test.labels, mnist.validation.labels))
        mnist = {'data': mnist_data,
                 'target': mnist_labels}
    else:
        local_data_path = local_data_path or mnist_path
        logger.info("Loading MNIST dataset from {}".format(local_data_path))
        if os.path.exists(local_data_path):
            mnist_imgs, mnist_labels = _load_mnist_from_file(local_data_path)
            if one_hot:
                mnist_labels = convert_to_one_hot(mnist_labels)
            mnist = {'data': mnist_imgs, 'target': mnist_labels}
        else:
            logger.error("Path to locally stored MNIST does not exist.")
            raise ValueError

    if binarised:
        mnist['data'] = binarise(mnist['data'], mode='threshold')

    return mnist


def _load_mnist_from_file(data_dir=None):
    """
    Load the binary files from disk. 
    
    Args:
        data_dir: path to folder containing the MNIST dataset blobs. 

    Returns:
        A numpy array with the images and a numpy array with the corresponding labels. 
    """
    # The files are assumed to have these names and should be found in 'path'
    image_files = ('train-images-idx3-ubyte', 't10k-images-idx3-ubyte')
    label_files = ('train-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')

    def read_labels(fname):
        with open(os.path.join(data_dir, fname), 'rb') as flbl:
            # remove header
            struct.unpack(">II", flbl.read(8))
            digit_labels = np.fromfile(flbl, dtype=np.int8)
        return digit_labels

    def read_images(fname):
        with open(os.path.join(data_dir, fname), 'rb') as fimg:
            # remove header
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            digit_images = np.fromfile(fimg, dtype=np.uint8).reshape(num, -1)
        return digit_images

    images = np.concatenate([read_images(fname) for fname in image_files]) / 255.
    labels = np.concatenate([read_labels(fname) for fname in label_files])

    return images, labels


def load_8schools():
    """
    Load Eight Schools experiment as in "Estimation in parallel randomized experiments, Donald B. Rubin, 1981"
    
    Returns:
        A dict with keys `effect` and `stderr` with numpy arrays of shape (8,) for each of the schools 
    """
    #   school effect stderr
    # 1      A  28.39   14.9
    # 2      B   7.94   10.2
    # 3      C  -2.75   16.3
    # 4      D   6.82   11.0
    # 5      E  -0.64    9.4
    # 6      F   0.63   11.4
    # 7      G  18.01   10.4
    # 8      H  12.16   17.6
    estimated_effects = np.array([28.39, 7.74, -2.75, 6.82, -0.64, 0.63, 18.01, 12.16])
    std_errors = np.array([14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6])
    return {'effect': estimated_effects, 'stderr': std_errors}


def load_npoints(n, noisy=False, n_variations=1):
    """
    Load a generalisation of the 4 points synthetic dataset as described in the Experiments section, Generative models, 
    Synthetic example in "Adversarial Variational Bayes, L. Mescheder et al., 2017". 

    Args:
        n: int, number of distinct data points (i.e. dimensionality of the (vector) space in which they reside)
        noisy: bool, whether small Gaussian noise should be added to the dataset(s)
        n_variations: int, number of samples per class (for the noisy case only)
        
    Returns:
        A dict with keys `data` and `target` containing the data points and the corresponding labels or a list of
        multiple such dataset dicts.
    """
    if isinstance(n, int):
        n = tuple([n])
    # else assume that it is a list or tuple of dimensionalities
    datasets = []
    for dim in n:
        data = np.eye(dim)
        labels = np.arange(dim)
        if noisy:
            if n_variations > 1:
                data = np.repeat(data, n_variations, axis=0)
                labels = np.repeat(labels, n_variations, axis=0)
            flattened_view = data.ravel()
            flattened_view[flattened_view == 0] = 0.2 + 0.1 * np.random.standard_normal(data.size - dim * n_variations)
            data = np.clip(data, 0, 1)
        datasets.append({'data': data, 'target': labels})
    if len(datasets) == 1:
        # unlist the result
        return datasets[0]

    return tuple(datasets)
