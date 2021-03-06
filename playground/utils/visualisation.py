from builtins import range

import logging
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)


def _cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                      for i in range(N+1)]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


def _colorbar_index(ncolors, cmap):
    cmap = _cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))


def plot_latent_2d(latent_vars, target=None, fig_dirpath=None, fig_name=None):
    """
    Plot in 2D samples from the latent space.

    Args:
        latent_vars: ndarray, the latent samples with shape (N, 2)
        target: ndarray, the numeric labels used for coloring of the latent samples, shape is (N, 1)
        fig_dirpath: str, optional path to folder where the figure will be saved and not showed
        fig_name: str, optional name of the plot figure

    Returns:

    """
    fig_name = fig_name or 'latent_samples.png'
    if not fig_name.endswith('.png'):
        fig_name += '.png'
    logger.info("Plotting 2D latent space.")
    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('viridis')
    if target is not None:
        if target.dtype.type is np.str_:
            map2digit = {k: i for i, k in enumerate(np.unique(target))}
            target = list(map(lambda x: map2digit[x], target))
        plt.scatter(latent_vars[:, 0], latent_vars[:, 1], c=target, s=1, cmap=cmap)
    else:
        raise NotImplementedError
    # n_distinct = np.unique(target).size
    # _colorbar_index(ncolors=n_distinct, cmap=cmap)
    # plt.colorbar()
    if fig_dirpath is not None:
        if not os.path.exists(fig_dirpath):
            os.makedirs(fig_dirpath)
        plt.savefig(os.path.join(fig_dirpath, fig_name))
    else:
        plt.show()


def plot_sampled_data(data, fig_dirpath=None, fig_name=None):
    """
    Plot the generated samples in a large square plot of the concatenated generated images.

    Args:
        data: ndarray, the generated samples with shape (N, data_dim)
        fig_dirpath: str, optional path to folder where the figure will be saved and not showed
        fig_name: str, optional name of the plot figure

    Returns:

    """
    logger.info("Plotting sampled data.")
    fig_name = fig_name or 'generated_samples.png'
    if not fig_name.endswith('.png'):
        fig_name += '.png'
    data_dim = data.shape[1]
    sample_side_size = int(np.sqrt(data_dim))
    data = data.reshape(-1, sample_side_size, sample_side_size)
    data_size = data.shape[0]
    samples_per_fig_side = int(np.sqrt(data_size))
    data = data[:samples_per_fig_side**2].reshape(samples_per_fig_side, samples_per_fig_side,
                                                  sample_side_size, sample_side_size)
    data = np.concatenate(np.concatenate(data, axis=1), axis=1)
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='Greys_r')
    if fig_dirpath is not None:
        if not os.path.exists(fig_dirpath):
            os.makedirs(fig_dirpath)
        plt.savefig(os.path.join(fig_dirpath, fig_name))
    else:
        plt.show()


def plot_reconstructed_data(data, reconstructed_data, fig_dirpath=None, fig_name=None, pad_to_size=None):
    """
    Plot pairwise data and reconstructed data images in a large plot.

    Args:
        data: ndarray, original data samples of shape (N, data_dim)
        reconstructed_data: ndarray, reconstructed data samples of the same shape as data
        fig_dirpath: str, optional path to folder where the figure will be saved and not showed
        fig_name: str, optional name of the plot figure
        pad_to_size: int, optional number of padding zeros in case the data dimensionality doesn't reshape to square
            and a customised size is wanted (otherwise the padding will be to the next largest square)

    Returns:

    """
    logger.info("Plotting reconstructed data.")
    fig_name = fig_name or 'reconstructed_samples.png'
    if not fig_name.endswith('.png'):
        fig_name += '.png'

    data_size, data_dim = data.shape
    sample_side_size = int(np.sqrt(data_dim))

    # pad data with 0s if data sample size is not a square number
    if sample_side_size ** 2 < data_dim:
        next_largest_square_num = int(np.ceil(np.sqrt(data_dim))) ** 2
        total_size = pad_to_size or next_largest_square_num
        data = np.concatenate([data, np.zeros((data_size, total_size - data_dim))], axis=-1)
        reconstructed_data = np.concatenate([reconstructed_data, np.zeros((data_size, total_size - data_dim))], axis=-1)
        sample_side_size = int(np.sqrt(total_size))

    reconstructed_data = reconstructed_data.reshape(-1, sample_side_size, sample_side_size)
    data = data.reshape(-1, sample_side_size, sample_side_size)
    combined_data_reconstructions = np.concatenate([data, reconstructed_data], axis=-1)
    # add a separating blank line between the reshaped column of image pairs for better visualisation
    combined_data_reconstructions = np.concatenate([combined_data_reconstructions,
                                                    np.ones((data_size, sample_side_size, 1))], axis=-1)
    samples_per_fig_side = int(np.sqrt(data_size))
    combined_data_reconstructions = combined_data_reconstructions[:samples_per_fig_side ** 2].reshape(
        samples_per_fig_side, samples_per_fig_side, sample_side_size, sample_side_size * 2 + 1)
    combined_data_reconstructions = np.concatenate(np.concatenate(combined_data_reconstructions, axis=1), axis=1)
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_data_reconstructions, cmap='Greys_r')
    if fig_dirpath is not None:
        if not os.path.exists(fig_dirpath):
            os.makedirs(fig_dirpath)
        plt.savefig(os.path.join(fig_dirpath, fig_name))
    else:
        plt.show()


def plot_iterative_background_transfer(data_pairs, transitions, fig_dirpath=None, fig_name=None):
    data_pairs = np.asarray(data_pairs)
    transitions = np.asarray(transitions)

    logger.info("Plotting background transition data.")
    fig_name = fig_name or 'background_transition.png'
    if not fig_name.endswith('.png'):
        fig_name += '.png'

    data_size, data_dim = data_pairs[0].shape
    sample_side_size = int(np.sqrt(data_dim))

    # first axis is the number of combined data samples (normally 2 -- fore- and background)
    data_pairs = data_pairs.reshape(-1, data_size, sample_side_size, sample_side_size).transpose(1, 0, 2, 3)
    data_pairs = np.concatenate(np.concatenate(data_pairs, axis=1), axis=1)
    # first axis is the number of iterations to refine the mixing, so swap with the sample axis to align to data_pairs
    transitions = transitions.reshape(-1, data_size, sample_side_size, sample_side_size).transpose(1, 0, 2, 3)
    transitions = np.concatenate(np.concatenate(transitions, axis=1), axis=1)

    combined_data_transitions = np.concatenate([data_pairs, transitions], axis=-1)

    plt.figure(figsize=(10, 10))
    plt.imshow(combined_data_transitions, cmap='Greys_r')
    if fig_dirpath is not None:
        if not os.path.exists(fig_dirpath):
            os.makedirs(fig_dirpath)
        plt.savefig(os.path.join(fig_dirpath, fig_name))
    else:
        plt.show()
