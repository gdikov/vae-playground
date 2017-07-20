from numpy import save as save_array
from os.path import join as path_join, isfile
from numpy import repeat, asarray, unique, zeros_like
from playground.utils.visualisation import plot_latent_2d, plot_sampled_data, plot_reconstructed_data
from playground.model_trainer import ConjointVAEModelTrainer, ConjointAVBModelTrainer
from playground.utils.datasets import load_conjoint_synthetic, load_mnist
from playground.utils.logger import logger
from keras.backend import clear_session
from playground.utils.data_factory import CustomMNIST
import numpy as np


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))


def run_synthetic_experiment(model='avb', pretrained_model=None):
    logger.info("Starting a conjoint model experiment on the synthetic dataset.")
    data_dims = (8, 8)
    latent_dims = (2, 2, 2)
    data = load_conjoint_synthetic(dims=data_dims)

    if model == 'vae':
        trainer = ConjointVAEModelTrainer(data_dims=data_dims, latent_dims=latent_dims,
                                          experiment_name='synthetic', architecture='synthetic',
                                          overwrite=True, optimiser_params={'lr': 0.001},
                                          pretrained_dir=pretrained_model)
    elif model == 'avb':
        trainer = ConjointAVBModelTrainer(data_dims=data_dims, latent_dims=latent_dims, noise_dim=data_dims[0],
                                          use_adaptive_contrast=False,
                                          optimiser_params={'encdec': {'lr': 1e-3, 'beta_1': 0.5},
                                                            'disc': {'lr': 1e-3, 'beta_1': 0.5}},
                                          overwrite=True,
                                          pretrained_dir=pretrained_model,
                                          architecture='synthetic',
                                          experiment_name='synthetic')
    else:
        raise ValueError("Currently only `avb` and `vae` are supported.")

    model_dir = trainer.run_training(data, batch_size=16, epochs=1000, save_interrupted=True)
    # model_dir = './output/tmp'
    trained_model = trainer.get_model()

    sampling_size = 1000

    latent_vars = trained_model.infer(data, batch_size=4, sampling_size=sampling_size)
    save_array(path_join(model_dir, 'latent_samples.npy'), latent_vars)
    plot_latent_2d(latent_vars[:, -2:], repeat(data[0]['target'], sampling_size),
                   fig_dirpath=model_dir, fig_name='shared.png')
    stop_id = 0
    for i, lat_id in enumerate(latent_dims[:-1]):
        start_id = stop_id
        stop_id += lat_id
        plot_latent_2d(latent_vars[:, start_id:stop_id], repeat(data[0]['tag'], sampling_size),
                       fig_dirpath=model_dir, fig_name='private_{}'.format(i))

    reconstructions = trained_model.reconstruct(data, batch_size=16, sampling_size=1)
    save_array(path_join(model_dir, 'reconstructed_samples.npy'), reconstructions)
    for i, rec in enumerate(reconstructions):
        plot_reconstructed_data(data[i]['data'], rec,
                                fig_dirpath=model_dir,
                                fig_name='reconstructed_{}'.format(i),
                                pad_to_size=16)

    generations = trained_model.generate(n_samples=100, batch_size=100)
    save_array(path_join(model_dir, 'generated_samples.npy'), generations)
    for i, gen in enumerate(generations):
        plot_sampled_data(gen, fig_dirpath=model_dir, fig_name='data_{}'.format(i))

    clear_session()
    return model_dir


def run_mnist_experiment(model='avb', pretrained_model=None, two_backgrounds_per_encoder=True, color_tags=True,
                         small_set=False):
    logger.info("Starting a conjoint model experiment on the MNIST Variations dataset.")
    data_dims = (784, 784)
    latent_dims = (2, 2, 2)
    test_size = 100
    if not two_backgrounds_per_encoder:
        experiment_name = 'mnist_variation'
        data_0 = load_mnist(local_data_path='data/MNIST_Custom_Variations/strippy_horizontal.npz',
                            one_hot=False, binarised=False, background='custom')
        data_1 = load_mnist(local_data_path='data/MNIST_Custom_Variations/strippy_vertical.npz',
                            one_hot=False, binarised=False, background='custom')
    elif two_backgrounds_per_encoder:
        # data sets are generated if not found
        experiment_name = 'mnist_variation_two_backgrounds_per_encoder'
        if small_set:
            setsize = 1000
            local_path_0 = 'data/MNIST_Custom_Variations/strippy_horizontal_and_vertical_small.npz'
            local_path_1 = 'data/MNIST_Custom_Variations/trippy_and_mandelbrot_small.npz'
        else:
            setsize = 50000
            local_path_0 = 'data/MNIST_Custom_Variations/strippy_horizontal_and_vertical.npz'
            local_path_1 = 'data/MNIST_Custom_Variations/trippy_and_mandelbrot.npz'
        if not isfile(local_path_0):
            cmnist = CustomMNIST()
            new_data = cmnist.generate(setsize, [0, 1, 0, 0], orientation='vertical_or_horizontal')
            save_as = 'strippy_horizontal_and_vertical'
            if small_set: save_as += '_small'
            cmnist.save_dataset(new_data, save_as)
        data_0 = load_mnist(local_path_0, one_hot=False, binarised=False, background='custom')
        if not isfile(local_path_1):
            cmnist = CustomMNIST()
            new_data = cmnist.generate(setsize, [0.5, 0, 0.5, 0])
            save_as = 'trippy_and_mandelbrot'
            if small_set: save_as += '_small'
            cmnist.save_dataset(new_data, 'trippy_and_mandelbrot')
        data_1 = load_mnist(local_path_1, one_hot=False, binarised=False, background='custom')

    train_data = ({'data': data_0['data'][:-test_size], 'target': data_0['target'][:-test_size]},
                  {'data': data_1['data'][:-test_size], 'target': data_1['target'][:-test_size]})
    test_data = ({'data': data_0['data'][-test_size:], 'target': data_0['target'][-test_size:]},
                 {'data': data_1['data'][-test_size:], 'target': data_1['target'][-test_size:]})

    if model == 'vae':
        trainer = ConjointVAEModelTrainer(data_dims=data_dims, latent_dims=latent_dims,
                                          experiment_name=experiment_name, architecture='mnist',
                                          overwrite=True, optimiser_params={'lr': 0.001, 'beta_1': 0.5},
                                          pretrained_dir=pretrained_model)
    elif model == 'avb':
        trainer = ConjointAVBModelTrainer(data_dims=data_dims, latent_dims=latent_dims, noise_dim=16,
                                          use_adaptive_contrast=False,
                                          optimiser_params=None,
                                          pretrained_dir=pretrained_model,
                                          architecture='mnist',
                                          experiment_name=experiment_name)
    else:
        raise ValueError("Currently only `avb` and `vae` are supported.")

    model_dir = trainer.run_training(train_data, batch_size=100, epochs=30, save_interrupted=True)
    trained_model = trainer.get_model()

    sampling_size = 1

    latent_vars = trained_model.infer(train_data, batch_size=100, sampling_size=sampling_size)
    save_array(path_join(model_dir, 'latent_samples.npy'), latent_vars)
    plot_latent_2d(latent_vars[:, -2:], repeat(train_data[0]['target'], sampling_size),
                   fig_dirpath=model_dir, fig_name='shared.png')

    stop_id = 0
    data = (data_0, data_1)

    for i, lat_id in enumerate(latent_dims[:-1]):
        start_id = stop_id
        stop_id += lat_id
        if color_tags:
            tags = data[i]['tag'][:-test_size]
            tags_num = zeros_like(train_data[0]['target'], dtype=np.int8)
            tag_dict = dict(enumerate(unique(tags)))
            assert len(tags_num) == len(tags)
            for k, v in tag_dict.items():
                tags_num[tags == v] = k
            plot_latent_2d(latent_vars[:, start_id:stop_id], repeat(tags_num, sampling_size),
                           fig_dirpath=model_dir, fig_name='private_{}'.format(i))
        else:
            plot_latent_2d(latent_vars[:, start_id:stop_id], repeat(train_data[i]['target'], sampling_size),
                           fig_dirpath=model_dir, fig_name='private_{}'.format(i))

    reconstructions = trained_model.reconstruct(test_data, batch_size=100, sampling_size=1)
    save_array(path_join(model_dir, 'reconstructed_samples.npy'), reconstructions)
    for i, rec in enumerate(reconstructions):
        plot_reconstructed_data(test_data[i]['data'], rec,
                                fig_dirpath=model_dir, fig_name='reconstructed_{}'.format(i))

    generations = trained_model.generate(n_samples=100, batch_size=100)
    save_array(path_join(model_dir, 'generated_samples.npy'), generations)
    for i, gen in enumerate(generations):
        plot_sampled_data(gen, fig_dirpath=model_dir, fig_name='data_{}'.format(i))

    clear_session()
    return model_dir


if __name__ == '__main__':
    # Two backgrounds per encoder (strippy horizontal/vertical and trippy/mandelbrot)
    # Datasets (N=50k) are created at 'data/MNIST_Custom_Variations' if they don't exist (may take a while)
    run_mnist_experiment(model='vae', two_backgrounds_per_encoder=True, color_tags=True)
