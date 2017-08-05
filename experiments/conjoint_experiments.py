from __future__ import division

from numpy import save as save_array
from numpy.random import choice as rand_choice
from os.path import join as path_join
from numpy import repeat
from playground.utils.visualisation import *
from playground.model_trainer import ConjointVAEModelTrainer, ConjointAVBModelTrainer
from playground.utils.datasets import load_conjoint_synthetic, load_mnist
from playground.utils.data_factory import CustomMNIST
from playground.utils.logger import logger
from keras.backend import clear_session


def synthetic(model='avb', pretrained_model=None, noise_mode='product'):
    logger.info("Starting a conjoint model experiment on the synthetic dataset.")
    data_dims = (8, 8)
    latent_dims = (2, 2, 2)
    data = load_conjoint_synthetic(dims=data_dims)

    if model == 'vae':
        trainer = ConjointVAEModelTrainer(data_dims=data_dims, latent_dims=latent_dims,
                                          experiment_name='synthetic', architecture='synthetic',
                                          overwrite=True, save_best=True, optimiser_params={'lr': 0.001},
                                          pretrained_dir=pretrained_model)
    elif model == 'avb':
        trainer = ConjointAVBModelTrainer(data_dims=data_dims, latent_dims=latent_dims, noise_dim=data_dims[0],
                                          use_adaptive_contrast=False,
                                          optimiser_params={'encdec': {'lr': 1e-3, 'beta_1': 0.5},
                                                            'disc': {'lr': 1e-3, 'beta_1': 0.5}},
                                          schedule={'iter_disc': 3, 'iter_encdec': 1},
                                          overwrite=True,
                                          save_best=True,
                                          pretrained_dir=pretrained_model,
                                          architecture='synthetic',
                                          experiment_name='synthetic',
                                          noise_mode=noise_mode)
    else:
        raise ValueError("{} model not supported. Choose between `avb` and `vae`.".format(model))

    model_dir = trainer.run_training(data, batch_size=4, epochs=10000,
                                     grouping_mode='by_targets',
                                     save_interrupted=True,
                                     validation_data=data,
                                     validation_frequency=500,
                                     validation_sampling_size=5)

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


def mnist_variations(n_datasets=1, model='avb', pretrained_model=None):
    if n_datasets == 1:
        mnist_variations_one(model=model, pretrained_model=pretrained_model)
    elif n_datasets == 2:
        mnist_variations_two(model=model, pretrained_model=pretrained_model)
    else:
        raise NotImplementedError("Currently only 1 and 2 number of datasets per encoder are implemented.")


def mnist_variations_one(model='avb', pretrained_model=None):
    logger.info("Starting a conjoint model experiment on the MNIST Variations "
                "dataset (one dataset per autoencoder).")
    data_dims = (784, 784)
    latent_dims = (2, 2, 2)
    data_0 = load_mnist(local_data_path='data/MNIST_Custom_Variations/strippy_horizontal.npz',
                        one_hot=False, binarised=False, background='custom')
    data_1 = load_mnist(local_data_path='data/MNIST_Custom_Variations/strippy_checker.npz',
                        one_hot=False, binarised=False, background='custom')
    train_data = ({'data': data_0['data'][:-100], 'target': data_0['target'][:-100], 'tag': data_0['tag'][:-100]},
                  {'data': data_1['data'][:-100], 'target': data_1['target'][:-100], 'tag': data_1['tag'][:-100]})
    test_data = ({'data': data_0['data'][-100:], 'target': data_0['target'][-100:], 'tag': data_0['tag'][-100:]},
                 {'data': data_1['data'][-100:], 'target': data_1['target'][-100:], 'tag': data_1['tag'][-100:]})

    if model == 'vae':
        trainer = ConjointVAEModelTrainer(data_dims=data_dims, latent_dims=latent_dims,
                                          experiment_name='mnist_variations_one', architecture='mnist',
                                          overwrite=True, save_best=True,
                                          optimiser_params={'lr': 0.0007, 'beta_1': 0.5},
                                          pretrained_dir=pretrained_model)
    elif model == 'avb':
        trainer = ConjointAVBModelTrainer(data_dims=data_dims, latent_dims=latent_dims, noise_dim=64,
                                          use_adaptive_contrast=False,
                                          optimiser_params={'encdec': {'lr': 1e-4, 'beta_1': 0.5},
                                                            'disc': {'lr': 2e-4, 'beta_1': 0.5}},
                                          schedule={'iter_disc': 1, 'iter_encdec': 1},
                                          overwrite=True, save_best=True,
                                          pretrained_dir=pretrained_model,
                                          architecture='mnist',
                                          experiment_name='mnist_variations_one')
    else:
        raise ValueError("Currently only `avb` and `vae` are supported.")

    model_dir = trainer.run_training(train_data, batch_size=100, epochs=1000,
                                     grouping_mode='by_pairs',
                                     save_interrupted=True,
                                     validation_data=test_data,
                                     validation_frequency=20,
                                     validation_sampling_size=5)
    # model_dir = 'output/tmp'
    trained_model = trainer.get_model()

    sampling_size = 1

    latent_vars = trained_model.infer(train_data, batch_size=100, sampling_size=sampling_size)
    save_array(path_join(model_dir, 'latent_samples.npy'), latent_vars)
    plot_latent_2d(latent_vars[:, -2:], repeat(train_data[0]['target'], sampling_size),
                   fig_dirpath=model_dir, fig_name='shared.png')
    stop_id = 0
    for i, lat_id in enumerate(latent_dims[:-1]):
        start_id = stop_id
        stop_id += lat_id
        plot_latent_2d(latent_vars[:, start_id:stop_id], repeat(train_data[0]['tag'], sampling_size),
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


def mnist_variations_two(model='avb', pretrained_model=None, **kwargs):
    logger.info("Starting a conjoint model experiment on the MNIST Variations "
                "dataset (two datasets per autoencoder).")
    data_dims = (784, 784)
    latent_dims = (2, 2, 2)

    datasets = kwargs.get('dataset_pairs', [('horizontal', 'trippy'), ('vertical', 'black')])
    cmnist = CustomMNIST()
    data = []
    for dataset in datasets:
        custom_data = cmnist.load_dataset('_'.join(dataset), generate_if_none=True)
        data.append(custom_data)
    train_data = tuple([{k: d[k][:-100] for k in d.keys()} for d in data])
    test_data = tuple([{k: d[k][-100:] for k in d.keys()} for d in data])

    if model == 'vae':
        trainer = ConjointVAEModelTrainer(data_dims=data_dims, latent_dims=latent_dims,
                                          experiment_name='mnist_variations_two', architecture='mnist',
                                          overwrite=True, save_best=True,
                                          optimiser_params={'lr': 0.0007, 'beta_1': 0.5},
                                          pretrained_dir=pretrained_model)
    elif model == 'avb':
        trainer = ConjointAVBModelTrainer(data_dims=data_dims, latent_dims=latent_dims, noise_dim=64,
                                          use_adaptive_contrast=False,
                                          noise_mode='add',
                                          optimiser_params={'encdec': {'lr': 0.001, 'beta_1': 0.5},
                                                            'disc': {'lr': 0.001, 'beta_1': 0.5}},
                                          schedule={'iter_disc': 3, 'iter_encdec': 1},
                                          overwrite=True, save_best=True,
                                          pretrained_dir=pretrained_model,
                                          architecture='mnist',
                                          experiment_name='mnist_variations_two')
    else:
        raise ValueError("Currently only `avb` and `vae` are supported.")

    model_dir = trainer.run_training(train_data, batch_size=100, epochs=1000,
                                     save_interrupted=True,
                                     grouping_mode='by_pairs',
                                     validation_data=test_data,
                                     validation_frequency=1,
                                     validation_sampling_size=5)
    # model_dir = 'output/tmp'
    trained_model = trainer.get_model()

    sampling_size = 1

    latent_vars = trained_model.infer(train_data, batch_size=100, sampling_size=sampling_size)
    save_array(path_join(model_dir, 'latent_samples.npy'), latent_vars)
    plot_latent_2d(latent_vars[:, -2:], repeat(train_data[0]['target'], sampling_size),
                   fig_dirpath=model_dir, fig_name='shared.png')
    stop_id = 0
    for i, lat_id in enumerate(latent_dims[:-1]):
        start_id = stop_id
        stop_id += lat_id
        plot_latent_2d(latent_vars[:, start_id:stop_id], repeat(train_data[i]['tag'], sampling_size),
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


def mnist_background_transfer(model='avb', pretrained_model=None, data=None, **kwargs):
    if pretrained_model is None:
        raise ValueError("Background transfer requires a pre-trained model. "
                         "Provide path to the {} model.".format(model))
        # or maybe a default model should be loaded?
    if data is None:
        datasets = kwargs.get('dataset_pairs', [('horizontal', 'trippy'), ('vertical', 'black')])
        logger.info("Loading random MNIST samples and backgrounds assuming dataset pairs are {}".format(datasets))
        n_samples = kwargs.get('n_samples', 10)

        cmnist = CustomMNIST()
        digit_data = cmnist.load_dataset('_'.join(datasets[0]), generate_if_none=True)

        background_data, background_tags = [], []
        for s in rand_choice(datasets[1], size=n_samples, replace=True):
            background = cmnist.generate_style(style=s)
            background_data.append(background['image'].ravel())
            background_tags.append(datasets[1].index(background['tag']))

        data_size = digit_data['target'].size
        ids = rand_choice(data_size, size=n_samples, replace=n_samples > data_size)
        data = tuple([{k: digit_data[k][ids] for k in digit_data.keys()},
                      {'data': background_data, 'target': digit_data['target'][ids], 'tag': background_tags}])
    else:
        n_samples = len(data[0]['target'])

    logger.info("Starting a background transfer experiment using a {} model.".format(model))

    data_dims = (784, 784)
    latent_dims = (2, 2, 2)

    if model == 'vae':
        trainer = ConjointVAEModelTrainer(data_dims=data_dims, latent_dims=latent_dims,
                                          experiment_name='mnist_background_transfer', architecture='mnist',
                                          pretrained_dir=pretrained_model)
    elif model == 'avb':
        trainer = ConjointAVBModelTrainer(data_dims=data_dims, latent_dims=latent_dims, noise_dim=64,
                                          noise_mode='add',
                                          pretrained_dir=pretrained_model,
                                          architecture='mnist',
                                          experiment_name='mnist_background_transfer')
    else:
        raise ValueError("{} model not supported. Choose between `avb` and `vae`.".format(model))

    model_dir = trainer.get_experiment_dir()
    trained_model = trainer.get_model()

    n_iter = kwargs.get('n_iter', 5)
    original_data = data
    transitions = []
    for i in range(n_iter):
        reconstructions = trained_model.reconstruct(data, batch_size=min(n_samples, 100), sampling_size=1)
        data = tuple([data[0], {'data': reconstructions[1], 'target': data[1]['target'], 'tag': data[1]['tag']}])
        transitions.append(reconstructions[1])
    save_array(path_join(model_dir, 'background_transitions.npy'), transitions)
    plot_iterative_background_transfer([d['data'] for d in original_data], transitions,
                                       fig_dirpath=model_dir,
                                       fig_name='background_transfer.png')

    clear_session()
    return model_dir
