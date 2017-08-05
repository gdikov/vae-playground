from __future__ import division

from numpy import save as save_array
from os.path import join as path_join
from numpy import repeat, asarray, concatenate, copy, zeros, ones
from numpy.random import randint
from playground.utils.visualisation import plot_latent_2d, plot_sampled_data, plot_reconstructed_data
from playground.model_trainer import ConjointVAEModelTrainer, ConjointAVBModelTrainer
from playground.utils.datasets import load_conjoint_synthetic, load_mnist
from playground.utils.data_factory import CustomMNIST, PatternFactory
from playground.utils.logger import logger
from keras.backend import clear_session
import matplotlib.pyplot as plt


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


def change_background_save_latent(model='avb', pretrained_model=None, **kwargs):
    """
    Experiment to change backgrounds by means of saving the shared latent space and updating the private latent space 
    with new image that has different background.
    Explanation of plotted digits:
    In each row, the left three digits correspond to the first encoder, the right three to the second.
    From left to right: Original, reconstruction, changed background.
    
    Args:
        model: avb or vae 
        pretrained_model: dictionary with h5 file in it 
        **kwargs: optionally specify backgrounds of data sets

    """
    logger.info("Creating plot of digits with changed background, by saving shared latent space")
    data_dims = (784, 784)
    latent_dims = (2, 2, 2)

    bg_same_target = kwargs.get('bg_same_target', False)
    datasets = kwargs.get('dataset_pairs', [('horizontal', 'vertical'), ('trippy', 'black')])
    cmnist = CustomMNIST()
    data = []
    for dataset in datasets:
        custom_data = cmnist.load_dataset('_'.join(dataset), generate_if_none=True)
        data.append(custom_data)
    data = tuple([{k: d[k][:] for k in d.keys()} for d in data])

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

    model_dir = trainer.run_training(data, batch_size=100, epochs=0,
                                     save_interrupted=True,
                                     grouping_mode='by_pairs')
    # model_dir = 'output/tmp'
    trained_model = trainer.get_model()

    max_index = 49999
    data_0, data_1 = data
    id_num_list = randint(max_index, size=12)
    # id_num is ID of image used for digit, id_bg of image used for background
    for id_num in id_num_list:
        # Data set with same digit in front of 2 different backgrounds
        num_data = (
            {'data': asarray(data_0['data'][id_num]).reshape((1, 784)),
             'target': asarray(data_0['target'][id_num])},
            {'data': asarray(data_1['data'][id_num]).reshape((1, 784)),
             'target': asarray(data_1['target'][id_num])})

        # Find a picture with a different background for first encoder
        # If the "bg_same_target"-flag is True, the background has the same digit written differently
        while True:
            id_bg_0 = randint(max_index)
            # Cannot have same background as image of digit
            tags_same = data_0['tag'][id_num] == data_0['tag'][id_bg_0]
            target_same = data_0['target'][id_num] == data_0['target'][id_bg_0]
            if not tags_same:
                if not bg_same_target and not target_same:
                    break
                if bg_same_target and target_same:
                    break

        # Same thing for second encoder
        while True:
            id_bg_1 = randint(max_index)
            tags_same = data_1['tag'][id_num] == data_1['tag'][id_bg_1]
            target_same = data_1['target'][id_num] == data_1['target'][id_bg_1]
            if not tags_same:
                if not bg_same_target and not target_same:
                    break
                if bg_same_target and target_same:
                    break

        # Create data set from found pictures, random digits with other backgrounds than num_data
        bg_data = (
            {'data': asarray(data_0['data'][id_bg_0]).reshape((1, 784)),
             'target': asarray(data_0['target'][id_bg_0])},
            {'data': asarray(data_1['data'][id_bg_1]).reshape((1, 784)),
             'target': asarray(data_1['target'][id_bg_1])})

        # Infer latent variables for num_data and bg_data set
        latent_num = trained_model.infer(num_data, batch_size=1, sampling_size=1)
        latent_bg = trained_model.infer(bg_data, batch_size=1, sampling_size=1)

        # Mixed has shared latent space of digits and private latent spaced of backgrounds
        latent_mixed = copy(latent_bg)
        for i in (4, 5):
            latent_mixed[0][i] = latent_num[0][i]

        # Reconstruct from mixed and the two pure shared latent spaces
        reconstruction_original = trained_model.generate(1, 1, latent_samples=latent_num)
        reconstruction_mixed = trained_model.generate(1, 1, latent_samples=latent_mixed)

        # Left encoder
        original_num_0 = num_data[0]['data'].reshape((28, 28))
        reconst_0 = reconstruction_original[0, 0].reshape((28, 28))
        new_bg_0 = reconstruction_mixed[0, 0].reshape((28, 28))

        # Right encoder
        original_num_1 = num_data[1]['data'].reshape((28, 28))
        reconst_1 = reconstruction_original[1, 0].reshape((28, 28))
        new_bg_1 = reconstruction_mixed[1, 0].reshape((28, 28))

        row = concatenate(
            (original_num_0, reconst_0, new_bg_0, ones((28, 5)),
             original_num_1, reconst_1, new_bg_1),
            axis=1)
        try:
            img = concatenate((img, row), axis=0)
        except NameError:
            img = row

    plt.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    filename = "ChangeBackground_SavedLatent"
    if bg_same_target:
        filename = '_'.join((filename, 'BgSameTarget'))
    save_path = path_join('output', 'change_background', filename)
    plt.savefig(save_path)

    clear_session()


def change_background_generate_digit(model='vae', pretrained_model=None, **kwargs):
    logger.info("Creating plot of digits with changed background, by generating digit on empty background")
    data_dims = (784, 784)
    latent_dims = (2, 2, 2)
    datasets = kwargs.get('dataset_pairs', [('horizontal', 'vertical'), ('trippy', 'black')])
    bg = kwargs.get('bg', 'black')
    cmnist = CustomMNIST()
    data = []
    for dataset in datasets:
        custom_data = cmnist.load_dataset('_'.join(dataset), generate_if_none=True)
        data.append(custom_data)
    data = tuple([{k: d[k][:] for k in d.keys()} for d in data])

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

    model_dir = trainer.run_training(data, batch_size=100, epochs=0,
                                     save_interrupted=True,
                                     grouping_mode='by_pairs')
    # model_dir = 'output/tmp'
    trained_model = trainer.get_model()

    max_index = 49999
    iterations = 14
    data_0, data_1 = data
    id_num_list = randint(max_index, size = iterations+1)
    # id_num is ID of image used for digit, id_bg of image used for background
    for id_num in id_num_list:
        if bg == 'black':
            new_img = zeros((1, 784))
        elif bg == 'trippy':
            new_img = PatternFactory().render_trippy((28,28),as_gray=True)['image'].reshape((1,784))
        org_img = asarray(data_0['data'][id_num]).reshape((1, 784))
        digits = [asarray(data_0['data'][id_num]).reshape((28,28))]
        target = asarray(data_0['target'][id_num])

        for iter in range(iterations):
            # Data set with original image and new image, at the beginning new image is just empty background
            input_data = ({'data': org_img, 'target': target}, {'data': new_img, 'target': target})

            # Infer latent variables
            latent = trained_model.infer(input_data, batch_size=1, sampling_size=1)

            # Reconstruct image
            reconstruction = trained_model.generate(1, 1, latent_samples=latent)
            digits.append(reconstruction[1, 0].reshape((28, 28)))
            new_img = reconstruction[1, 0].reshape((1, 784))

        row = concatenate(digits, axis=1)
        try:
            img = concatenate((img, row), axis=0)
        except NameError:
            img = row

    plt.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    filename = "ChangeBackground_GenerateDigit"
    filename += ('_' + bg)
    save_path = path_join('output', 'change_background', filename)
    plt.savefig(save_path)

    clear_session()
