from numpy import save as save_array
from os.path import join as path_join, isfile
from numpy import repeat, asarray, unique, zeros_like, int8, concatenate, copy
from numpy.random import randint
from playground.utils.visualisation import plot_latent_2d, plot_sampled_data, plot_reconstructed_data
from playground.model_trainer import ConjointVAEModelTrainer, ConjointAVBModelTrainer
from playground.utils.datasets import load_conjoint_synthetic, load_mnist
from playground.utils.logger import logger
from keras.backend import clear_session
from playground.utils.data_factory import CustomMNIST
from matplotlib import pyplot as plt


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
                         small_set=False, change_background=False, epochs=100):
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

    if change_background:
        epochs = 0

    model_dir = trainer.run_training(train_data, batch_size=100, epochs=epochs, save_interrupted=True)
    trained_model = trainer.get_model()

    if change_background:
        # Create pairs of image IDs (randomly or not)
        random = True
        if random:
            test_img_ids = asarray([randint(49999, size=2) for _ in range(8)])
        else:
            test_img_ids = ([[49689, 10742],
                             [38467, 919],
                             [13927, 34140],
                             [2583, 28232],
                             [27354, 45960]])

        for id_0, id_1 in test_img_ids:
            # Create example data set with N=1 for each pair of images
            example_data = (
                {'data': asarray(data_0['data'][id_0]).reshape((1, 784)), 'target': asarray(data_0['target'][id_0])},
                {'data': asarray(data_1['data'][id_1]).reshape((1, 784)), 'target': asarray(data_1['target'][id_1])})

            # Infer latent variables for three cases: IMG0-IMG0, IMG1-IMG1, IMG0-IMG1
            latent_img0_twice = trained_model.infer((example_data[0], example_data[0]), batch_size=1, sampling_size=1)
            latent_img1_twice = trained_model.infer((example_data[1], example_data[1]), batch_size=1, sampling_size=1)
            latent_mixed = trained_model.infer(example_data, batch_size=1, sampling_size=1)

            # Copy latent variables and replace mixed shared latent space with pure shared latent space
            # Pure latent space means both input images were identical
            latent_num0_bg1 = copy(latent_mixed)
            latent_num1_bg0 = copy(latent_mixed)
            for i in (4, 5):
                latent_num0_bg1[0][i] = latent_img0_twice[0][i]
                latent_num1_bg0[0][i] = latent_img1_twice[0][i]

            # Reconstruct from mixed and the two pure shared latent spaces
            reconstruction_original = trained_model.generate(1, 1, latent_samples=latent_mixed)
            reconstruction_num0_bg1 = trained_model.generate(1, 1, latent_samples=latent_num0_bg1)
            reconstruction_num1_bg0 = trained_model.generate(1, 1, latent_samples=latent_num1_bg0)

            # Create one Row of Visualizations (See description below)
            img0 = example_data[0]['data'].reshape((28, 28))
            reconst0 = reconstruction_original[0, 0].reshape((28, 28))
            num0_bg0 = reconstruction_num0_bg1[0, 0].reshape((28, 28))
            num0_bg1 = reconstruction_num0_bg1[1, 0].reshape((28, 28))

            # Same thing for other image
            img1 = example_data[1]['data'].reshape((28, 28))
            reconst1 = reconstruction_original[1, 0].reshape((28, 28))
            num1_bg1 = reconstruction_num1_bg0[1, 0].reshape((28, 28))
            num1_bg0 = reconstruction_num1_bg0[0, 0].reshape((28, 28))
            row = concatenate((img0, reconst0, num0_bg0, num0_bg1, img1, reconst1, num1_bg1, num1_bg0), axis=1)
            try:
                img = concatenate((img, row), axis=0)
            except NameError:
                img = row

        plt.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        # Description of this plot:
        # The left four images in a row correspond to a picture form dataset0, the right four from dataset1
        # The pictures from left to right:
        # original,
        # reconstruction from mixed shared latent
        # reconstruction from pure shared latent with own background,
        # reconstruction from pure shared latent space with other background
        plt.savefig('output/conjoint_gaussian_vae/reconstructed_changed_background.png')

    else:
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
                tags_num = zeros_like(train_data[0]['target'], dtype=int8)
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
    pretrained_dir = "output/conjoint_gaussian_vae/vae_mnist_2ds_per_encoder_272epochs_fullsize"
    run_mnist_experiment(model='vae', two_backgrounds_per_encoder=True, color_tags=True,
                         pretrained_model=pretrained_dir, change_background=True)
