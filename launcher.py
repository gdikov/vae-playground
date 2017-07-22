from numpy import save as save_array
from os.path import join as path_join
from numpy import repeat
from playground.utils.visualisation import plot_latent_2d, plot_sampled_data, plot_reconstructed_data
from playground.model_trainer import ConjointVAEModelTrainer, ConjointAVBModelTrainer
from playground.utils.datasets import load_conjoint_synthetic, load_mnist
from playground.utils.logger import logger
from keras.backend import clear_session

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))


def run_synthetic_experiment(model='avb', pretrained_model=None, noise_mode='concat'):
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
                                          optimiser_params={'encdec': {'lr': 3e-4, 'beta_1': 0.5},
                                                            'disc': {'lr': 3e-4, 'beta_1': 0.5}},
                                          overwrite=True,
                                          pretrained_dir=pretrained_model,
                                          architecture='synthetic',
                                          experiment_name='synthetic',
                                          noise_mode=noise_mode)
    else:
        raise ValueError("Currently only `avb` and `vae` are supported.")

    model_dir = trainer.run_training(data, batch_size=16, epochs=100, save_interrupted=True)
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


def run_mnist_experiment(model='avb', pretrained_model=None):
    logger.info("Starting a conjoint model experiment on the MNIST Variations dataset.")
    data_dims = (784, 784)
    latent_dims = (2, 2, 2)
    # data_0 = load_mnist(one_hot=False, binarised=False, background=None, rotated=False)
    # data_1 = load_mnist(one_hot=False, binarised=False, background='image', rotated=False)
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
                                          experiment_name='mnist_variations', architecture='mnist',
                                          overwrite=True, optimiser_params={'lr': 0.0007, 'beta_1': 0.5},
                                          pretrained_dir=pretrained_model)
    elif model == 'avb':
        trainer = ConjointAVBModelTrainer(data_dims=data_dims, latent_dims=latent_dims, noise_dim=64,
                                          use_adaptive_contrast=False,
                                          optimiser_params={'encdec': {'lr': 1e-4, 'beta_1': 0.5},
                                                            'disc': {'lr': 2e-4, 'beta_1': 0.5}},
                                          overwrite=True,
                                          pretrained_dir=pretrained_model,
                                          architecture='mnist',
                                          experiment_name='mnist_variations')
    else:
        raise ValueError("Currently only `avb` and `vae` are supported.")

    model_dir = trainer.run_training(train_data, batch_size=100, epochs=1000, save_interrupted=True)
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


if __name__ == '__main__':
    run_synthetic_experiment(model='avb', noise_mode='product')
    # run_mnist_experiment(model='avb')
