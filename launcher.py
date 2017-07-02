from numpy import save as save_array
from os.path import join as path_join
from numpy import repeat
from avb.utils.visualisation import plot_latent_2d, plot_sampled_data, plot_reconstructed_data
from avb.model_trainer import AVBModelTrainer, VAEModelTrainer
from avb.utils.datasets import load_npoints
from avb.utils.logger import logger

from keras.backend import clear_session
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))


def run_synthetic_experiment(pretrained_model=None):
    logger.info("Starting a conjoint model experiment on the synthetic dataset.")
    data_dim = 4
    data = load_npoints(n=data_dim)

    train_data, train_labels = data['data'], data['target']

    trainer = VAEModelTrainer(data_dim=data_dim, latent_dim=2, experiment_name='synthetic', overwrite=True,
                              optimiser_params={'lr': 0.001}, pretrained_dir=pretrained_model)

    model_dir = trainer.run_training(train_data, batch_size=400, epochs=200)
    # model_dir = "output/"
    trained_model = trainer.get_model()

    sampling_size = 1000
    augmented_data = repeat(train_data, sampling_size, axis=0)
    augmented_labels = repeat(train_labels, sampling_size, axis=0)

    reconstructions = trained_model.reconstruct(train_data, batch_size=1000, sampling_size=sampling_size)
    save_array(path_join(model_dir, 'reconstructed_samples.npy'), reconstructions)
    plot_reconstructed_data(augmented_data[:100], reconstructions[:100], fig_dirpath=model_dir)

    latent_vars = trained_model.infer(train_data, batch_size=1000, sampling_size=sampling_size)
    save_array(path_join(model_dir, 'latent_samples.npy'), latent_vars)
    plot_latent_2d(latent_vars, augmented_labels, fig_dirpath=model_dir)

    generations = trained_model.generate(n_samples=100, batch_size=100)
    save_array(path_join(model_dir, 'generated_samples.npy'), generations)
    plot_sampled_data(generations, fig_dirpath=model_dir)

    clear_session()
    return model_dir


if __name__ == '__main__':
    run_synthetic_experiment('avb+ac', pretrained_model='output/avb_with_ac/synthetic/final')
