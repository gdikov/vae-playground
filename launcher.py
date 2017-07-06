from numpy import save as save_array
from os.path import join as path_join
from numpy import repeat
from avb.utils.visualisation import plot_latent_2d, plot_sampled_data, plot_reconstructed_data
from avb.model_trainer import ConjointVAEModelTrainer
from avb.utils.datasets import load_npoints
from avb.utils.logger import logger

from keras.backend import clear_session
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))


def run_synthetic_experiment():
    logger.info("Starting a conjoint model experiment on the synthetic dataset.")
    data_dims = (4, 4)
    latent_dims = (2, 2, 2)
    data = load_npoints(n=data_dims[0])

    trainer = ConjointVAEModelTrainer(data_dims=data_dims, latent_dims=latent_dims,
                                      experiment_name='synthetic', overwrite=True,
                                      optimiser_params={'lr': 0.001})

    model_dir = trainer.run_training((data, data), batch_size=400, epochs=200)
    trained_model = trainer.get_model()

    sampling_size = 400

    latent_vars = trained_model.infer((data, data), batch_size=400, sampling_size=sampling_size)
    save_array(path_join(model_dir, 'latent_samples.npy'), latent_vars)
    plot_latent_2d(latent_vars[:, -2:], repeat(data['target'], sampling_size),
                   fig_dirpath=model_dir, fig_name='shared.png')
    plot_latent_2d(latent_vars[:, :2], repeat(data['target'], sampling_size),
                   fig_dirpath=model_dir, fig_name='private_1.png')
    plot_latent_2d(latent_vars[:, 2:4], repeat(data['target'], sampling_size),
                   fig_dirpath=model_dir, fig_name='private_2.png')

    clear_session()
    return model_dir


if __name__ == '__main__':
    run_synthetic_experiment()
