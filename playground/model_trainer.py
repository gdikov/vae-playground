from __future__ import absolute_import

import logging
import shutil
import os
import matplotlib.pyplot as plt

from numpy import savez
from datetime import datetime

from playground.models import *
from playground.utils.config import load_config

config = load_config('global_config.yaml')
logger = logging.getLogger(__name__)


class ModelTrainer(object):
    """
    ModelTrainer is a wrapper around the AVBModel and VAEModel to train it, log and store the resulting models.
    """
    def __init__(self, model, experiment_name, overwrite=True, save_best=False):
        """
        Args:
            model: the model object to be trained  
            experiment_name: str, the name of the experiment/training used for logging purposes
            overwrite: bool, whether the trained model should overwrite existing one with the same experiment_name
            save_best: bool, whether to save the model performing best on validation data
        """
        self.model = model
        self.overwrite = overwrite
        self.experiment_name = experiment_name
        self.model_name = model.name
        if self.overwrite:
            self.experiment_dir = os.path.join(config['output_dir'], self.model_name, self.experiment_name)
        else:
            self.experiment_dir = os.path.join(config['output_dir'], self.model_name,
                                               self.experiment_name + '_{}'.format(datetime.now().isoformat()))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.save_best = save_best

    def get_model(self):
        """
        Return the model which can be used for evaluation. 
        
        Returns:
            The model instance. 
        """
        return self.model

    def get_experiment_dir(self):
        """
        Return the model and experiment specific directory.

        Returns:
            A string with the relative experiment path.
        """
        return self.experiment_dir

    def _plot_loss(self, loss_history):
        """
        Plot the graph of the loss during training.

        Args:
            loss_history: dict, loss history object for each of the (sub)models.

        Returns:
            In-place method.
        """
        for fname, loss in loss_history.items():
            plt.plot(loss)
            plt.savefig(os.path.join(self.experiment_dir, fname + '.png'))
            plt.gcf().clear()

    def cb_checkpoint_best_model(self):
        """
        Callback to save the best-so-far model in a dedicated folder.

        Returns:
            In-place method.
        """
        best_model_dir = os.path.join(self.experiment_dir, 'best')
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        logger.debug("Checkpointing model at {}".format(best_model_dir))
        self.model.save(best_model_dir)

    @staticmethod
    def prepare_training():
        """
        Prepare dirs and info for training, clean old cache.
        
        Returns:
            Formatted training start timestamp.
        """
        if os.path.exists(config['temp_dir']):
            shutil.rmtree(config['temp_dir'])
        os.makedirs(config['temp_dir'])
        training_starttime = datetime.now().isoformat()
        return training_starttime

    def finalise_training(self, training_starttime, loss_history=None, meta_info=None):
        """
        Clean up and store best model after training.
        
        Args:
            training_starttime: str, the formatted starting timestamp
            loss_history: dict, of the loss history for each model loss layer
            meta_info: dict, additional training related information such as pre-trained model initialisation (if used),
                optimiser parameters, etc.
            
        Returns:
            In-place method.
        """
        try:
            # save some meta info related to the training and experiment:
            with open(os.path.join(self.experiment_dir, 'meta.txt'), 'w') as f:
                f.write("Training on {} started on {} and finished on {}\n".format(self.experiment_name,
                                                                                   training_starttime,
                                                                                   datetime.now().isoformat()))
                meta_info = meta_info or {}
                f.write("Model initialisation from: {}\n".format(meta_info.get('pretrained_model', 'random state')))
                f.write("Optimiser parameters: {}\n".format(meta_info.get('opt_params', 'default (see model module)')))
            if loss_history is not None:
                savez(os.path.join(self.experiment_dir, 'loss.npz'), **loss_history)
                self._plot_loss(loss_history)
        except IOError:
            logger.error("Saving train history and other meta-information failed.")

    def run_training(self, data, batch_size=32, epochs=1, save_interrupted=False, **kwargs):
        """
        Run training of the model on training data.
        
        Args:
            data: ndarray, the data array of shape (N, data_dim) 
            batch_size: int, the number of samples for one pass
            epochs: int, the number of whole data iterations for training
            save_interrupted: bool, whether the model should be dumped on KeyboardInterrup signal

        Returns:
            The folder name of trained the model.
        """
        training_starttime = self.prepare_training()
        loss_history = None
        try:
            loss_history = self.fit_model(data, batch_size, epochs,
                                          checkpoint_callback=self.cb_checkpoint_best_model,
                                          **kwargs)
            end_model_dir = os.path.join(self.experiment_dir, 'final')
            self.model.save(end_model_dir)
        except KeyboardInterrupt:
            if save_interrupted:
                interrupted_dir = os.path.join(self.experiment_dir,
                                               'interrupted_{}'.format(datetime.now().isoformat()))
                self.model.save(interrupted_dir)
                logger.warning("Training has been interrupted and the model "
                               "has been dumped in {}.".format(interrupted_dir))
            else:
                logger.warning("Training has been interrupted.")
        finally:
            self.finalise_training(training_starttime, loss_history)
            return self.experiment_dir

    def fit_model(self, data, batch_size, epochs, **kwargs):
        """
        Fit particular model to the training data. Should be implemented by each model separately.
        
        Args:
            data: ndarray, the training data array (or tuple of such)
            batch_size: int, the number of samples for one iteration
            epochs: int, number of whole data iterations per training

        Keyword Args:
            validation_data: ndarray, validation data array (or tuple of such)
            validation_frequency: int, after how many epoch the model should be validated

        Returns:
            Model history dict object with the training losses.
        """
        return None


class AVBModelTrainer(ModelTrainer):
    """
    ModelTrainer class for the AVBModel.
    """
    def __init__(self, data_dim, latent_dim, noise_dim, experiment_name, architecture,
                 schedule=None, pretrained_dir=None, overwrite=True, save_best=True,
                 use_adaptive_contrast=False, noise_basis_dim=None, optimiser_params=None):
        """
        Args:
            data_dim: int, flattened data dimensionality 
            latent_dim: int, flattened latent dimensionality
            noise_dim: int, flattened noise dimensionality
            experiment_name: str, name of the training/experiment for logging purposes
            architecture: str, name of the network architecture to be used
            schedule: dict, schedule of training discriminator and encoder-decoder networks
            overwrite: bool, whether to overwrite the existing trained model with the same experiment_name
            save_best: bool, whether to save the best performing model on the validation set (if provided later)
            use_adaptive_contrast: bool, whether to train according to the Adaptive Contrast algorithm
            noise_basis_dim: int, the dimensionality of the noise basis vectors if AC is used.
            optimiser_params: dict, parameters for the optimiser
            pretrained_dir: str, directory from which pre-trained models (hdf5 files) can be loaded
        """
        avb_model = AdversarialVariationalBayes(data_dim=data_dim, latent_dim=latent_dim, noise_dim=noise_dim,
                                                noise_basis_dim=noise_basis_dim,
                                                use_adaptive_contrast=use_adaptive_contrast,
                                                optimiser_params=optimiser_params,
                                                resume_from=pretrained_dir,
                                                experiment_architecture=architecture)
        self.schedule = schedule or {'iter_disc': 1, 'iter_encdec': 1}
        super(AVBModelTrainer, self).__init__(model=avb_model, experiment_name=experiment_name,
                                              overwrite=overwrite, save_best=save_best)

    def fit_model(self, data, batch_size, epochs, **kwargs):
        """
        Fit the AVBModel to the training data.
        
        Args:
            data: ndarray, training data
            batch_size: int, batch size
            epochs: int, number of epochs

        Returns:
            A loss history dict with discriminator and encoder-decoder losses.
        """
        loss_hisotry = self.model.fit(data, batch_size, epochs=epochs,
                                      discriminator_repetitions=self.schedule['iter_disc'],
                                      adaptive_contrast_sampling_steps=10, **kwargs)
        return loss_hisotry


class VAEModelTrainer(ModelTrainer):
    """
    ModelTrainer class for the GaussianVariationalAutoencoder (as per [TODO: add citation to Kingma, Welling]).
    """

    def __init__(self, data_dim, latent_dim, experiment_name, architecture, overwrite=True,
                 save_best=True, optimiser_params=None, pretrained_dir=None):
        """
        Args:
            data_dim: int, flattened data dimensionality 
            latent_dim: int, flattened latent dimensionality
            experiment_name: str, name of the training/experiment for logging purposes
            architecture: str, name of the network architecture to be used
            overwrite: bool, whether to overwrite the existing trained model with the same experiment_name
            save_best: bool, whether to save the best performing model on the validation set (if provided later)
            optimiser_params: dict, parameters for the optimiser
            pretrained_dir: str, optional path to the pre-trained model directory with the hdf5 and json files
        """
        vae = GaussianVariationalAutoencoder(data_dim=data_dim, latent_dim=latent_dim,
                                             experiment_architecture=architecture,
                                             optimiser_params=optimiser_params,
                                             resume_from=pretrained_dir)
        super(VAEModelTrainer, self).__init__(model=vae, experiment_name=experiment_name,
                                              overwrite=overwrite, save_best=save_best)

    def fit_model(self, data, batch_size, epochs, **kwargs):
        """
        Fit the GaussianVAE to the training data.

        Args:
            data: ndarray, training data
            batch_size: int, batch size
            epochs: int, number of epochs

        Returns:
            A loss history dict with the encoder-decoder loss.
        """
        loss_hisotry = self.model.fit(data, batch_size, epochs=epochs, **kwargs)
        return loss_hisotry


class ConjointVAEModelTrainer(ModelTrainer):
    """
    ModelTrainer class for the Conjoint Variational Autoencoder.
    """

    def __init__(self, data_dims, latent_dims, experiment_name, architecture,
                 optimiser_params=None, overwrite=True, save_best=True, pretrained_dir=None):
        """
        Args:
            data_dims: int, flattened data dimensionality
            latent_dims: int, flattened latent dimensionality
            experiment_name: str, name of the training/experiment for logging purposes
            architecture: str, name of the network architecture to be used
            overwrite: bool, whether to overwrite the existing trained model with the same experiment_name
            save_best: bool, whether to save the best performing model on the validation set (if provided later)
            optimiser_params: dict, parameters for the optimiser
            pretrained_dir: str, optional path to the pre-trained model directory with the hdf5 and json files
        """
        conj_vae = ConjointGaussianVariationalAutoencoder(data_dims=data_dims, latent_dims=latent_dims,
                                                          experiment_architecture=architecture,
                                                          optimiser_params=optimiser_params,
                                                          resume_from=pretrained_dir)
        super(ConjointVAEModelTrainer, self).__init__(model=conj_vae, experiment_name=experiment_name,
                                                      overwrite=overwrite, save_best=save_best)

    def fit_model(self, data, batch_size, epochs, **kwargs):
        """
        Fit the Conjoint VAE model to multiple datasets.

        Args:
            data: ndarray, training data
            batch_size: int, batch size
            epochs: int, number of epochs

        Returns:
            A loss history dict with the encoder-decoder loss.
        """
        loss_hisotry = self.model.fit(data, batch_size, epochs=epochs, **kwargs)
        return loss_hisotry


class ConjointAVBModelTrainer(ModelTrainer):
    """
    ModelTrainer class for the Conjoint Variational Autoencoder.
    """

    def __init__(self, data_dims, latent_dims, noise_dim, experiment_name, architecture,
                 use_adaptive_contrast=False, noise_basis_dim=None, noise_mode='add', 
                 optimiser_params=None, schedule=None, pretrained_dir=None, overwrite=True, save_best=True,):
        """
        Args:
            data_dims: tuple, all flattened data dimensionalities for each dataset
            latent_dims: tuple, all private and the shared latent dimensionalities for each dataset
            noise_dim: int, flattened noise dimensionality
            experiment_name: str, name of the training/experiment for logging purposes
            architecture: str, name of the network architecture to be used
            schedule: dict, schedule of training discriminator and encoder-decoder networks
            overwrite: bool, whether to overwrite the existing trained model with the same experiment_name
            save_best: bool, whether to save the best performing model on the validation set (if provided later)
            use_adaptive_contrast: bool, whether to train according to the Adaptive Contrast algorithm
            noise_basis_dim: int, the dimensionality of the noise basis vectors if AC is used.
            optimiser_params: dict, parameters for the optimiser
            pretrained_dir: str, directory from which pre-trained models (hdf5 files) can be loaded
            noise_mode: str, the way the noise will be merged with the input('add', 'concat', 'product')
        """
        conj_avb = ConjointAdversarialVariationalBayes(data_dims=data_dims, latent_dims=latent_dims,
                                                       noise_dim=noise_dim,
                                                       noise_basis_dim=noise_basis_dim,
                                                       noise_mode=noise_mode,
                                                       use_adaptive_contrast=use_adaptive_contrast,
                                                       optimiser_params=optimiser_params,
                                                       resume_from=pretrained_dir,
                                                       experiment_architecture=architecture)
        self.schedule = schedule or {'iter_disc': 1, 'iter_encdec': 1}
        super(ConjointAVBModelTrainer, self).__init__(model=conj_avb, experiment_name=experiment_name,
                                                      overwrite=overwrite, save_best=save_best)

    def fit_model(self, data, batch_size, epochs, **kwargs):
        """
        Fit the Conjoint VAE model to multiple datasets.

        Args:
            data: ndarray, training data
            batch_size: int, batch size
            epochs: int, number of epochs

        Returns:
            A loss history dict with the encoder-decoder loss.
        """
        loss_hisotry = self.model.fit(data, batch_size, epochs=epochs,
                                      discriminator_repetitions=self.schedule['iter_disc'],
                                      adaptive_contrast_sampling_steps=10,
                                      **kwargs)
        return loss_hisotry
