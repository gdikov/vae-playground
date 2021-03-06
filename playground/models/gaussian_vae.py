from __future__ import absolute_import
from builtins import range, next

from keras.models import Model
from keras.optimizers import RMSprop, Adam
from tqdm import tqdm, trange
from numpy import inf as float_inf

from ..utils.config import load_config
from .losses import VAELossLayer
from .networks import ReparametrisedGaussianEncoder, StandardDecoder, \
    ReparametrisedGaussianConjointEncoder, ConjointDecoder
from ..data_iterator import VAEDataIterator, ConjointVAEDataIterator
from ..models import BaseVariationalAutoencoder

config = load_config('global_config.yaml')


class GaussianVariationalAutoencoder(BaseVariationalAutoencoder):
    def __init__(self, data_dim, latent_dim, resume_from=None,
                 experiment_architecture='synthetic', optimiser_params=None):
        """
        Args:
            data_dim: int, flattened data dimensionality 
            latent_dim: int, flattened latent dimensionality
            resume_from: str, optional folder name with pre-trained models
            experiment_architecture: str, network architecture descriptor
            optimiser_params: dict, optional optimiser parameters
        """
        self.name = "gaussian_vae"
        self.models_dict = {'vae_model': None}

        self.encoder = ReparametrisedGaussianEncoder(data_dim=data_dim, noise_dim=latent_dim, latent_dim=latent_dim,
                                                     network_architecture=experiment_architecture)
        self.decoder = StandardDecoder(data_dim=data_dim, latent_dim=latent_dim,
                                       network_architecture=experiment_architecture)

        # init the base class' inputs and testing models and reuse them
        super(GaussianVariationalAutoencoder, self).__init__(data_dim=data_dim, noise_dim=latent_dim,
                                                             latent_dim=latent_dim, name_prefix=self.name)

        posterior_approximation, latent_mean, latent_log_var = self.encoder(self.data_input, is_learning=True)
        reconstruction_log_likelihood = self.decoder([self.data_input, posterior_approximation], is_learning=True)
        vae_loss = VAELossLayer(name='vae_loss')([reconstruction_log_likelihood, latent_mean, latent_log_var])

        self.vae_model = Model(inputs=self.data_input, outputs=vae_loss)

        if resume_from is not None:
            self.load(resume_from, custom_layers={'VAELossLayer': VAELossLayer})

        optimiser_params = optimiser_params or {'lr': 1e-3}
        self.vae_model.compile(optimizer=RMSprop(**optimiser_params), loss=None)

        self.models_dict['vae_model'] = self.vae_model
        self.data_iterator = VAEDataIterator(data_dim=data_dim, latent_dim=latent_dim, seed=config['seed'])

    def fit(self, data, batch_size=32, epochs=1, **kwargs):
        """
        Fit the Gaussian Variational Autoencoder onto the training data.
        
        Args:
            data: ndarray, training data
            batch_size: int, number of samples to be fit at one pass
            epochs: int, number of whole-size iterations on the training data

        Keyword Args:
            validation_data: ndarray, validation data to monitor the model performance
            validation_frequency: int, after how many epochs the model should be validated
            validation_sampling_size: int, number of noisy computations for each validation sample
            checkpoint_callback: function, python function to execute when performance is improved

        Returns:
            A training history dict.
        """
        val_data = kwargs.get('validation_data', None)
        val_freq = kwargs.get('validation_frequency', 10)
        val_sampling_size = kwargs.get('validation_sampling_size', 10)
        checkpoint_callback = kwargs.get('checkpoint_callback', None)

        data_iterator, batches_per_epoch = self.data_iterator.iter(data, batch_size, mode='training',
                                                                   shuffle=True, grouping_mode='by_pairs')

        history = {'vae_loss': [], 'elbo': []}
        current_best_score = -float_inf
        for ep in trange(epochs):
            epoch_loss_history_vae = []
            for it in range(batches_per_epoch):
                data_batch = next(data_iterator)
                loss_autoencoder = self.vae_model.train_on_batch(data_batch[:-1], None)
                epoch_loss_history_vae.append(loss_autoencoder)
            history['vae_loss'].append(epoch_loss_history_vae)

            if val_data is not None and (ep + 1) % val_freq == 0:
                elbo, kl_marginal, rec_err = self.evaluate(val_data, batch_size=batch_size,
                                                           sampling_size=val_sampling_size,
                                                           verbose=False)
                # possibly think of some combination of all three metrics for the score calculation
                score = elbo
                tqdm.write("ELBO estimate: {}, Posterior abnormality: {}, "
                           "Reconstruction error: {}".format(elbo, kl_marginal, rec_err))
                if checkpoint_callback is not None and score > current_best_score:
                    checkpoint_callback()
                    current_best_score = score
                history['elbo'].append(elbo)
        return history


class ConjointGaussianVariationalAutoencoder(BaseVariationalAutoencoder):
    def __init__(self, data_dims, latent_dims, resume_from=None,
                 experiment_architecture='synthetic', optimiser_params=None):
        """
        Args:
            data_dims: int, flattened data dimensionality
            latent_dims: int, flattened latent dimensionality
            resume_from: str, optional folder name with pre-trained models
            experiment_architecture: str, network architecture descriptor
            optimiser_params: dict, optional optimiser parameters
        """
        self.name = "conjoint_gaussian_vae"
        self.models_dict = {'conjoint_vae_model': None}

        self.encoder = ReparametrisedGaussianConjointEncoder(data_dims=data_dims, latent_dims=latent_dims,
                                                             network_architecture=experiment_architecture)
        self.decoder = ConjointDecoder(data_dims=data_dims, latent_dims=latent_dims,
                                       network_architecture=experiment_architecture)
        # init the base class' inputs and testing models and reuse them
        super(ConjointGaussianVariationalAutoencoder, self).__init__(data_dim=data_dims, noise_dim=sum(latent_dims),
                                                                     latent_dim=sum(latent_dims),
                                                                     name_prefix=self.name)

        posterior_approximation, latent_mean, latent_log_var = self.encoder(self.data_input, is_learning=True)
        reconstruction_log_likelihood = self.decoder(self.data_input + [posterior_approximation], is_learning=True)
        vae_loss = VAELossLayer(name='vae_loss')([reconstruction_log_likelihood, latent_mean, latent_log_var])

        self.conjoint_vae_model = Model(inputs=self.data_input, outputs=vae_loss)

        if resume_from is not None:
            self.load(resume_from, custom_layers={'VAELossLayer': VAELossLayer})

        optimiser_params = optimiser_params or {'lr': 1e-3}
        self.conjoint_vae_model.compile(optimizer=Adam(**optimiser_params), loss=None)

        self.models_dict['conjoint_vae_model'] = self.conjoint_vae_model
        self.data_iterator = ConjointVAEDataIterator(data_dim=data_dims, latent_dim=latent_dims, seed=config['seed'])

    def fit(self, data, batch_size=32, epochs=1, **kwargs):
        """
        Fit the Gaussian Variational Autoencoder onto the training data.

        Args:
            data: ndarray, training data
            batch_size: int, number of samples to be fit at one pass
            epochs: int, number of whole-size iterations on the training data

        Keyword Args:
            validation_data: ndarray, validation data to monitor the model performance
            validation_frequency: int, after how many epochs the model should be validated
            validation_sampling_size: int, number of noisy computations for each validation sample
            checkpoint_callback: function, python function to execute when performance is improved

        Returns:
            A training history dict.
        """
        val_data = kwargs.get('validation_data', None)
        val_freq = kwargs.get('validation_frequency', 10)
        val_sampling_size = kwargs.get('validation_sampling_size', 10)
        checkpoint_callback = kwargs.get('checkpoint_callback', None)
        iter_grouping_mode = kwargs.get('grouping_mode', 'by_targets')

        data_iterator, batches_per_epoch = self.data_iterator.iter(data, batch_size, mode='training',
                                                                   shuffle=True, grouping_mode=iter_grouping_mode)

        history = {'conjoint_vae_loss': [], 'elbo': []}
        current_best_score = -float_inf
        for ep in trange(epochs):
            epoch_loss_history_vae = []
            for it in range(batches_per_epoch):
                data_batch = next(data_iterator)
                loss_autoencoder = self.conjoint_vae_model.train_on_batch(data_batch, None)
                epoch_loss_history_vae.append(loss_autoencoder)
            history['conjoint_vae_loss'].append(epoch_loss_history_vae)

            if val_data is not None and (ep + 1) % val_freq == 0:
                elbo, kl_marginal, rec_err = self.evaluate(val_data, batch_size=batch_size,
                                                           sampling_size=val_sampling_size,
                                                           verbose=False)
                # possibly think of some combination of all three metrics for the score calculation
                score = elbo
                tqdm.write("ELBO estimate: {}, Posterior abnormality: {}, "
                           "Reconstruction error: {}".format(elbo, kl_marginal, rec_err))
                if checkpoint_callback is not None and score > current_best_score:
                    checkpoint_callback()
                    current_best_score = score
                history['elbo'].append(elbo)
        return history
