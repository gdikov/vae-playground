from __future__ import absolute_import
from builtins import range, next

from numpy import inf as float_inf
from tqdm import tqdm, trange
from keras.optimizers import Adam

from ..utils.config import load_config
from .losses import AVBDiscriminatorLossLayer, AVBEncoderDecoderLossLayer
from .networks import *
from ..data_iterator import AVBDataIterator, ConjointAVBDataIterator
from ..models.base_vae import BaseVariationalAutoencoder
from ..models.freezable import FreezableModel

config = load_config('global_config.yaml')


class AdversarialVariationalBayes(BaseVariationalAutoencoder):
    """
    Adversarial Variational Bayes as per "Adversarial Variational Bayes, 
    Unifying Variational Autoencoders with Generative Adversarial Networks, L. Mescheder et al., arXiv 2017".
    """
    def __init__(self, data_dim, latent_dim=2, noise_dim=None,
                 resume_from=None,
                 experiment_architecture='synthetic',
                 use_adaptive_contrast=False,
                 noise_basis_dim=None,
                 optimiser_params=None):
        """
        Args:
            data_dim: int, flattened data dimensionality
            latent_dim: int, flattened latent dimensionality
            noise_dim: int, flattened noise, dimensionality
            resume_from: str, directory with h5 and json files with the model weights and architecture
            experiment_architecture: str, network architecture descriptor
            use_adaptive_contrast: bool, whether to use an auxiliary distribution with known density,
                which is closer to q(z|x) and allows for improving the power of the discriminator.
            noise_basis_dim: int, noise basis dimensionality in case adaptive contrast is used.
            optimiser_params: dict, optional optimiser parameters
        """
        self.data_dim = data_dim
        self.noise_dim = noise_dim or data_dim
        self.latent_dim = latent_dim

        self.models_dict = {'avb_trainable_discriminator': None, 'avb_trainable_encoder_decoder': None}

        if use_adaptive_contrast:
            self.name = "avb_with_ac"
            self.noise_basis_dim = noise_basis_dim
            self.encoder = MomentEstimationEncoder(data_dim=data_dim, noise_dim=noise_dim,
                                                   noise_basis_dim=noise_basis_dim, latent_dim=latent_dim,
                                                   network_architecture=experiment_architecture)
            self.discriminator = AdaptivePriorDiscriminator(data_dim=data_dim, latent_dim=latent_dim,
                                                            network_architecture=experiment_architecture)
        else:
            self.name = "avb"
            self.encoder = StandardEncoder(data_dim=data_dim, noise_dim=noise_dim, latent_dim=latent_dim,
                                           network_architecture=experiment_architecture)
            self.discriminator = Discriminator(data_dim=data_dim, latent_dim=latent_dim,
                                               network_architecture=experiment_architecture)
        self.decoder = StandardDecoder(latent_dim=latent_dim, data_dim=data_dim,
                                       network_architecture=experiment_architecture)

        super(AdversarialVariationalBayes, self).__init__(data_dim=data_dim, noise_dim=noise_dim,
                                                          latent_dim=latent_dim, name_prefix=self.name)
        if use_adaptive_contrast:
            posterior_approximation, norm_posterior_approximation, \
                posterior_mean, posterior_var, log_adaptive_prior, log_latent_space = \
                self.encoder(self.data_input, is_learning=True)
            discriminator_output_prior = self.discriminator([self.data_input, posterior_mean, posterior_var],
                                                            from_posterior=False)
            discriminator_output_posterior = self.discriminator([self.data_input, norm_posterior_approximation],
                                                                from_posterior=True)
            reconstruction_log_likelihood = self.decoder([self.data_input, posterior_approximation],
                                                         is_learning=True)

            discriminator_loss = AVBDiscriminatorLossLayer(name='disc_loss')([discriminator_output_prior,
                                                                              discriminator_output_posterior],
                                                                             is_in_logits=True)
            decoder_loss = AVBEncoderDecoderLossLayer(name='dec_loss')([reconstruction_log_likelihood,
                                                                        discriminator_output_posterior,
                                                                        log_adaptive_prior,
                                                                        log_latent_space],
                                                                       use_adaptive_contrast=True)
        else:
            posterior_approximation = self.encoder(self.data_input, is_learning=True)
            discriminator_output_prior = self.discriminator(self.data_input, from_posterior=False)
            discriminator_output_posterior = self.discriminator([self.data_input, posterior_approximation],
                                                                from_posterior=True)
            reconstruction_log_likelihood = self.decoder([self.data_input, posterior_approximation],
                                                         is_learning=True)

            discriminator_loss = AVBDiscriminatorLossLayer(name='disc_loss')([discriminator_output_prior,
                                                                              discriminator_output_posterior],
                                                                             is_in_logits=True)
            decoder_loss = AVBEncoderDecoderLossLayer(name='dec_loss')([reconstruction_log_likelihood,
                                                                        discriminator_output_posterior],
                                                                       use_adaptive_contrast=False)

        # define the trainable models
        self.avb_trainable_discriminator = FreezableModel(inputs=self.data_input,
                                                          outputs=discriminator_loss,
                                                          name='freezable_discriminator')
        self.avb_trainable_encoder_decoder = FreezableModel(inputs=self.data_input,
                                                            outputs=decoder_loss,
                                                            name='freezable_encoder_decoder')

        if resume_from is not None:
            self.load(resume_from, custom_layers={'AVBDiscriminatorLossLayer': AVBDiscriminatorLossLayer,
                                                  'AVBEncoderDecoderLossLayer': AVBEncoderDecoderLossLayer,
                                                  'FreezableModel': FreezableModel})

        optimiser_params = optimiser_params or {'encdec': {'lr': 1e-4, 'beta_1': 0.5},
                                                'disc': {'lr': 2e-4, 'beta_1': 0.5}}
        self.avb_trainable_discriminator.freeze(freezable_layers_prefix=['disc'], deep_freeze=True)
        self.avb_trainable_encoder_decoder.unfreeze(unfreezable_layers_prefix=['dec', 'enc'], deep_unfreeze=True)
        optimiser_params_encdec = optimiser_params['encdec']
        self.avb_trainable_encoder_decoder.compile(optimizer=Adam(**optimiser_params_encdec), loss=None)

        self.avb_trainable_discriminator.unfreeze()
        self.avb_trainable_encoder_decoder.freeze()
        optimiser_params_disc = optimiser_params['disc']
        self.avb_trainable_discriminator.compile(optimizer=Adam(**optimiser_params_disc), loss=None)

        self.models_dict['avb_trainable_encoder_decoder'] = self.avb_trainable_encoder_decoder
        self.models_dict['avb_trainable_discriminator'] = self.avb_trainable_discriminator

        self.data_iterator = AVBDataIterator(data_dim=data_dim, latent_dim=latent_dim,
                                             seed=config['seed'])

    def fit(self, data, batch_size=32, epochs=1, **kwargs):
        """
        Fit the model to the training data.
        
        Args:
            data: ndarray, data array of shape (N, data_dim)
            batch_size: int, the number of samples to be used at one training pass
            epochs: int, the number of epochs for training (whole size data iterations)
        
        Keyword Args:
            discriminator_repetitions: int, gives the number of iterations for the discriminator network 
                for each batch training of the encoder-decoder network 
            checkpoint_callback: function, python function to execute when performance is improved
            validation_data: ndarray, validation data to monitor the model performance
            validation_frequency: int, after how many epochs the model should be validated
            validation_sampling_size: int, number of noisy computations for each validation sample

        Returns:
            The training history as a dict of lists of the epoch-wise losses.
        """
        discriminator_repetitions = kwargs.get('discriminator_repetitions', 1)
        val_data = kwargs.get('validation_data', None)
        val_freq = kwargs.get('validation_frequency', 10)
        val_sampling_size = kwargs.get('validation_sampling_size', 10)
        checkpoint_callback = kwargs.get('checkpoint_callback', None)

        data_iterator, iters_per_epoch = self.data_iterator.iter(data, batch_size, mode='training', shuffle=True)

        history = {'encoder_decoder_loss': [], 'discriminator_loss': [], 'elbo': []}
        current_best_score = -float_inf
        for ep in trange(epochs):
            epoch_loss_history_encdec = []
            epoch_loss_history_disc = []
            for it in range(iters_per_epoch):
                training_batch = next(data_iterator)
                loss_autoencoder = self.avb_trainable_encoder_decoder.train_on_batch(training_batch, None)
                epoch_loss_history_encdec.append(loss_autoencoder)
                for _ in range(discriminator_repetitions):
                    loss_discriminator = self.avb_trainable_discriminator.train_on_batch(training_batch, None)
                    epoch_loss_history_disc.append(loss_discriminator)

            history['encoder_decoder_loss'].append(epoch_loss_history_encdec)
            history['discriminator_loss'].append(epoch_loss_history_disc)

            if val_data is not None and (ep + 1) % val_freq == 0:
                elbo, kl_marginal, rec_err = self.evaluate(val_data, batch_size=batch_size,
                                                           sampling_size=val_sampling_size,
                                                           verbose=False)
                # possibly think of some combination of all three metrics for the score calculation
                score = elbo
                tqdm.write("ELBO estimate: {}, Posterior abnormality: {}, "
                           "Reconstruction error: {}".format(elbo, kl_marginal, rec_err))
                if checkpoint_callback is not None and current_best_score < score:
                    checkpoint_callback()
                history['elbo'].append(elbo)

        return history


class ConjointAdversarialVariationalBayes(BaseVariationalAutoencoder):
    """
    A conjoint VAE variation using Adversarial Variational Bayes model as per "Adversarial Variational Bayes,
    Unifying Variational Autoencoders with Generative Adversarial Networks, L. Mescheder et al., arXiv 2017"
    """

    def __init__(self, data_dims, latent_dims, noise_dim=None,
                 resume_from=None,
                 experiment_architecture='synthetic',
                 use_adaptive_contrast=False,
                 noise_basis_dim=None,
                 optimiser_params=None):
        """
        Args:
            data_dims: int, flattened data dimensionality
            latent_dims: int, flattened latent dimensionality
            noise_dim: int, flattened noise, dimensionality
            resume_from: str, directory with h5 and json files with the model weights and architecture
            experiment_architecture: str, network architecture descriptor
            use_adaptive_contrast: bool, whether to use an auxiliary distribution with known density,
                which is closer to q(z|x) and allows for improving the power of the discriminator.
            noise_basis_dim: int, noise basis dimensionality in case adaptive contrast is used.
            optimiser_params: dict, optional optimiser parameters
        """
        self.data_dim = data_dims
        self.noise_dim = noise_dim or data_dims
        self.latent_dim = latent_dims

        self.models_dict = {'avb_trainable_discriminator': None, 'avb_trainable_encoder_decoder': None}

        if use_adaptive_contrast:
            self.name = "conjoint_avb_with_ac"
            self.noise_basis_dim = noise_basis_dim
            raise NotImplementedError("AC for the conjoint AVB model is not supported yet.")
        else:
            self.name = "conjoint_avb"
            self.encoder = StandardConjointEncoder(data_dims=data_dims, noise_dim=noise_dim, latent_dims=latent_dims,
                                                   network_architecture=experiment_architecture)
            self.discriminator = ConjointDiscriminator(data_dims=data_dims, latent_dims=latent_dims,
                                                       network_architecture=experiment_architecture)
        self.decoder = ConjointDecoder(latent_dims=latent_dims, data_dims=data_dims,
                                       network_architecture=experiment_architecture)

        super(ConjointAdversarialVariationalBayes, self).__init__(data_dim=data_dims, noise_dim=noise_dim,
                                                                  latent_dim=sum(latent_dims), name_prefix=self.name)
        if use_adaptive_contrast:
            raise NotImplementedError("AC for the conjoint AVB model is not supported yet.")
        else:
            posterior_approximation = self.encoder(self.data_input, is_learning=True)
            discriminator_output_prior = self.discriminator(self.data_input, from_posterior=False)
            discriminator_output_posterior = self.discriminator(self.data_input + [posterior_approximation],
                                                                from_posterior=True)
            reconstruction_log_likelihood = self.decoder(self.data_input + [posterior_approximation],
                                                         is_learning=True)

            discriminator_loss = AVBDiscriminatorLossLayer(name='disc_loss')([discriminator_output_prior,
                                                                              discriminator_output_posterior],
                                                                             is_in_logits=True)
            decoder_loss = AVBEncoderDecoderLossLayer(name='dec_loss')([reconstruction_log_likelihood,
                                                                        discriminator_output_posterior],
                                                                       use_adaptive_contrast=False)

        # define the trainable models
        self.avb_trainable_discriminator = FreezableModel(inputs=self.data_input,
                                                          outputs=discriminator_loss,
                                                          name='freezable_discriminator')
        self.avb_trainable_encoder_decoder = FreezableModel(inputs=self.data_input,
                                                            outputs=decoder_loss,
                                                            name='freezable_encoder_decoder')

        if resume_from is not None:
            self.load(resume_from, custom_layers={'AVBDiscriminatorLossLayer': AVBDiscriminatorLossLayer,
                                                  'AVBEncoderDecoderLossLayer': AVBEncoderDecoderLossLayer,
                                                  'FreezableModel': FreezableModel})

        optimiser_params = optimiser_params or {'encdec': {'lr': 1e-4, 'beta_1': 0.5},
                                                'disc': {'lr': 2e-4, 'beta_1': 0.5}}
        self.avb_trainable_discriminator.freeze(freezable_layers_prefix=['disc'], deep_freeze=True)
        self.avb_trainable_encoder_decoder.unfreeze(unfreezable_layers_prefix=['dec', 'enc'], deep_unfreeze=True)
        optimiser_params_encdec = optimiser_params['encdec']
        self.avb_trainable_encoder_decoder.compile(optimizer=Adam(**optimiser_params_encdec), loss=None)

        self.avb_trainable_discriminator.unfreeze()
        self.avb_trainable_encoder_decoder.freeze()
        optimiser_params_disc = optimiser_params['disc']
        self.avb_trainable_discriminator.compile(optimizer=Adam(**optimiser_params_disc), loss=None)

        self.models_dict['avb_trainable_encoder_decoder'] = self.avb_trainable_encoder_decoder
        self.models_dict['avb_trainable_discriminator'] = self.avb_trainable_discriminator

        self.data_iterator = ConjointAVBDataIterator(data_dim=data_dims, latent_dim=latent_dims,
                                                     seed=config['seed'])

    def fit(self, data, batch_size=32, epochs=1, **kwargs):
        """
        Fit the model to the training data.

        Args:
            data: ndarray, data array of shape (N, data_dim)
            batch_size: int, the number of samples to be used at one training pass
            epochs: int, the number of epochs for training (whole size data iterations)

        Keyword Args:
            discriminator_repetitions: int, gives the number of iterations for the discriminator network
                for each batch training of the encoder-decoder network
            checkpoint_callback: function, python function to execute when performance is improved
            validation_data: ndarray, validation data to monitor the model performance
            validation_frequency: int, after how many epochs the model should be validated
            validation_sampling_size: int, number of noisy computations for each validation sample

        Returns:
            The training history as a dict of lists of the epoch-wise losses.
        """
        discriminator_repetitions = kwargs.get('discriminator_repetitions', 1)
        val_data = kwargs.get('validation_data', None)
        val_freq = kwargs.get('validation_frequency', 10)
        val_sampling_size = kwargs.get('validation_sampling_size', 10)
        checkpoint_callback = kwargs.get('checkpoint_callback', None)

        data_iterator, iters_per_epoch = self.data_iterator.iter(data, batch_size, mode='training', shuffle=True)

        history = {'encoder_decoder_loss': [], 'discriminator_loss': [], 'elbo': []}
        current_best_score = -float_inf
        for ep in trange(epochs):
            epoch_loss_history_encdec = []
            epoch_loss_history_disc = []
            for it in range(iters_per_epoch):
                training_batch = next(data_iterator)
                loss_autoencoder = self.avb_trainable_encoder_decoder.train_on_batch(training_batch, None)
                epoch_loss_history_encdec.append(loss_autoencoder)
                for _ in range(discriminator_repetitions):
                    loss_discriminator = self.avb_trainable_discriminator.train_on_batch(training_batch, None)
                    epoch_loss_history_disc.append(loss_discriminator)

            history['encoder_decoder_loss'].append(epoch_loss_history_encdec)
            history['discriminator_loss'].append(epoch_loss_history_disc)

            if val_data is not None and (ep + 1) % val_freq == 0:
                elbo, kl_marginal, rec_err = self.evaluate(val_data, batch_size=batch_size,
                                                           sampling_size=val_sampling_size,
                                                           verbose=False)
                # possibly think of some combination of all three metrics for the score calculation
                score = elbo
                tqdm.write("ELBO estimate: {}, Posterior abnormality: {}, "
                           "Reconstruction error: {}".format(elbo, kl_marginal, rec_err))
                if checkpoint_callback is not None and current_best_score < score:
                    checkpoint_callback()
                history['elbo'].append(elbo)

        return history
