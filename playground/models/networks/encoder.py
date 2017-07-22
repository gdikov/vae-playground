from __future__ import absolute_import
from builtins import range

import logging
import keras.backend as ker

from numpy import pi as pi_const
from numpy import sqrt
from keras.layers import Lambda, Multiply, Add, Dense, Concatenate
from keras.models import Input
from keras.models import Model

from .sampling import sample_standard_normal_noise
from .architectures import get_network_by_name
from ...utils.config import load_config

config = load_config('global_config.yaml')
logger = logging.getLogger(__name__)


class BaseEncoder(object):
    def __init__(self, data_dim, noise_dim, latent_dim, network_architecture='synthetic', name='encoder'):
        logger.info("Initialising {} model with {}-dimensional data and {}-dimensional noise input "
                    "and {} dimensional latent output".format(name, data_dim, noise_dim, latent_dim))
        self.name = '_'.join(name.lower().split())
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.network_architecture = network_architecture
        self.data_input = Input(shape=(data_dim,), name='enc_data_input')

        self.standard_normal_sampler = Lambda(sample_standard_normal_noise, name='enc_standard_normal_sampler')
        self.standard_normal_sampler.arguments = {'data_dim': self.data_dim, 'noise_dim': self.noise_dim,
                                                  'seed': config['seed']}

        self.standard_normal_sampler2 = Lambda(sample_standard_normal_noise, name='enc_standard_normal_sampler2')
        self.standard_normal_sampler2.arguments = {'data_dim': self.data_dim, 'noise_dim': self.noise_dim,
                                                   'seed': config['seed']}

    def __call__(self, *args, **kwargs):
        return None


class StandardEncoder(BaseEncoder):
    """
    An Encoder model is trained to parametrise an arbitrary posterior approximate distribution given some 
    input x, i.e. q(z|x). The model takes as input concatenated data samples and arbitrary noise and produces
    a latent encoding:
    
      Data                              Input
     - - - - - - - - -   
       |       Noise                      
       |         |                        
       ----------- <-- concatenation    
            |                           Encoder model
       -----------
       | Encoder |                      
       -----------
            |
        Latent space                    Output
    
    """
    def __init__(self, data_dim, noise_dim, latent_dim, network_architecture='synthetic'):
        """
        Args:
            data_dim: int, flattened data space dimensionality 
            noise_dim: int, flattened noise space dimensionality
            latent_dim: int, flattened latent space dimensionality
            network_architecture: str, the architecture name for the body of the Encoder model
        """
        super(StandardEncoder, self).__init__(data_dim=data_dim, noise_dim=noise_dim, latent_dim=latent_dim,
                                              network_architecture=network_architecture, name='Standard Encoder')

        noise_input = self.standard_normal_sampler(self.data_input)
        encoder_body_model = get_network_by_name['encoder'][network_architecture](data_dim, noise_dim, latent_dim)
        latent_factors = encoder_body_model([self.data_input, noise_input])
        self.encoder_model = Model(inputs=self.data_input, outputs=latent_factors, name='encoder')

    def __call__(self, *args, **kwargs):
        """
        Make the Encoder model callable on a list of Input layers.
        
        Args:
            *args: a list of input layers from the super-model or numpy arrays in case of test-time inference.
            **kwargs: 

        Returns:
            An Encoder model.
        """
        return self.encoder_model(args[0])


class StandardConjointEncoder(object):
    """
        A StandardConjointEncoder parametrises an arbitrary latent distribution given two (or more) datasets,
        by partially sharing the latent vector between two (or more) encoders:

              Data_1                      Data_2
                |                           |
           -------------              -------------
           | Encoder_1 |              | Encoder_2 |
           -------------              -------------
                |                           |
            Latent_1 -- Latent_shared -- Latent_2
            \_______________  __________________/
                            V
                   concatenated latent space

        """

    def __init__(self, data_dims, noise_dim, latent_dims, network_architecture='synthetic', noise_mode='add'):
        """
        Args:
            data_dims: tuple, flattened data dimension for each dataset
            noise_dim: int, flattened noise dimensionality
            latent_dims: tuple, flattened latent dimensions for each private latent space and the dimension
                of the shared space.
            network_architecture: str, the codename of the encoder network architecture (will be the same for all)
            noise_mode: str, the way the noise will be merged with the input('add', 'concat', 'product')
        """
        assert len(latent_dims) == len(data_dims) + 1, \
            "Expected too receive {} private latent spaces and one shared for {} data inputs " \
            "but got {} instead.".format(len(data_dims) + 1, len(data_dims), len(latent_dims))

        name = 'Standard Conjoint Encoder'
        logger.info("Initialising {} model with {}-dimensional data inputs "
                    "and {}-dimensional latent outputs (last one is shared)".format(name, data_dims, latent_dims))

        latent_space, features = [], []
        inputs = [Input(shape=(dim,), name="enc_data_input_{}".format(i)) for i, dim in enumerate(data_dims)]
        standard_normal_sampler = Lambda(sample_standard_normal_noise, name='enc_normal_sampler')

        if noise_mode == 'concatenate':
            if network_architecture == 'mnist':
                assert ((noise_dim % sqrt(data_dims[0]) == 0) and (noise_dim % sqrt(data_dims[1]) == 0)), \
                    "Expected to receive a noise_dim that, when concatenated, can form a rectangle with " \
                    "the given inputs_dims. Received {} noise and {} data dimensions.".format(noise_dim, data_dims)
        elif noise_mode == 'product':
            assert data_dims[0] == noise_dim and data_dims[1] == noise_dim, \
                "Expected to receive a noise_dim that is equal to the given inputs dimensions. " \
                "Received {} noise and {} data dimensions.".format(noise_dim, data_dims)
        else:
            raise ValueError("Only the noise modes 'add', 'concat' and 'product' are available.")
        standard_normal_sampler.arguments = {'seed': config['seed'], 'noise_dim': noise_dim, 'mode': noise_mode}

        for i, inp in enumerate(inputs):
            noise_input = standard_normal_sampler(inp)
            feature = get_network_by_name['conjoint_encoder'][network_architecture](noise_input,
                                                                                    'enc_feat_{}'.format(i))
            latent_factors = Dense(latent_dims[i], activation=None, name='enc_latent_private_{}'.format(i))(feature)
            latent_space.append(latent_factors)
            features.append(feature)

        merged_features = Concatenate(axis=-1, name='enc_concat_features')(features)
        shared_latent = Dense(latent_dims[-1], activation=None, name='enc_shared_latent')(merged_features)
        latent_space = Concatenate(axis=-1, name='enc_concat_all_features')(latent_space + [shared_latent])
        self.encoder_model = Model(inputs=inputs, outputs=latent_space, name='encoder')

    def __call__(self, *args, **kwargs):
        """
        Make the encoder callable on a list of data inputs.

        Args:
            *args: list or tuple, Keras input tensors for the data arrays from each dataset

        Returns:
            The output of the model called on the input tensor (total, i.e. private and shared latent space)
        """
        return self.encoder_model(args[0])


class MomentEstimationEncoder(BaseEncoder):
    """
    An Encoder model is trained to parametrise an arbitrary posterior approximate distribution given some 
    input x, i.e. q(z|x). The model takes as input concatenated data samples and arbitrary noise and produces
    a latent encoding. Additionally the first two moments (mean and variance) are estimated empirically, which is
    necessary for the Adaptive Contrast learning algorithm. Schematically it can be represented as follows:

       Data  Noise
        |      |
        |      |
        |      | 
       -----------
       | Encoder |  ----> empirical mean and variance
       -----------
            |
        Latent space

    """

    def __init__(self, data_dim, noise_dim, noise_basis_dim, latent_dim, network_architecture='mnist'):
        """
        Args:
            data_dim: int, flattened data space dimensionality 
            noise_dim: int, flattened noise space dimensionality
            noise_basis_dim: int, noise basis vectors dimensionality
            latent_dim: int, flattened latent space dimensionality
            network_architecture: str, the architecture name for the body of the moment estimation Encoder model
        """
        super(MomentEstimationEncoder, self).__init__(data_dim=data_dim, noise_dim=noise_dim, latent_dim=latent_dim,
                                                      network_architecture=network_architecture,
                                                      name='Posterior Moment Estimation Encoder')

        models = get_network_by_name['moment_estimation_encoder'][network_architecture](
            data_dim=data_dim, noise_dim=noise_dim, noise_basis_dim=noise_basis_dim, latent_dim=latent_dim)

        data_feature_extraction, noise_basis_extraction = models

        self.standard_normal_sampler.arguments['n_basis'] = noise_basis_dim
        noise = self.standard_normal_sampler(self.data_input)
        noise_basis_vectors = noise_basis_extraction(noise)

        coefficients_and_z0 = data_feature_extraction(self.data_input)
        coefficients = coefficients_and_z0[:-1]
        z_0 = coefficients_and_z0[-1]

        latent_factors = []
        for i, (a, v) in enumerate(zip(coefficients, noise_basis_vectors)):
            latent_factors.append(Multiply(name='enc_elemwise_coeff_vecs_mult_{}'.format(i))([a, v]))
        latent_factors = Add(name='enc_add_weighted_vecs')(latent_factors)
        latent_factors = Add(name='add_z0_to_linear_combination')([z_0, latent_factors])

        self.standard_normal_sampler2.arguments['n_basis'] = noise_basis_dim
        self.standard_normal_sampler2.arguments['n_samples'] = 100
        more_noise = self.standard_normal_sampler2(self.data_input)
        sampling_basis_vectors = noise_basis_extraction(more_noise)

        posterior_mean = []
        posterior_var = []
        for i in range(noise_basis_dim):
            # compute empirical mean as the batchsize-wise mean of all sampling vectors for each basis dimension
            mean_basis_vectors_i = Lambda(lambda x: ker.mean(x, axis=0),
                                          name='enc_noise_basis_vectors_mean_{}'.format(i))(sampling_basis_vectors[i])
            # and do the same for the empirical variance and compute similar posterior parametrization for the variance
            var_basis_vectors_i = Lambda(lambda x: ker.var(x, axis=0),
                                         name='enc_noise_basis_vectors_var_{}'.format(i))(sampling_basis_vectors[i])
            # and parametrise the posterior moment as described in the AVB paper
            posterior_mean.append(Lambda(lambda x: x[0] * x[1],
                                         name='enc_moments_mult_mean_{}'.format(i))([coefficients[i],
                                                                                     mean_basis_vectors_i]))

            # compute similar posterior parametrization for the variance
            posterior_var.append(Lambda(lambda x: x[0]*x[0]*x[1],
                                        name='enc_moments_mult_var_{}'.format(i))([coefficients[i],
                                                                                   var_basis_vectors_i]))
        posterior_mean = Add(name='enc_moments_mean')(posterior_mean)
        posterior_mean = Add(name='enc_moments_mean_add_z0')([posterior_mean, z_0])
        posterior_var = Add(name='enc_moments_var')(posterior_var)

        normalised_latent_factors = Lambda(lambda x: (x[0] - x[1]) / ker.sqrt(x[2] + 1e-5),
                                           name='enc_norm_posterior')([latent_factors, posterior_mean, posterior_var])

        log_latent_space = Lambda(lambda x: -0.5 * ker.sum(x**2 + ker.log(2*pi_const), axis=1),
                                  name='enc_log_approx_posterior')(latent_factors)

        log_adaptive_prior = Lambda(lambda x: -0.5 * ker.sum(x[0]**2 + ker.log(x[1]) + ker.log(2*pi_const), axis=1),
                                    name='enc_log_adaptive_prior')([normalised_latent_factors, posterior_var])

        self.encoder_inference_model = Model(inputs=self.data_input, outputs=latent_factors,
                                             name='encoder_inference_model')
        self.encoder_trainable_model = Model(inputs=self.data_input,
                                             outputs=[latent_factors, normalised_latent_factors,
                                                      posterior_mean, posterior_var,
                                                      log_adaptive_prior, log_latent_space],
                                             name='encoder_trainable_model')

    def __call__(self, *args, **kwargs):
        """
        Make the Encoder model callable on a list of Input layers.

        Args:
            *args: a list of input layers from the super-model or numpy arrays in case of test-time inference.
            **kwargs: 

        Returns:
            An Encoder model.
        """
        is_learning = kwargs.get('is_learning', True)
        if is_learning:
            return self.encoder_trainable_model(args[0])
        return self.encoder_inference_model(args[0])


class ReparametrisedGaussianEncoder(BaseEncoder):
    """
    A ReparametrisedGaussianEncoder model is trained to parametrise a Gaussian latent variables:

           Data              
            | 
       -----------
       | Encoder |
       -----------
            |
    mu + sigma * Noise   <--- Reparametrised Gaussian latent space

    """

    def __init__(self, data_dim, noise_dim, latent_dim, network_architecture='synthetic', name=None):
        """
        Args:
            data_dim: int, flattened data space dimensionality 
            noise_dim: int, flattened noise space dimensionality
            latent_dim: int, flattened latent space dimensionality
            network_architecture: str, the architecture name for the body of the reparametrised Gaussian Encoder model
            name: str, optional identifier of the model
        """
        super(ReparametrisedGaussianEncoder, self).__init__(data_dim=data_dim,
                                                            noise_dim=noise_dim,
                                                            latent_dim=latent_dim,
                                                            network_architecture=network_architecture,
                                                            name=name or 'Reparametrised Gaussian Encoder')

        # FIXME: change names to be unique
        latent_mean, latent_log_var = get_network_by_name['reparametrised_encoder'][network_architecture](
            self.data_input, latent_dim, name_prefix=self.name)

        noise = self.standard_normal_sampler(self.data_input)

        # due to some BUG in Keras, the module name `ker` is not visible within the lambda expression
        # as a workaround, define the function outside the Lambda layer
        def lin_transform_standard_gaussian(params):
            from keras.backend import exp
            mu, log_sigma, z = params
            transformed_z = mu + exp(log_sigma / 2.0) * z
            return transformed_z

        latent_factors = Lambda(lin_transform_standard_gaussian,
                                name=self.name + 'enc_reparametrised_latent')([latent_mean, latent_log_var, noise])

        self.encoder_inference_model = Model(inputs=self.data_input, outputs=latent_factors,
                                             name=self.name + 'encoder_inference')
        self.encoder_learning_model = Model(inputs=self.data_input,
                                            outputs=[latent_factors, latent_mean, latent_log_var],
                                            name=self.name + 'encoder_learning')

    def __call__(self, *args, **kwargs):
        """
        Make the Encoder model callable on a list of Input layers.

        Args:
            *args: a list of input layers from the super-model or numpy arrays in case of test-time inference.

        Keyword Args:
            is_learning: bool, whether the model is used for training or inference. The output is either 
                the latent space or the latent space and the means and variances from which it is reparametrised.  

        Returns:
            An Encoder model.
        """
        is_learning = kwargs.get('is_learning', True)
        if is_learning:
            return self.encoder_learning_model(args[0])
        else:
            return self.encoder_inference_model(args[0])


class ReparametrisedGaussianConjointEncoder(object):
    """
    A ReparametrisedGaussianConjointEncoder parametrises a Gaussian latent distribution given two (or more) datasets,
    by partially sharing the latent vector between two (or more) encoders:

          Data_1                      Data_2
            |                           |
       -------------              -------------
       | Encoder_1 |              | Encoder_2 |
       -------------              -------------
            |                           |
        Latent_1 -- Latent_shared -- Latent_2
        \_______________  __________________/
                        V
               latent mu & sigma
                        V
            mu + sigma * Noise   <--- Reparametrised Gaussian latent space

    """
    def __init__(self, data_dims, latent_dims, network_architecture='synthetic'):
        """
        Args:
            data_dims: tuple, flattened data dimension for each dataset
            latent_dims: tuple, flattened latent dimensions for each private latent space and the dimension
                of the shared space.
            network_architecture: str, the codename of the encoder network architecture (will be the same for all)
        """
        assert len(latent_dims) == len(data_dims) + 1, \
            "Expected too receive {} private latent spaces and one shared for {} data inputs " \
            "but got {} instead.".format(len(data_dims) + 1, len(data_dims), len(latent_dims))

        # NOTE: this encoder is better off not having BaseEncoder as super class.
        # This might be refactored in the future if multiple variations of the conjoint model are to be implemented,
        # and this class can be used as a base class for them.
        name = 'Reparametrised Gaussian Conjoint Encoder'
        logger.info("Initialising {} model with {}-dimensional data inputs "
                    "and {}-dimensional latent outputs (last one is shared)".format(name, data_dims, latent_dims))

        def lin_transform_standard_gaussian(params):
            from keras.backend import exp
            mu, log_sigma, z = params
            transformed_z = mu + exp(log_sigma / 2.0) * z
            return transformed_z

        inputs, latent_space, latent_means, latent_log_vars, features = [], [], [], [], []
        for i in range(len(data_dims)):
            data_input = Input(shape=(data_dims[i],), name="enc_data_input_{}".format(i))
            features_i = get_network_by_name['conjoint_encoder'][network_architecture](data_input, 'conj_{}'.format(i))
            private_latent_mean = Dense(latent_dims[i], activation=None, name='enc_mean_{}'.format(i))(features_i)
            # since the variance must be positive and this is not easy to restrict, take it in the log domain
            # and exponentiate it during the reparametrisation
            private_latent_log_var = Dense(latent_dims[i], activation=None, name='enc_log_var_{}'.format(i))(features_i)

            inputs.append(data_input)
            latent_means.append(private_latent_mean)
            latent_log_vars.append(private_latent_log_var)
            features.append(features_i)

        features = Concatenate(axis=-1, name='enc_concat_all_features')(features)

        shared_latent_mean = Dense(latent_dims[-1], activation=None, name='enc_shared_mean')(features)
        shared_latent_log_var = Dense(latent_dims[-1], activation=None, name='enc_shared_log_var')(features)
        total_latent_mean = Concatenate(axis=-1, name='enc_total_mean')(latent_means + [shared_latent_mean])
        total_latent_log_var = Concatenate(axis=-1, name='enc_total_log_var')(latent_log_vars + [shared_latent_log_var])

        # introduce the noise and reparametrise it
        standard_normal_sampler = Lambda(sample_standard_normal_noise, name='enc_normal_sampler')
        standard_normal_sampler.arguments = {'data_dim': sum(data_dims),
                                             'noise_dim': sum(latent_dims),
                                             'seed': config['seed']}
        noise = standard_normal_sampler(total_latent_mean)

        total_latent_space = Lambda(lin_transform_standard_gaussian,
                                    name='enc_latent')([total_latent_mean, total_latent_log_var, noise])

        self.encoder_inference_model = Model(inputs=inputs, outputs=total_latent_space, name='encoder_inference')
        self.encoder_learning_model = Model(inputs=inputs,
                                            outputs=[total_latent_space, total_latent_mean, total_latent_log_var],
                                            name='encoder_learning')

    def __call__(self, *args, **kwargs):
        """
        Make the encoder callable on a list of data inputs.

        Args:
            *args: list or tuple, Keras input tensors for the data arrays from each dataset

        Keyword Args:
            is_learning: bool, whether the model is used in training or testing

        Returns:
            The output of the model called on the input tensor (total, i.e. private and shared latent space)
        """
        is_learning = kwargs.get('is_learning', True)
        if is_learning:
            return self.encoder_learning_model(args[0])
        else:
            return self.encoder_inference_model(args[0])
