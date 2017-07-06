from __future__ import absolute_import

import logging

from keras.layers import Lambda, Dense
from keras.models import Model, Input

from .architectures import get_network_by_name

logger = logging.getLogger(__name__)


class BaseDecoder(object):
    def __init__(self, latent_dim, data_dim, network_architecture='synthetic', name='decoder'):
        logger.info("Initialising {} model with {}-dimensional data "
                    "and {}-dimensional latent input.".format(name, data_dim, latent_dim))
        self.name = name
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.network_architecture = network_architecture
        self.data_input = Input(shape=(self.data_dim,), name='dec_ll_estimator_data_input')
        self.latent_input = Input(shape=(self.latent_dim,), name='dec_latent_input')

    def __call__(self, *args, **kwargs):
        return None


class StandardDecoder(BaseDecoder):
    """
    A StandardDecoder model has inputs comprising of a latent encoding given by an Encoder model, a prior sampler
    or other custom input and the raw Encoder data input, which is needed to estimate the reconstructed 
    data log likelihood. It can be visualised as:
     
      Data    Latent
       |        |
       |    -----------
       |    | Decoder |
       |    -----------
       |        |
       |      Output
       |    probability    --->  Generated data
       |        |
       ---> Log Likelihood ---> -(reconstruction loss)
    
    Note that the reconstruction loss is not used when the model training ends. It serves only the purpose to 
    define a measure of loss which is optimised. 
    """
    def __init__(self, latent_dim, data_dim, network_architecture='synthetic'):
        """
        Args:
            latent_dim: int, the flattened dimensionality of the latent space 
            data_dim: int, the flattened dimensionality of the output space (data space)
            network_architecture: str, the architecture name for the body of the StandardDecoder model
        """
        super(StandardDecoder, self).__init__(latent_dim=latent_dim, data_dim=data_dim,
                                              network_architecture=network_architecture,
                                              name='Standard Decoder')

        generator_body = get_network_by_name['decoder'][network_architecture](self.latent_input)

        # NOTE: all decoder layers have names prefixed by `dec`.
        # This is essential for the partial model freezing during training.
        sampler_params = Dense(self.data_dim, activation='sigmoid', name='dec_sampler_params')(generator_body)

        # a probability clipping is necessary for the Bernoulli `log_prob` property produces NaNs in the border cases.
        sampler_params = Lambda(lambda x: 1e-6 + (1 - 2e-6) * x, name='dec_probs_clipper')(sampler_params)

        def bernoulli_log_probs(args):
            from tensorflow.contrib.distributions import Bernoulli
            mu, x = args
            log_px = Bernoulli(probs=mu, name='dec_bernoulli').log_prob(x)
            return log_px

        log_probs = Lambda(bernoulli_log_probs, name='dec_bernoulli_logprob')([sampler_params, self.data_input])

        self.generator = Model(inputs=self.latent_input, outputs=sampler_params, name='dec_sampling')
        self.ll_estimator = Model(inputs=[self.data_input, self.latent_input], outputs=log_probs, name='dec_trainable')

    def __call__(self, *args, **kwargs):
        """
        Make the StandardDecoder model callable on lists of Input layers or tensors.
        
        Args:
            *args: a list of input layers or tensors or numpy arrays, or a single input layer, tensor or numpy array.
        Keyword Args:
            is_learning: bool, whether the model is used for training or data generation. The output is either 
                the reconstruction log likelihood or the output probabilities in the data space respectively.

        Returns:
            A StandardDecoder model in `training` or `data generation` mode.
        """
        is_learninig = kwargs.get('is_learning', True)
        if is_learninig:
            return self.ll_estimator(args[0])
        else:
            return self.generator(args[0])


class ConjointDecoder(object):
    """
    A ConjointDecoder model takes multiple latent dimensions and from each generates a data output. There is also a
    shared latent input which is used by all decoders:

               Latent_1   Latent_shared   Latent_1
                  |       /           \      |
             -------------             -------------
             | Decoder_1 |             | Decoder_2 |
             -------------             -------------
                  |                         |
               Output_1                  Output_2


    For more details about the loss computation, see the StandardDecoder class.
    """

    def __init__(self, latent_dims, data_dims, network_architecture='synthetic'):
        """
        Args:
            latent_dims: tuple, the flattened dimensions of the private latent spaces and the shared latent space
            data_dims: int, the flattened dimensions of the output spaces (data spaces)
            network_architecture: str, the architecture name for the body of all the decoders
        """
        assert len(latent_dims) == len(data_dims) + 1, \
            "Expected too receive {} private latent spaces and one shared for {} data inputs " \
            "but got {} instead.".format(len(data_dims) + 1, len(data_dims), len(latent_dims))

        name = "Conjoint Decoder"
        logger.info("Initialising {} model with {}-dimensional outputs "
                    "and {}-dimensional latent inputs.".format(name, data_dims, latent_dims))

        def bernoulli_log_probs(args):
            from tensorflow.contrib.distributions import Bernoulli
            mu, x = args
            log_px = Bernoulli(probs=mu, name='dec_bernoulli').log_prob(x)
            return log_px

        latent_inputs, data_inputs, sampling_outputs, log_probs = [], [], [], []
        for i in range(len(data_dims)):
            latent_input = Input(shape=(latent_dims[i],), name='dec_latent_input_{}'.format(i))
            generator_body = get_network_by_name['decoder'][network_architecture](latent_input)
            sampler_params = Dense(data_dims[i], activation='sigmoid',
                                   name='dec_sampler_params_{}'.format(i))(generator_body)
            # probability clipping is necessary for the Bernoulli `log_prob` property produces NaNs in the limit cases
            sampler_params = Lambda(lambda x: 1e-6 + (1 - 2e-6) * x,
                                    name='dec_probs_clipper_{}'.format(i))(sampler_params)
            data_input = Input(shape=(data_dims[i],), name='dec_data_input_{}'.format(i))
            log_prob = Lambda(bernoulli_log_probs,
                              name='dec_bernoulli_logprob_{}'.format(i))([sampler_params, data_input])
            latent_inputs.append(latent_input)
            data_inputs.append(data_input)
            sampling_outputs.append(sampler_params)
            log_probs.append(log_prob)

        self.generator = Model(inputs=latent_inputs, outputs=sampling_outputs, name='dec_sampling')
        self.ll_estimator = Model(inputs=[data_inputs, latent_inputs], outputs=log_probs, name='dec_trainable')

    def __call__(self, *args, **kwargs):
        """
        Make the StandardDecoder model callable on lists of Input layers or tensors.

        Args:
            *args: a list of input layers or tensors or numpy arrays, or a single input layer, tensor or numpy array.
        Keyword Args:
            is_learning: bool, whether the model is used for training or data generation. The output is either
                the reconstruction log likelihood or the output probabilities in the data space respectively.

        Returns:
            A StandardDecoder model in `training` or `data generation` mode.
        """
        is_learninig = kwargs.get('is_learning', True)
        if is_learninig:
            return self.ll_estimator(args[0])
        else:
            return self.generator(args[0])