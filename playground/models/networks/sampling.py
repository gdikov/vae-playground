from keras.layers import Dense, Concatenate, Add, Lambda, Reshape
from keras import backend as K
from numpy import newaxis


def outer_product(inputs):
    """
    inputs: list of two tensors (of equal dimensions,
        for which you need to compute the outer product
    """
    x, y = inputs
    feature_size = K.shape(x)[1] ** 2
    outer_Product = x[:, :, newaxis] * y[:, newaxis, :]
    outer_Product = K.reshape(outer_Product, (-1, feature_size))
    # returns a flattened batch-wise set of tensors
    return outer_Product


def sample_standard_normal_noise(inputs, **kwargs):
    from keras.backend import shape, random_normal
    n_samples = kwargs.get('n_samples', shape(inputs)[0])
    n_basis_noise_vectors = kwargs.get('n_basis', -1)
    data_dim = kwargs.get('data_dim', 1)
    noise_dim = kwargs.get('noise_dim', data_dim)
    seed = kwargs.get('seed', 7)

    if n_basis_noise_vectors > 0:
        samples_isotropic = random_normal(shape=(n_samples, n_basis_noise_vectors, noise_dim),
                                          mean=0, stddev=1, seed=seed)
    else:
        samples_isotropic = random_normal(shape=(n_samples, noise_dim),
                                          mean=0, stddev=1, seed=seed)
    op_mode = kwargs.get('mode', 'none')
    if op_mode == 'concatenate':
        concat = Concatenate(axis=1, name='enc_noise_concatenation')([inputs, samples_isotropic])
        return concat
    elif op_mode == 'add':
        resized_noise = Dense(data_dim, activation=None, name='enc_resized_noise_sampler')(samples_isotropic)
        added_noise_data = Add(name='enc_adding_noise_data')([inputs, resized_noise])
        return added_noise_data
    elif op_mode == 'product':
        outputshape = noise_dim**2
        bilinearProduct = Lambda(outer_product, output_shape=(noise_dim ** 2, ))([inputs, samples_isotropic])
        # bilinearProduct = Reshape((outputshape),bilinearProduct)
        return bilinearProduct
    return samples_isotropic


def sample_adaptive_normal_noise(inputs, **kwargs):
    from keras.backend import shape, random_normal, sqrt

    seed = kwargs.get('seed', 7)
    latent_dim = kwargs.get('latent_dim', 2)

    if isinstance(inputs, list):
        mu, sigma2 = inputs
        n_samples = kwargs.get('n_samples', shape(mu)[0])
        samples_isotropic = random_normal(shape=(n_samples, latent_dim),
                                          mean=0, stddev=1, seed=seed)
        samples = mu + sqrt(sigma2) * samples_isotropic
        return samples
    else:
        samples_isotropic = random_normal(shape=(shape(inputs)[0], latent_dim),
                                          mean=0, stddev=1, seed=seed)
        return samples_isotropic
