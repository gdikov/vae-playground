from keras import backend as ker


# evt. set the output_shape arg.
def slice_vector(x, start=0, stop=None):
    if stop is None:
        return x[:, start:]
    else:
        return x[:, start:stop]


def compute_outer_product(inputs):
    """
    Compute the outer product between two 2-d tensors.

    Args:
        inputs: list of two tensors (of equal dimensions, for which the outer product is computed
    """
    x, y = inputs
    feature_size = ker.shape(x)[1] ** 2
    outer_product = x[:, :, None] * y[:, None, :]
    outer_product = ker.reshape(outer_product, (-1, feature_size))
    # returns a flattened batch-wise set of tensors
    return outer_product
