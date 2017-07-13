

# evt. set the output_shape arg.
def slice_vector(x, start=0, stop=None):
    if stop is None:
        return x[:, start:]
    else:
        return x[:, start:stop]
