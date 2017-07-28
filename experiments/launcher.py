
def limit_gpu_memory_allocation(fraction=0.5):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))


def run_from_config(config, limit_gpu_mem_fraction=1, **kwargs):
    if limit_gpu_mem_fraction < 1:
        limit_gpu_memory_allocation(fraction=limit_gpu_mem_fraction)
    raise NotImplementedError


def run_from_code(experiment, limit_gpu_mem_fraction=1, **kwargs):
    if limit_gpu_mem_fraction < 1:
        # this should be called before any other tensorflow/keras import statement
        limit_gpu_memory_allocation(fraction=limit_gpu_mem_fraction)
    return experiment(**kwargs)
