from experiments.launcher import run_from_code
from experiments import mnist_variations, synthetic


if __name__ == '__main__':
    # run_from_code(experiment=synthetic, model='avb', pretrained_model=None, noise_mode='product')
    run_from_code(experiment=mnist_variations, n_datasets=2, model='avb', pretrained_model=None)
