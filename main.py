from experiments.launcher import run_from_code
from experiments import mnist_variations, synthetic, mnist_background_transfer


if __name__ == '__main__':
    # run_from_code(experiment=synthetic, model='avb', pretrained_model=None, noise_mode='product')
    # run_from_code(experiment=mnist_variations, n_datasets=2, model='avb', pretrained_model=None)
    run_from_code(experiment=mnist_background_transfer, model='avb',
                  pretrained_model='output/conjoint_avb/mnist_variations_two/best')
