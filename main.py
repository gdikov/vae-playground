from experiments.launcher import run_from_code
from experiments.conjoint_experiments import synthetic


if __name__ == '__main__':
    run_from_code(experiment=synthetic, model='avb', pretrained_model=None, noise_mode='product')
