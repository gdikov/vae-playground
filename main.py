from experiments.launcher import run_from_code
from experiments import mnist_variations, synthetic, change_background_save_latent, change_background_generate_digit

if __name__ == '__main__':
    # run_from_code(experiment=synthetic, model='avb', pretrained_model=None, noise_mode='product')
    # run_from_code(experiment=mnist_variations, n_datasets=2, model='avb', pretrained_model=None)
    # run_from_code(experiment=change_background_save_latent, model='vae', pretrained_model = "output", bg_same_target = False)
    run_from_code(experiment=change_background_generate_digit, model='vae', pretrained_model="output")#, bg = 'trippy')
