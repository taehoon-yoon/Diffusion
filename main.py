from src.model import Unet
from src.trainer import Trainer
from src.diffusion import GaussianDiffusion, DDIM_Sampler


def main():
    model = Unet(dim=32, dim_multiply=(1, 2, 4, 8), device='cuda').to('cuda')
    diffusion = GaussianDiffusion(model, image_size=32).to('cuda')

    ddim_samplers=list()
    ddim_samplers.append(DDIM_Sampler(diffusion, ddim_sampling_steps=100, sample_every=500, calculate_fid=True,
                                      num_fid_sample=5000, save=True))
    ddim_samplers.append(DDIM_Sampler(diffusion, ddim_sampling_steps=300, sample_every=700, calculate_fid=True,
                                      num_fid_sample=2000, save=True))

    trainer = Trainer(diffusion, 'cifar10', batch_size=128, lr=2e-4, clip='both', total_step=500000,
                      ddim_samplers=ddim_samplers, save_and_sample_every=1000, fid_estimate_batch_size=128)
    trainer.train()


if __name__ == '__main__':
    main()
