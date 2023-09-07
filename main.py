from src.model import Unet
from src.trainer import Trainer
from src.diffusion import GaussianDiffusion


def main():
    model = Unet(dim=32, dim_multiply=(1, 2, 4, 8), device='cuda').to('cuda')
    diffusion = GaussianDiffusion(model, image_size=32, ddim_sampling_steps=100, eta=0).to('cuda')
    trainer = Trainer(diffusion, 'cifar10', batch_size=128, lr=2e-4, clip=False, total_step=500000,
                      calculate_fid=False, save_and_sample_every=500)
    trainer.train()


if __name__ == '__main__':
    main()
