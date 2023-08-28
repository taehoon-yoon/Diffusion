from src.model import Unet
from src.trainer import Trainer
from src.diffusion import GaussianDiffusion


def main():
    model = Unet(dim=32, dim_multiply=(1, 2, 4, 8), device='cuda').to('cuda')
    diffusion = GaussianDiffusion(model, image_size=32).to('cuda')
    trainer = Trainer(diffusion, 'cifar10', batch_size=128, lr=2e-4, clip=False)
    trainer.train()


if __name__ == '__main__':
    main()
