from src import model_torch
from src import model_original
from src.diffusion import GaussianDiffusion, DDIM_Sampler
from src.inferencer import Inferencer
import yaml
import argparse


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    unet_cfg = config['unet']
    ddim_cfg = config['ddim']
    trainer_cfg = config['inferencer']
    image_size = unet_cfg['image_size']

    if config['type'] == 'original':
        unet = model_original.Unet(**unet_cfg).to(args.device)
    elif config['type'] == 'torch':
        unet = model_torch.Unet(**unet_cfg).to(args.device)
    else:
        unet = None
        print("Unet type must be one of ['original', 'torch']")
        exit()

    diffusion = GaussianDiffusion(unet, image_size=image_size).to(args.device)

    ddim_samplers = list()
    if isinstance(ddim_cfg, dict):
        for sampler_cfg in ddim_cfg.values():
            ddim_samplers.append(DDIM_Sampler(diffusion, **sampler_cfg))

    inferencer = Inferencer(diffusion, ddim_samplers=ddim_samplers, time_step=diffusion.time_step, **trainer_cfg)
    inferencer.load(args.load)
    inferencer.inference()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='DDPM & DDIM')
    parse.add_argument('-c', '--config', type=str, default='./config/inference/cifar10.yaml')
    parse.add_argument('-l', '--load', type=str, default=None)
    parse.add_argument('-d', '--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    args = parse.parse_args()
    main(args)
