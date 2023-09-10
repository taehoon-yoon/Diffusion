from src import model_torch
from src import model_original
from src.trainer import Trainer
from src.diffusion import GaussianDiffusion, DDIM_Sampler
import yaml
import argparse


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    unet_cfg = config['unet']
    ddim_cfg = config['ddim']
    trainer_cfg = config['trainer']
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
    for sampler_cfg in ddim_cfg.values():
        ddim_samplers.append(DDIM_Sampler(diffusion, **sampler_cfg))

    trainer = Trainer(diffusion, ddim_samplers=ddim_samplers, **trainer_cfg)
    if args.load is not None:
        trainer.load(args.load, args.tensorboard)
    trainer.train()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='DDPM & DDIM')
    parse.add_argument('-c', '--config', type=str, default='./config/cifar10.yaml')
    parse.add_argument('-d', '--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parse.add_argument('-l', '--load', type=str, default=None)
    parse.add_argument('-t', '--tensorboard', type=str, default=None)
    args = parse.parse_args()

    data = {
        'type': 'original',
        'unet': {
            'dim': 128,
            'image_size': 32,
            'dim_multiply': [1, 2, 2, 2],
            'attn_resolutions': [16, ],
            'dropout': 0.1,
            'num_res_blocks': 2
        },
        'ddim': {
            0: {
                'ddim_sampling_steps': 20,
                'sample_every': 1000,
                'calculate_fid': True,
                'num_fid_sample': 60000,
                'save': True
            },
            1: {
                'ddim_sampling_steps': 100,
                'sample_every': 5000,
                'calculate_fid': True,
                'num_fid_sample': 6000,
                'save': True
            },
            2: {
                'ddim_sampling_steps': 500,
                'sample_every': 10000,
                'calculate_fid': True,
                'num_fid_sample': 6000,
                'save': True
            }
        },
        'trainer': {
            'dataset': 'cifar10',
            'batch_size': 128,
            'lr': 2e-4,
            'clip': 'both',
            'total_step': 600000,
            'save_and_sample_every': 1000,
            'fid_estimate_batch_size': 128,
            'num_samples': 64
        }

    }
    # with open('./config/cifar10.yaml', 'w') as f:
    #     yaml.dump(data, f, sort_keys=False)

    main(args)
