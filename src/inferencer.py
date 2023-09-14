import os
import math
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .dataset import dataset_wrapper
from .utils import *
from termcolor import colored
from tqdm import tqdm
from glob import glob
import imageio
from imageio import mimsave
from functools import partial


class Inferencer:
    def __init__(self, diffusion_model, dataset, ddim_samplers=None, batch_size=32, num_samples_per_image=25,
                 result_folder='./inference_results', num_images_to_generate=1, ddpm_fid_estimate=True, time_step=1000,
                 ddpm_num_fid_samples=None, clip=True, return_all_step=True, make_denoising_gif=True, num_gif=50):
        """
        Inferenceer for Diffusion model. Sampling is supported by DDPM sampling & DDIM sampling
        :param diffusion_model: GaussianDiffusion model
        :param dataset: either 'cifar10' or path to the custom dataset you've prepared, where images are saved
        :param ddim_samplers: List containing DDIM samplers.
        :param batch_size: batch_size for inferencing
        :param num_samples_per_image: # of generating images, must be square number ex) 25, 36, 49...
        :param result_folder: where inference result will be saved.
        :param num_images_to_generate: # of generated image set. For example if num_samples_per_image==25 and
        num_images_to_generate==3 then, in result folder there will be 3 generated image with each image containing
        25 generated sub-images merged into one image file with 5 rows, 5 columns.
        :param ddpm_fid_estimate: Whether to  calculate FID score based on DDPM sampling.
        :param time_step: Gaussian diffusion length T. In DDPM paper they used T=1000
        :param ddpm_num_fid_samples: # of generating images for FID calculation using DDPM sampler. If you set
        ddpm_fid_estimate to False, i.e. not using DDPM sampler for FID calculation, then this value will
        be just ignored.
        :param clip: [True, False, 'both'] you can find detail in p_sample function
        and ddim_p_sample function in diffusion.py file.
        :param return_all_step: Whether to save the entire de-noising processed image to result folder.
        :param make_denoising_gif: Whether to make gif which contains de-noising process visually.
        :param num_gif: # of images to make one gif which contains de-noising process visually.
        """
        dataset_name = os.path.basename(dataset)
        if dataset_name == '':
            dataset_name=os.path.basename(os.path.dirname(dataset))
        self.diffusion_model = diffusion_model
        self.ddim_samplers = ddim_samplers
        self.batch_size = batch_size
        self.time_step = time_step
        self.num_samples = num_samples_per_image
        self.num_images = num_images_to_generate
        self.nrow = int(math.sqrt(self.num_samples))
        assert (self.nrow ** 2) == self.num_samples, 'num_samples must be a square number. ex) 25, 36, 49, ...'
        self.image_size = self.diffusion_model.image_size
        self.result_folder = os.path.join(result_folder, dataset_name)
        self.ddpm_result_folder = os.path.join(self.result_folder, 'DDPM')
        self.device = self.diffusion_model.device
        self.return_all_step = return_all_step or make_denoising_gif
        self.make_denoising_gif = make_denoising_gif
        self.num_gif = num_gif
        self.clip = clip
        self.ddpm_fid_flag = ddpm_fid_estimate
        self.cal_fid = True if self.ddpm_fid_flag else False
        self.fid_score_log = dict()
        assert clip in [True, False, 'both'], "clip must be one of [True, False, 'both']"
        if clip is True or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'clip'), exist_ok=True)
        if clip is False or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'no_clip'), exist_ok=True)

        # Dataset
        notification = make_notification('Dataset', color='light_green')
        print(notification)
        dataSet = dataset_wrapper(dataset, self.image_size, augment_horizontal_flip=False)
        dataLoader = DataLoader(dataSet, batch_size=batch_size)
        print(colored('Dataset Length: {}\n'.format(len(dataSet)), 'light_green'))

        # DDIM sampler setting
        for idx, sampler in enumerate(self.ddim_samplers):
            sampler.sampler_name = 'DDIM_{}_steps{}_eta{}'.format(idx + 1, sampler.ddim_steps, sampler.eta)
            save_path = os.path.join(self.result_folder, sampler.sampler_name)
            sampler.save_path = save_path
            if sampler.generate_image:
                if sampler.clip is True or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'clip'), exist_ok=True)
                if sampler.clip is False or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'no_clip'), exist_ok=True)
            if sampler.calculate_fid:
                self.cal_fid = True
                sampler.num_fid_sample = sampler.num_fid_sample if sampler.num_fid_sample is not None else len(dataSet)

        # Image generation log
        notification = make_notification('Image Generation', color='light_cyan')
        print(notification)
        print(colored('Image will be generated with the following sampler(s)', 'light_cyan'))
        print(colored('-> DDPM Sampler', 'light_cyan'))
        for sampler in self.ddim_samplers:
            if sampler.generate_image:
                print(colored('-> {}'.format(sampler.sampler_name), 'light_cyan'))
        print('\n')

        # FID score
        notification = make_notification('FID', color='light_magenta')
        print(notification)
        if not self.cal_fid:
            print(colored('No FID evaluation will be executed!\n'
                          'If you want FID evaluation consider using DDIM sampler.', 'light_magenta'))
        else:
            self.fid_scorer = FID(self.batch_size, dataLoader, dataset_name=dataset_name, device=self.device,
                                  no_label=os.path.isdir(dataset))
            print(colored('FID score will be calculated with the following sampler(s)', 'light_magenta'))
            if self.ddpm_fid_flag:
                self.ddpm_num_fid_samples = ddpm_num_fid_samples if ddpm_num_fid_samples is not None else len(dataSet)
                print(colored('-> DDPM Sampler / FID calculation with {} generated samples'
                              .format(self.ddpm_num_fid_samples), 'light_magenta'))
            for sampler in self.ddim_samplers:
                if sampler.calculate_fid:
                    print(colored('-> {} / FID calculation with {} generated samples'
                                  .format(sampler.sampler_name, sampler.num_fid_sample), 'light_magenta'))
            print('\n')
        del dataset
        del dataLoader

    @torch.inference_mode()
    def inference(self):
        self.diffusion_model.eval()
        notification = make_notification('Inferencing', color='light_yellow', boundary='+')
        print(notification)
        print(colored('Image Generation\n', 'light_yellow'))

        # DDPM sampler
        for idx in tqdm(range(self.num_images), desc='DDPM image sampling'):
            batches = num_to_groups(self.num_samples, self.batch_size)
            for i, j in zip([True, False], ['clip', 'no_clip']):
                if self.clip not in [i, 'both']:
                    continue
                imgs = list(map(lambda n: self.diffusion_model.sample(n, self.return_all_step, clip=i), batches))
                imgs = torch.cat(imgs, dim=0)  # (batch, steps, ch, h, w)
                if self.return_all_step:
                    path = os.path.join(self.ddpm_result_folder, '{}'.format(j), '{}'.format(idx + 1))
                    os.makedirs(path, exist_ok=True)
                    for step in range(imgs.shape[1]):
                        save_image(imgs[:, step], nrow=self.nrow, fp=os.path.join(path, '{:04d}.png'.format(step)))
                    if self.make_denoising_gif:
                        gif_step = int(self.time_step/self.num_gif)
                        gif_step = max(1, gif_step)
                        file_names = list(reversed(sorted(glob(os.path.join(path, '*.png')))))[::gif_step]
                        file_names = reversed(file_names)
                        gif = [imageio.v2.imread(names) for names in file_names]
                        mimsave(os.path.join(self.ddpm_result_folder, '{}_{}.gif'.format(idx+1, j)),
                                gif, **{'duration': self.num_gif / imgs.shape[1]})
                last_img = imgs[:, -1] if self.return_all_step else imgs
                save_image(last_img, nrow=self.nrow,
                           fp=os.path.join(self.ddpm_result_folder, '{}_{}.png'.format(idx+1, j)))
        # DDIM sampler
        for sampler in self.ddim_samplers:
            if sampler.generate_image:
                for idx in tqdm(range(self.num_images), desc='{} image sampling'.format(sampler.sampler_name)):
                    batches = num_to_groups(self.num_samples, self.batch_size)
                    for i, j in zip([True, False], ['clip', 'no_clip']):
                        if sampler.clip not in [i, 'both']:
                            continue
                        imgs = list(map(lambda n: sampler.sample(n, None, self.return_all_step, clip=i), batches))
                        imgs = torch.cat(imgs, dim=0)
                        if self.return_all_step:
                            path = os.path.join(sampler.save_path, '{}'.format(j), '{}'.format(idx + 1))
                            os.makedirs(path, exist_ok=True)
                            for step in range(imgs.shape[1]):
                                save_image(imgs[:, step], nrow=self.nrow,
                                           fp=os.path.join(path, '{:04d}.png'.format(step)))
                            if self.make_denoising_gif:
                                gif_step = int(sampler.ddim_steps/self.num_gif)
                                gif_step = max(1, gif_step)
                                file_names = list(reversed(sorted(glob(os.path.join(path, '*.png')))))[::gif_step]
                                file_names = reversed(file_names)
                                gif = [imageio.v2.imread(names) for names in file_names]
                                mimsave(os.path.join(sampler.save_path, '{}_{}.gif'.format(idx + 1, j)),
                                        gif, **{'duration': self.num_gif / imgs.shape[1]})
                        last_img = imgs[:, -1] if self.return_all_step else imgs
                        save_image(last_img, nrow=self.nrow,
                                   fp=os.path.join(sampler.save_path, '{}_{}.png'.format(idx + 1, j)))
        if self.cal_fid:
            print(colored('\nFID score estimation\n', 'light_yellow'))
            if self.ddpm_fid_flag:
                print(colored('DDPM FID calculation...', 'yellow'))
                ddpm_fid = self.fid_scorer.fid_score(self.diffusion_model.sample, self.ddpm_num_fid_samples)
                self.fid_score_log['DDPM'] = ddpm_fid
            for sampler in self.ddim_samplers:
                print(colored('{} FID calculation...'.format(sampler.sampler_name), 'yellow'))
                if sampler.calculate_fid:
                    sample_ = partial(sampler.sample, self.diffusion_model)
                    ddim_fid = self.fid_scorer.fid_score(sample_, sampler.num_fid_sample)
                    self.fid_score_log[f'{sampler.sampler_name}'] = ddim_fid
            print(colored('-'*50, 'yellow'))
            for key, val in self.fid_score_log.items():
                print(colored('Sampler: {} -> FID score: {}'.format(key, val), 'yellow'))
            with open(os.path.join(self.result_folder, 'FID.txt'), 'w') as f:
                f.write('Results\n')
                f.write('='*50)
                f.write('\n')
                for key, val in self.fid_score_log.items():
                    f.write('Sampler: {} -> FID score: {}\n'.format(key, val))

    def load(self, path):
        if not os.path.exists(path):
            print(make_notification('ERROR', color='red', boundary='*'))
            print(colored('No saved checkpoint is detected. Please check you gave existing path!', 'red'))
            exit()
        print(make_notification('Loading Checkpoint', color='green'))
        data = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(data['model'])
        print(colored('Successfully loaded checkpoint!\n', 'green'))