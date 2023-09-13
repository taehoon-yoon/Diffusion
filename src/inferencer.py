import os
import math
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .dataset import dataset_wrapper
from .utils import *
from termcolor import colored
from tqdm import tqdm
from glob import glob
import imageio
from imageio import mimsave


class Inferencer:
    def __init__(self, diffusion_model, dataset, ddim_samplers=None, batch_size=32, num_samples_per_image=25,
                 result_folder='./inference_results', num_images_to_generate=1, ddpm_fid_estimate=True,
                 ddpm_num_fid_samples=None, clip=True, return_all_step=True, make_denoising_gif=True, gif_duration=3):
        dataset_name = os.path.basename(dataset)
        if dataset_name == '':
            dataset_name=os.path.basename(os.path.dirname(dataset))
        self.diffusion_model = diffusion_model
        self.ddim_samplers = ddim_samplers
        self.batch_size = batch_size
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
        self.gif_duration = gif_duration
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
                    sampler.num_fid_sample = sampler.num_fid_sample if sampler.num_fid_sample is not None else len(
                        dataSet)
                    self.fid_score_log[sampler.sampler_name] = list()

        # Image generation log
        notification = make_notification('Image Generation', color='light_cyan')
        print(notification)
        print(colored('Image will be generated with the following sampler(s)', 'light_cyan'))
        print(colored('-> DDPM Sampler', 'light_cyan'))
        for sampler in self.ddim_samplers:
            if sampler.generate_image:
                print(colored('-> {}', 'light_cyan'))
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
        notification = make_notification('Inferencing', color='light_yellow', boundary='+')
        print(notification)
        print(colored('Image Generation', 'light_yellow'))

        # DDPM sampler
        for idx in tqdm(range(self.num_images), desc='DDPM image sampling'):
            batches = num_to_groups(self.num_samples, self.batch_size)
            for i, j in zip([True, False], ['clip', 'no_clip']):
                if self.clip not in [i, 'both']:
                    continue
                imgs = list(map(lambda n: self.diffusion_model.sample(batch_size=n, clip=i,
                                                                      return_all_timestep=self.return_all_step), batches))
                imgs = torch.cat(imgs, dim=0)  # (batch, steps, ch, h, w)
                if self.return_all_step:
                    path = os.path.join(self.ddpm_result_folder, 'j', '{}'.format(idx + 1))
                    os.makedirs(path)
                    for step in range(imgs.shape[1]):
                        save_image(imgs[:, step], nrow=self.nrow, fp=os.path.join(path, '{:04d}.png'.format(step)))
                    if self.make_denoising_gif:
                        file_names = sorted(glob(os.path.join(path, '*.png')))
                        gif = [imageio.v2.imread(names) for names in file_names]
                        mimsave('{}_{}.gif'.format(idx+1, j), gif, **{'duration': self.gif_duration/imgs.shape[1]})

