import os
import math
import numpy as np
from .dataset import dataset_wrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from multiprocessing import cpu_count
from functools import partial
from tqdm import tqdm
import datetime
from termcolor import colored
from .utils import *


def cycle_with_label(dl):
    while True:
        for data in dl:
            img, label = data
            yield img


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer:
    def __init__(self, diffusion_model, dataset, batch_size=32, lr=2e-5, total_step=100000, ddim_samplers=None,
                 save_and_sample_every=1000, num_samples=25, result_folder='./results', cpu_percentage=0,
                 fid_estimate_batch_size=None, ddpm_fid_score_estimate_every=None, ddpm_num_fid_samples=None,
                 max_grad_norm=1., tensorboard=True, exp_name=None, clip=True):
        """
        Trainer for Diffusion model.
        :param diffusion_model: GaussianDiffusion model
        :param dataset: either 'cifar10' or path to the custom dataset you've prepared, where images are saved
        :param batch_size: batch size for training. DDPM author used 128 for cifar10 and 64 for 256X256 image
        :param lr: DDPM author used 2e-4 for cifar10 and 2e-5 for 256X256 image
        :param total_step: total training step. DDPM used 800K for cifar10, CelebA-HQ for 0.5M
        :param ddim_samplers: List containing DDIM samplers.
        :param save_and_sample_every: Step interval for saving model and generated image(by DDPM sampling).
        For example if it is set to 1000, then trainer will save models in every 1000 step and save generated images
        based on DDPM sampling schema. If you want to generate image based on DDIM sampling, you have to pass a list
        containing corresponding DDIM sampler.
        :param num_samples: # of generating images, must be square number ex) 25, 36, 49...
        :param result_folder: where model, generated images will be saved
        :param cpu_percentage: The percentage of CPU used for Dataloader i.e. num_workers in Dataloader.
        Value must be [0, 1] where 1 means using all cpu for dataloader. If you are Windows user setting value other
        than 0 will cause problem, so set to 0
        :param fid_estimate_batch_size: batch size for FID calculation. It has nothing to do with training.
        :param ddpm_fid_score_estimate_every: Step interval for FID calculation using DDPM. If set to None, FID score
        will not be calculated with DDPM sampling. If you use DDPM sampling for FID calculation, it can be very
        time consuming, so it is wise to set this value to None, and use DDIM sampler for FID calculation. But anyway
        you can calculate FID score with DDPM sampler if you insist to.
        :param ddpm_num_fid_samples: # of generating images for FID calculation using DDPM sampler. If you set
        ddpm_fid_score_estimate_every to None, i.e. not using DDPM sampler for FID calculation, then this value will
        be just ignored.
        :param max_grad_norm: Restrict the norm of maximum gradient to this value
        :param tensorboard: Set to ture if you want to monitor training
        :param exp_name: experiment name. If set to None, it will be decided automatically as folder name of dataset.
        :param clip: [True, False, 'both'] you can find detail in p_sample function in diffusion.py file.
        """

        # Metadata & Initialization & Make directory for saving files.
        now = datetime.datetime.now()
        self.cur_time = now.strftime('%Y-%m-%d_%Hh%Mm')
        if exp_name is None:
            exp_name = os.path.basename(dataset)
            if exp_name == '':
                exp_name = os.path.basename(os.path.dirname(dataset))
        self.exp_name = exp_name
        self.diffusion_model = diffusion_model
        self.ddim_samplers = ddim_samplers
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.nrow = int(math.sqrt(self.num_samples))
        assert (self.nrow ** 2) == self.num_samples, 'num_samples must be a square number. ex) 25, 36, 49, ...'
        self.save_and_sample_every = save_and_sample_every
        self.image_size = self.diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.result_folder = os.path.join(result_folder, exp_name, self.cur_time)
        self.ddpm_result_folder = os.path.join(self.result_folder, 'DDPM')
        self.device = self.diffusion_model.device
        self.clip = clip
        self.ddpm_fid_flag = True if ddpm_fid_score_estimate_every is not None else False
        self.ddpm_fid_score_estimate_every = ddpm_fid_score_estimate_every
        self.cal_fid = True if self.ddpm_fid_flag else False
        self.tqdm_sampler_name = None
        self.tensorboard = tensorboard
        self.tensorboard_name = None
        self.writer = None
        self.global_step = 0
        self.total_step = total_step
        self.fid_score_log = dict()
        assert clip in [True, False, 'both'], "clip must be one of [True, False, 'both']"
        if clip is True or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'clip'), exist_ok=True)
        if clip is False or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'no_clip'), exist_ok=True)

        # Dataset & DataLoader & Optimizer
        notification = make_notification('Dataset', color='light_green')
        print(notification)
        dataSet = dataset_wrapper(dataset, self.image_size)
        assert len(dataSet) >= 100, 'you should have at least 100 images in your folder.at least 10k images recommended'
        print(colored('Dataset Length: {}\n'.format(len(dataSet)), 'light_green'))
        CPU_cnt = cpu_count()
        # TODO: pin_memory?
        num_workers = int(CPU_cnt * cpu_percentage)
        assert num_workers <= CPU_cnt, "cpu_percentage must be [0.0, 1.0]"
        dataLoader = DataLoader(dataSet, batch_size=self.batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
        self.dataLoader = cycle(dataLoader) if os.path.isdir(dataset) else cycle_with_label(dataLoader)
        self.optimizer = Adam(self.diffusion_model.parameters(), lr=lr)

        # DDIM sampler setting
        self.ddim_sampling_schedule = list()
        for idx, sampler in enumerate(self.ddim_samplers):
            sampler.sampler_name = 'DDIM_{}_steps{}_eta{}'.format(idx + 1, sampler.ddim_steps, sampler.eta)
            self.ddim_sampling_schedule.append(sampler.sample_every)
            save_path = os.path.join(self.result_folder, sampler.sampler_name)
            sampler.save_path = save_path
            if sampler.save:
                os.makedirs(save_path, exist_ok=True)
            if sampler.generate_image:
                if sampler.clip is True or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'clip'), exist_ok=True)
                if sampler.clip is False or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'no_clip'), exist_ok=True)
            if sampler.calculate_fid:
                self.cal_fid = True
                if self.tqdm_sampler_name is None:
                    self.tqdm_sampler_name = sampler.sampler_name
                sampler.num_fid_sample = sampler.num_fid_sample if sampler.num_fid_sample is not None else len(dataSet)
                self.fid_score_log[sampler.sampler_name] = list()
            if sampler.fixed_noise:
                sampler.register_buffer('noise', torch.randn([self.num_samples, sampler.channel,
                                                              sampler.image_size, sampler.image_size]))

        # Image generation log
        notification = make_notification('Image Generation', color='light_cyan')
        print(notification)
        print(colored('Image will be generated with the following sampler(s)', 'light_cyan'))
        print(colored('-> DDPM Sampler / Image generation every {} steps'.format(self.save_and_sample_every),
                      'light_cyan'))
        for sampler in self.ddim_samplers:
            if sampler.generate_image:
                print(colored('-> {} / Image generation every {} steps / Fixed Noise : {}'
                              .format(sampler.sampler_name, sampler.sample_every, sampler.fixed_noise), 'light_cyan'))
        print('\n')

        # FID score
        notification = make_notification('FID', color='light_magenta')
        print(notification)
        if not self.cal_fid:
            print(colored('No FID evaluation will be executed!\n'
                          'If you want FID evaluation consider using DDIM sampler.', 'light_magenta'))
        else:
            self.fid_batch_size = fid_estimate_batch_size if fid_estimate_batch_size is not None else self.batch_size
            dataSet_fid = dataset_wrapper(dataset, self.image_size,
                                          augment_horizontal_flip=False, info_color='light_magenta', min1to1=False)
            dataLoader_fid = DataLoader(dataSet_fid, batch_size=self.fid_batch_size, num_workers=num_workers)

            self.fid_scorer = FID(self.fid_batch_size, dataLoader_fid, dataset_name=exp_name, device=self.device,
                                  no_label=os.path.isdir(dataset))

            print(colored('FID score will be calculated with the following sampler(s)', 'light_magenta'))
            if self.ddpm_fid_flag:
                self.ddpm_num_fid_samples = ddpm_num_fid_samples if ddpm_num_fid_samples is not None else len(dataSet)
                print(colored('-> DDPM Sampler / FID calculation every {} steps with {} generated samples'
                              .format(self.ddpm_fid_score_estimate_every, self.ddpm_num_fid_samples), 'light_magenta'))
            for sampler in self.ddim_samplers:
                if sampler.calculate_fid:
                    print(colored('-> {} / FID calculation every {} steps with {} generated samples'
                                  .format(sampler.sampler_name, sampler.sample_every,
                                          sampler.num_fid_sample), 'light_magenta'))
            print('\n')
            if self.ddpm_fid_flag:
                self.tqdm_sampler_name = 'DDPM'
                self.fid_score_log['DDPM'] = list()
                notification = make_notification('WARNING', color='red', boundary='*')
                print(notification)
                msg = """
                FID computation witm DDPM sampler requires a lot of generated samples and can therefore be very time 
                consuming.\nTo accelerate sampling, only using DDIM sampling is recommended. To disable DDPM sampling,
                set [ddpm_fid_score_estimate_every] parameter to None while instantiating Trainer.\n
                """
                print(colored(msg, 'red'))
            del dataLoader_fid
            del dataSet_fid

    def train(self):
        # Tensorboard
        if self.tensorboard:
            os.makedirs('./tensorboard', exist_ok=True)
            self.tensorboard_name = self.exp_name + '_' + self.cur_time \
                if self.tensorboard_name is None else self.tensorboard_name
            notification = make_notification('Tensorboard', color='light_blue')
            print(notification)
            print(colored('Tensorboard Available!', 'light_blue'))
            print(colored('Tensorboard name: {}'.format(self.tensorboard_name), 'light_blue'))
            print(colored('Launch Tensorboard by running following command on terminal', 'light_blue'))
            print(colored('tensorboard --logdir ./tensorboard\n', 'light_blue'))
            self.writer = SummaryWriter(os.path.join('./tensorboard', self.tensorboard_name))
        notification = make_notification('Training', color='light_yellow', boundary='+')
        print(notification)
        cur_fid = 'NAN'
        ddpm_best_fid = 1e10
        stepTQDM = tqdm(range(self.global_step, self.total_step))
        for cur_step in stepTQDM:
            self.diffusion_model.train()
            self.optimizer.zero_grad()
            image = next(self.dataLoader).to(self.device)
            loss = self.diffusion_model(image)
            loss.backward()
            nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            vis_fid = cur_fid if isinstance(cur_fid, str) else '{:.04f}'.format(cur_fid)
            stepTQDM.set_postfix({'loss': '{:.04f}'.format(loss.item()), 'FID': vis_fid, 'step':self.global_step})

            self.diffusion_model.eval()
            # DDPM Sampler for generating images
            if cur_step != 0 and (cur_step % self.save_and_sample_every) == 0:
                if self.writer is not None:
                    self.writer.add_scalar('Loss', loss.item(), cur_step)
                with torch.inference_mode():
                    batches = num_to_groups(self.num_samples, self.batch_size)
                    for i, j in zip([True, False], ['clip', 'no_clip']):
                        if self.clip not in [i, 'both']:
                            continue
                        imgs = list(map(lambda n: self.diffusion_model.sample(batch_size=n, clip=i), batches))
                        imgs = torch.cat(imgs, dim=0)
                        save_image(imgs, nrow=self.nrow,
                                   fp=os.path.join(self.ddpm_result_folder, j, f'sample_{cur_step}.png'))
                        if self.writer is not None:
                            self.writer.add_images('DDPM sampling result ({})'.format(j), imgs, cur_step)
                self.save('latest')

            # DDPM Sampler for FID score evaluation
            if self.ddpm_fid_flag and cur_step != 0 and (cur_step % self.ddpm_fid_score_estimate_every) == 0:
                ddpm_cur_fid, _ = self.fid_scorer.fid_score(self.diffusion_model.sample, self.ddpm_num_fid_samples)
                if ddpm_best_fid > ddpm_cur_fid:
                    ddpm_best_fid = ddpm_cur_fid
                    self.save('best_fid_ddpm')
                if self.writer is not None:
                    self.writer.add_scalars('FID', {'DDPM': ddpm_cur_fid}, cur_step)
                cur_fid = ddpm_cur_fid
                self.fid_score_log['DDPM'].append((self.global_step, ddpm_cur_fid))

            # DDIM Sampler
            for sampler in self.ddim_samplers:
                if cur_step != 0 and (cur_step % sampler.sample_every) == 0:
                    # DDPM Sampler for generating images
                    if sampler.generate_image:
                        with torch.inference_mode():
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            c_batch = np.insert(np.cumsum(np.array(batches)), 0, 0)
                            for i, j in zip([True, False], ['clip', 'no_clip']):
                                if sampler.clip not in [i, 'both']:
                                    continue
                                if sampler.fixed_noise:
                                    imgs = list()
                                    for b in range(len(batches)):
                                        imgs.append(sampler.sample(self.diffusion_model, batch_size=None, clip=i,
                                                                   noise=sampler.noise[c_batch[b]:c_batch[b+1]]))
                                else:
                                    imgs = list(map(lambda n: sampler.sample(self.diffusion_model,
                                                                             batch_size=n, clip=i), batches))
                                imgs = torch.cat(imgs, dim=0)
                                save_image(imgs, nrow=self.nrow,
                                           fp=os.path.join(sampler.save_path, j, f'sample_{cur_step}.png'))
                                if self.writer is not None:
                                    self.writer.add_images('{} sampling result ({})'
                                                           .format(sampler.sampler_name, j), imgs, cur_step)

                    # DDPM Sampler for FID score evaluation
                    if sampler.calculate_fid:
                        sample_ = partial(sampler.sample, self.diffusion_model)
                        ddim_cur_fid, _ = self.fid_scorer.fid_score(sample_, sampler.num_fid_sample)
                        if sampler.best_fid[0] > ddim_cur_fid:
                            sampler.best_fid[0] = ddim_cur_fid
                            if sampler.save:
                                self.save('best_fid_{}'.format(sampler.sampler_name))
                        if sampler.sampler_name == self.tqdm_sampler_name:
                            cur_fid = ddim_cur_fid
                        if self.writer is not None:
                            self.writer.add_scalars('FID', {sampler.sampler_name: ddim_cur_fid}, cur_step)
                        self.fid_score_log[sampler.sampler_name].append((self.global_step, ddim_cur_fid))

            self.global_step += 1

        print(colored('Training Finished!', 'light_yellow'))
        if self.writer is not None:
            self.writer.close()

    def save(self, name):
        data = {
            'global_step': self.global_step,
            'model': self.diffusion_model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'fid_logger': self.fid_score_log,
            'tensorboard': self.tensorboard_name
        }
        for sampler in self.ddim_samplers:
            data[sampler.sampler_name] = sampler.state_dict()
        torch.save(data, os.path.join(self.result_folder, 'model_{}.pt'.format(name)))

    def load(self, path, tensorboard_path=None, no_prev_ddim_setting=False):
        if not os.path.exists(path):
            print(make_notification('ERROR', color='red', boundary='*'))
            print(colored('No saved checkpoint is detected. Please check you gave existing path!', 'red'))
            exit()
        if tensorboard_path is not None and not os.path.exists(tensorboard_path):
            print(make_notification('ERROR', color='red', boundary='*'))
            print(colored('No tensorboard is detected. Please check you gave existing path!', 'red'))
            exit()
        print(make_notification('Loading Checkpoint', color='green'))
        data = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(data['model'])
        self.global_step = data['global_step']
        self.optimizer.load_state_dict(data['opt'])
        fid_score_log = data['fid_logger']
        if no_prev_ddim_setting:
            for key, val in self.fid_score_log.items():
                if key not in fid_score_log:
                    fid_score_log[key] = val
        else:
            for sampler in self.ddim_samplers:
                sampler.load_state_dict(data[sampler.sampler_name])
        self.fid_score_log = fid_score_log
        if tensorboard_path is not None:
            self.tensorboard_name = data['tensorboard']
        print(colored('Successfully loaded checkpoint!\n', 'green'))
