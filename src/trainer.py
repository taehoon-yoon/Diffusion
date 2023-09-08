import os
import math
from .dataset import dataset_wrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from multiprocessing import cpu_count
from ema_pytorch import EMA
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
                 ema_decay=0.9999, ema_update_every=10, max_grad_norm=1., tensorboard=True, exp_name=None, clip=True):

        # Metadata & Initialization & Make directory for saving files.
        now = datetime.datetime.now()
        cur_time = now.strftime('%Y-%m-%d_%Hh%Mm')
        if exp_name is None:
            exp_name = os.path.basename(dataset)
            if exp_name == '':
                exp_name = os.path.basename(os.path.dirname(dataset))
        self.diffusion_model = diffusion_model
        self.ddim_samplers = ddim_samplers
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.nrow = int(math.sqrt(self.num_samples))
        assert (self.nrow ** 2) == self.num_samples, 'num_samples must be a square number. ex) 25, 36, 49, ...'
        self.save_and_sample_every = save_and_sample_every
        self.image_size = self.diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.result_folder = os.path.join(result_folder, exp_name, cur_time)
        self.ddpm_result_folder = os.path.join(self.result_folder, 'DDPM')
        self.device = self.diffusion_model.device
        self.clip = clip
        self.ddpm_fid_flag = True if ddpm_fid_score_estimate_every is not None else False
        self.ddpm_fid_score_estimate_every = ddpm_fid_score_estimate_every
        self.cal_fid = True if self.ddpm_fid_flag else False
        self.tqdm_sampler_name = None
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
        dataLoader = DataLoader(dataSet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.dataLoader = cycle(dataLoader) if os.path.isdir(dataset) else cycle_with_label(dataLoader)
        self.optimizer = Adam(self.diffusion_model.parameters(), lr=lr)
        self.ema = EMA(self.diffusion_model, beta=ema_decay, update_every=ema_update_every).to(self.device)

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

        # FID score
        notification = make_notification('FID', color='light_magenta')
        print(notification)
        if not self.cal_fid:
            print(colored('No FID evaluation will be executed!\n'
                          'If you want FID evaluation consider using DDIM sampler.', 'light_magenta'))
        else:
            self.fid_batch_size = fid_estimate_batch_size if fid_estimate_batch_size is not None else self.batch_size
            dataSet_fid = dataset_wrapper(dataset, self.image_size,
                                          augment_horizontal_flip=False, info_color='light_magenta')
            dataLoader_fid = DataLoader(dataSet_fid, batch_size=self.fid_batch_size, num_workers=num_workers)

            self.fid_scorer = FID(self.fid_batch_size, dataLoader_fid, dataset_name=exp_name, device=self.device)

            print(colored('FID score will be calculated with following sampler(s)', 'light_magenta'))
            if self.ddpm_fid_flag:
                self.ddpm_num_fid_samples = ddpm_num_fid_samples if ddpm_num_fid_samples is not None else len(dataSet)
                print(colored('- DDPM Sampler / FID calculation every {} steps with {} generated samples\n'
                              .format(self.ddpm_fid_score_estimate_every, self.ddpm_num_fid_samples), 'light_magenta'))
            for sampler in self.ddim_samplers:
                if sampler.calculate_fid:
                    print(colored('- {} / FID calculation every {} steps with {} generated samples'
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

        # Tensorboard
        self.writer = None
        if tensorboard:
            os.makedirs('./tensorboard', exist_ok=True)
            tensorboard_name = exp_name + '_' + cur_time
            notification = make_notification('Tensorboard', color='light_blue')
            print(notification)
            print(colored('Tensorboard Available!', 'light_blue'))
            print(colored('Tensorboard name: {}'.format(tensorboard_name), 'light_blue'))
            print(colored('Launch Tensorboard by running following command on terminal', 'light_blue'))
            print(colored('tensorboard --logdir ./tensorboard\n', 'light_blue'))
            self.writer = SummaryWriter(os.path.join('./tensorboard', tensorboard_name))

    def train(self):
        notification = make_notification('Training', color='light_yellow', boundary='+')
        print(notification)
        cur_fid = 'NAN'
        ddpm_best_fid = 1e10
        stepTQDM = tqdm(range(self.global_step, self.total_step))
        for cur_step in stepTQDM:
            self.ema.ema_model.train()
            self.optimizer.zero_grad()
            image = next(self.dataLoader).to(self.device)
            loss = self.diffusion_model(image)
            loss.backward()
            nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.ema.update()

            vis_fid = cur_fid if isinstance(cur_fid, str) else '{:.04f}'.format(cur_fid)
            stepTQDM.set_postfix({'loss': '{:.04f}'.format(loss.item()), 'FID': vis_fid})
            if self.writer is not None:
                self.writer.add_scalar('Loss', loss.item(), cur_step)

            self.ema.ema_model.eval()
            # DDPM Sampler for generating images
            if cur_step != 0 and (cur_step % self.save_and_sample_every) == 0:
                with torch.inference_mode():
                    batches = num_to_groups(self.num_samples, self.batch_size)
                    if self.clip is True or self.clip == 'both':
                        imgs = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, clip=True), batches))
                        imgs = torch.cat(imgs, dim=0)
                        save_image(imgs, nrow=self.nrow,
                                   fp=os.path.join(self.ddpm_result_folder, 'clip', f'sample_{cur_step}.png'))
                        if self.writer is not None:
                            self.writer.add_images('DDPM sampling result (clip)', imgs, cur_step)
                    if self.clip is False or self.clip == 'both':
                        imgs_no_clip = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, clip=False), batches))
                        imgs_no_clip = torch.cat(imgs_no_clip, dim=0)
                        save_image(imgs_no_clip, nrow=self.nrow,
                                   fp=os.path.join(self.ddpm_result_folder, 'no_clip', f'sample_{cur_step}.png'))
                        if self.writer is not None:
                            self.writer.add_images('DDPM sampling result (no clip)', imgs_no_clip, cur_step)
                self.save('latest')

            # DDPM Sampler for FID score evaluation
            if self.ddpm_fid_flag and cur_step != 0 and (cur_step % self.ddpm_fid_score_estimate_every) == 0:
                ddpm_cur_fid = self.fid_scorer.fid_score(self.diffusion_model.sample, self.ddpm_num_fid_samples)
                if ddpm_best_fid > ddpm_cur_fid:
                    ddpm_best_fid = ddpm_cur_fid
                    self.save('best_fid_ddpm')
                if self.writer is not None:
                    self.writer.add_scalars('FID', {'DDPM': ddpm_cur_fid}, cur_step)
                cur_fid = ddpm_cur_fid
                self.fid_score_log['DDPM'].append(ddpm_cur_fid)

            # DDIM Sampler
            for sampler in self.ddim_samplers:
                if cur_step != 0 and (cur_step % sampler.sample_every) == 0:
                    # DDPM Sampler for generating images
                    if sampler.generate_image:
                        with torch.inference_mode():
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            if sampler.clip is True or sampler.clip == 'both':
                                imgs = list(map(lambda n: sampler.sample(batch_size=n, clip=True), batches))
                                imgs = torch.cat(imgs, dim=0)
                                save_image(imgs, nrow=self.nrow,
                                           fp=os.path.join(sampler.save_path, 'clip', f'sample_{cur_step}.png'))
                                if self.writer is not None:
                                    self.writer.add_images('{} sampling result (clip)'
                                                           .format(sampler.sampler_name), imgs, cur_step)
                            if sampler.clip is False or sampler.clip == 'both':
                                imgs_no_clip = list(map(lambda n: sampler.sample(batch_size=n, clip=False), batches))
                                imgs_no_clip = torch.cat(imgs_no_clip, dim=0)
                                save_image(imgs_no_clip, nrow=self.nrow,
                                           fp=os.path.join(sampler.save_path, 'no_clip', f'sample_{cur_step}.png'))
                                if self.writer is not None:
                                    self.writer.add_images('{} sampling result (no clip)'
                                                           .format(sampler.sampler_name), imgs_no_clip, cur_step)

                    # DDPM Sampler for FID score evaluation
                    if sampler.calculate_fid:
                        ddim_cur_fid = self.fid_scorer.fid_score(sampler.sample, sampler.num_fid_sample)
                        if sampler.best_fid[0] > ddim_cur_fid:
                            sampler.best_fid[0] = ddim_cur_fid
                            if sampler.save:
                                self.save('best_fid_{}'.format(sampler.sampler_name))
                        if sampler.sampler_name == self.tqdm_sampler_name:
                            cur_fid = ddim_cur_fid
                        if self.writer is not None:
                            self.writer.add_scalars('FID', {sampler.sampler_name: ddim_cur_fid}, cur_step)
                        self.fid_score_log[sampler.sampler_name].append(ddim_cur_fid)

            self.global_step += 1

        print(colored('Training Finished!', 'light_yellow'))
        if self.writer is not None:
            self.writer.close()

    def save(self, name):
        data = {
            'global_step': self.global_step,
            'model': self.diffusion_model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'fid_logger': self.fid_score_log
        }
        for sampler in self.ddim_samplers:
            data[sampler.sampler_name] = sampler.state_dict()
        torch.save(data, os.path.join(self.result_folder, 'model_{}.pt'.format(name)))

    def load(self, name):
        data = torch.load(os.path.join(self.result_folder, 'model_{}.pt'.format(name)), map_location=self.device)
        self.diffusion_model.load_state_dict(data['model'])
        self.global_step = data['global_step']
        self.optimizer.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.fid_score_log = data['fid_logger']
        for sampler in self.ddim_samplers:
            sampler.load_state_dict(data[sampler.sampler_name])
