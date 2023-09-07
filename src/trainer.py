import os
import math
from .dataset import dataset_wrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, diffusion_model, dataset, batch_size=32, lr=2e-5, total_step=100000, save_and_sample_every=1000,
                 num_samples=25, result_folder='./results', cpu_percentage=0, calculate_fid=True, num_fid_samples=None,
                 ema_decay=0.9999, ema_update_every=10, max_grad_norm=1., tensorboard=True, exp_name=None, clip=True):
        now = datetime.datetime.now()
        cur_time = now.strftime('%Y-%m-%d_%Hh%Mm')
        if exp_name is None:
            exp_name = os.path.basename(dataset)
            if exp_name == '':
                exp_name = os.path.basename(os.path.dirname(dataset))
        self.diffusion_model = diffusion_model
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.image_size = self.diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.result_folder = os.path.join(result_folder, exp_name, cur_time)
        self.device = self.diffusion_model.device
        self.clip = clip
        self.is_ddim_sampling = self.diffusion_model.is_ddim_sampling
        self.calculate_fid = calculate_fid
        self.global_step = 0

        notification = make_notification('Dataset', color='light_green')
        print(notification)
        dataSet = dataset_wrapper(dataset, self.image_size)
        assert len(dataSet) >= 100, 'you should have at least 100 images in your folder.at least 10k images recommended'
        print(colored('Dataset Length: {}\n'.format(len(dataSet)), 'light_green'))
        CPU_cnt = cpu_count()
        # TODO: pin_memory?
        dataLoader = DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=CPU_cnt * cpu_percentage)
        self.dataLoader = cycle(dataLoader) if os.path.isdir(dataset) else cycle_with_label(dataLoader)
        self.optimizer = Adam(self.diffusion_model.parameters(), lr=lr)

        self.ema = EMA(self.diffusion_model, beta=ema_decay, update_every=ema_update_every).to(self.device)
        self.sampler = self.ema.ema_model.ddim_sample if self.is_ddim_sampling else self.ema.ema_model.sample

        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(os.path.join(self.result_folder, 'clip'), exist_ok=True)
        os.makedirs(os.path.join(self.result_folder, 'no_clip'), exist_ok=True)
        self.total_step = total_step

        self.num_fid_samples = len(dataSet) if num_fid_samples is None else num_fid_samples
        if self.calculate_fid:
            if not self.is_ddim_sampling:
                notification = make_notification('WARNING', color='red', boundary='*')
                print(notification)
                print(colored("Robust FID computation requires a lot of generated samples and "
                              "can therefore be very time consuming.", 'red'))
                print('To accelerate sampling, DDIM sampling method is recommended. To enable DDIM sampling,'
                      'set [ddim_sampling_steps] parameter to some value while instantiating diffusion model.\n')
            self.fid_scorer = FID(batch_size, self.dataLoader, sampler=self.sampler,
                                  dataset_name=exp_name, num_samples=self.num_fid_samples)

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
        best_fid = 1e10
        stepTQDM = tqdm(range(self.global_step, self.total_step))
        for cur_step in stepTQDM:
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
            if cur_step != 0 and (cur_step % self.save_and_sample_every) == 0:
                self.ema.ema_model.eval()

                with torch.inference_mode():
                    milestone = cur_step // self.save_and_sample_every
                    batches = num_to_groups(self.num_samples, self.batch_size)
                    img_list = list(map(lambda n: self.ema.ema_model.ddim_sample(batch_size=n, clip=True), batches))
                    img_no_clip = list(map(lambda n: self.ema.ema_model.ddim_sample(batch_size=n, clip=False), batches))

                    img_list2 = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, clip=True), batches))
                    img_no_clip2 = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, clip=False), batches))

                all_images = torch.cat(img_list, dim=0)
                all_images_no_clip = torch.cat(img_no_clip, dim=0)

                all_images2 = torch.cat(img_list2, dim=0)
                all_images_no_clip2 = torch.cat(img_no_clip2, dim=0)

                print(colored('-'*50, 'red'))
                print(all_images.min(), all_images.max())
                print(all_images_no_clip.min(), all_images_no_clip.max())
                print(all_images2.min(), all_images2.max())
                print(all_images_no_clip2.min(), all_images_no_clip2.max())
                print(colored('-' * 50, 'red'))

                torchvision.utils.save_image(all_images, nrow=int(math.sqrt(self.num_samples)),
                                             fp=os.path.join(self.result_folder, 'clip', f'sample_{milestone}_ddim.png'))
                torchvision.utils.save_image(all_images_no_clip, nrow=int(math.sqrt(self.num_samples)),
                                             fp=os.path.join(self.result_folder, 'no_clip', f'sample_{milestone}_ddim.png'))

                torchvision.utils.save_image(all_images2, nrow=int(math.sqrt(self.num_samples)),
                                             fp=os.path.join(self.result_folder, 'clip', f'sample_{milestone}_ddpm.png'))
                torchvision.utils.save_image(all_images_no_clip2, nrow=int(math.sqrt(self.num_samples)),
                                             fp=os.path.join(self.result_folder, 'no_clip', f'sample_{milestone}_ddpm.png'))

                if self.writer is not None:
                    all_images2.clamp_(0.0, 1.0)
                    self.writer.add_images('sampling result', all_images2, cur_step)

                if self.calculate_fid:
                    cur_fid = self.fid_scorer.fid_score()
                    if best_fid > cur_fid:
                        best_fid = cur_fid
                        self.save('best_fid')
                    if self.writer is not None:
                        self.writer.add_scalar('FID', cur_fid, cur_step)
                self.save('latest')
                self.ema.ema_model.train()

        print(colored('Training Finished!', 'light_yellow'))
        if self.writer is not None:
            self.writer.close()

    def save(self, name):
        data = {
            'global_step': self.global_step,
            'model': self.diffusion_model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'ema': self.ema.state_dict()
        }
        torch.save(data, os.path.join(self.result_folder, 'model_{}.pt'.format(name)))

    def load(self, name):
        data = torch.load(os.path.join(self.result_folder, 'model_{}.pt'.format(name)), map_location=self.device)
        self.diffusion_model.load_state_dict(data['model'])
        self.global_step = data['global_step']
        self.optimizer.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
