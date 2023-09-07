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


def cycle_with_label(dl):
    while True:
        for data in dl:
            img, label = data
            yield img


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Trainer:
    def __init__(self, diffusion_model, dataset, batch_size=32, lr=2e-5, total_step=100000, save_and_sample_every=1000,
                 num_samples=25, result_folder='./results', cpu_percentage=0,
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

        dataSet = dataset_wrapper(dataset, self.image_size)
        assert len(dataSet) >= 100, 'you should have at least 100 images in your folder.at least 10k images recommended'
        print('Dataset Length: {}'.format(len(dataSet)))
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

        self.writer = None
        if tensorboard:
            os.makedirs('./tensorboard', exist_ok=True)
            tensorboard_name = exp_name + '_' + cur_time
            print('Tensorboard name: {}'.format(tensorboard_name))
            self.writer = SummaryWriter(os.path.join('./tensorboard', tensorboard_name))

    def train(self):
        stepTQDM = tqdm(range(self.total_step))
        for cur_step in stepTQDM:
            self.optimizer.zero_grad()
            image = next(self.dataLoader).to(self.device)
            loss = self.diffusion_model(image)
            loss.backward()
            nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.ema.update()

            stepTQDM.set_postfix({'loss': '{:.04f}'.format(loss.item())})
            if self.writer is not None:
                self.writer.add_scalar('Loss', loss.item(), cur_step)
            if cur_step != 0 and (cur_step % self.save_and_sample_every) == 0:
                self.ema.ema_model.eval()

                with torch.inference_mode():
                    milestone = cur_step // self.save_and_sample_every
                    batches = num_to_groups(self.num_samples, self.batch_size)
                    all_images_list = list(map(lambda n: self.sampler(batch_size=n, clip=True), batches))
                    all_images_list_no_clip = list(map(lambda n: self.sampler(batch_size=n, clip=False), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images_no_clip = torch.cat(all_images_list_no_clip, dim=0)
                torchvision.utils.save_image(all_images, nrow=int(math.sqrt(self.num_samples)),
                                             fp=os.path.join(self.result_folder, 'clip', f'sample_{milestone}.png'))
                torchvision.utils.save_image(all_images_no_clip, nrow=int(math.sqrt(self.num_samples)),
                                             fp=os.path.join(self.result_folder, 'no_clip', f'sample_{milestone}.png'))
