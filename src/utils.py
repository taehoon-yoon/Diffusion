import math
import torch
import torch.nn as nn
import numpy as np
import os
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from termcolor import colored


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        half_d_model = d_model // 2
        log_denominator = -math.log(10000) / (half_d_model - 1)
        denominator_ = torch.exp(torch.arange(half_d_model) * log_denominator)
        self.register_buffer('denominator', denominator_)

    def forward(self, time):
        """
        :param time: shape=(B, )
        :return: Positional Encoding shape=(B, d_model)
        """
        argument = time[:, None] * self.denominator[None, :]  # (B, half_d_model)
        return torch.cat([argument.sin(), argument.cos()], dim=-1)  # (B, d_model)


class FID:
    def __init__(self, batch_size, dataLoader, sampler, dataset_name, cache_dir='./results/fid_cache/',
                 num_samples=None, device='cuda'):
        self.batch_size = batch_size
        self.dataLoader = dataLoader
        self.sampler = sampler
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.device = device
        self.inception = InceptionV3([3]).to(device)

        os.makedirs(cache_dir, exist_ok=True)
        notification = make_notification('FID', color='light_magenta')
        print(notification)
        self.m2, self.s2 = self.load_dataset_stats()

    def calculate_inception_features(self, samples):
        self.inception.eval()
        features = self.inception(samples)[0]
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        return features.squeeze()

    def load_dataset_stats(self):
        path = os.path.join(self.cache_dir, self.dataset_name + '.npz')
        if os.path.exists(path):
            with np.load(path) as f:
                m2, s2 = f['m2'], f['s2']
            print(colored('Successfully loaded pre-computed Inception feature from cached file\n', 'light_magenta'))
        else:
            num_batches = int(math.ceil(self.num_samples / self.batch_size))
            stacked_real_features = list()
            print(colored('Computing Inception features for {} '
                          'samples from real dataset.'.format(num_batches * self.batch_size), 'light_magenta'))
            for _ in tqdm(range(num_batches), desc='Calculating stats for data distribution', leave=False):
                real_samples = next(self.dataLoader).to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = torch.cat(stacked_real_features, dim=0).cpu().numpy()
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            print(colored('Dataset stats cached to {} for future use\n'.format(path), 'light_magenta'))
        return m2, s2

    @torch.inference_mode()
    def fid_score(self):
        batches = num_to_groups(self.num_samples, self.batch_size)
        stacked_fake_features = list()
        for batch in tqdm(batches, desc='FID score calculation', leave=False):
            fake_samples = self.sampler(batch, clip=True)
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, dim=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)
        return calculate_frechet_distance(m1, s1, self.m2, self.s2)


def make_notification(content, color, boundary='-'):
    notice = boundary * 30 + '\n'
    side = boundary if boundary != '-' else '|'
    notice += '{}{:^28}{}\n'.format(side, content, side)
    notice += boundary * 30
    return colored(notice, color)
