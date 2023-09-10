import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from termcolor import colored
import ssl
import os
from glob import glob
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context


class customDataset(Dataset):
    def __init__(self, folder, transform, exts=['jpg', 'jpeg', 'png', 'tiff']):
        super().__init__()
        self.paths = [p for ext in exts for p in glob(os.path.join(folder, f'*.{ext}'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img = Image.open(self.paths[item])
        return self.transform(img)


def dataset_wrapper(dataset, image_size, augment_horizontal_flip=True, info_color='light_green'):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip() if augment_horizontal_flip else torch.nn.Identity(),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # turn into torch Tensor of shape CHW, 0 ~ 1
        transforms.Lambda(lambda x: ((x * 2) - 1))  # -1 ~ 1
    ])
    if os.path.isdir(dataset):
        print(colored('Loading local file directory', info_color))
        dataSet = customDataset(dataset, transform)
        print(colored('Successfully loaded {} images!'.format(len(dataSet)), info_color))
        return dataSet
    else:
        dataset = dataset.lower()
        assert dataset in ['cifar10']
        print(colored('Loading {} dataset'.format(dataset), info_color))
        if dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            fullset = torch.utils.data.ConcatDataset([trainset, testset])
            return fullset
