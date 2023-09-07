import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from termcolor import colored
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context


def dataset_wrapper(dataset, image_size, augment_horizontal_flip=True):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip() if augment_horizontal_flip else torch.nn.Identity(),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # turn into torch Tensor of shape CHW, 0 ~ 1
        transforms.Lambda(lambda x: ((x * 2) - 1))  # -1 ~ 1
    ])
    if os.path.isdir(dataset):
        print(colored('Loading local file directory', 'light_green'))
    else:
        dataset = dataset.lower()
        assert dataset in ['cifar10']
        print(colored('Loading {} dataset'.format(dataset), 'light_green'))
        if dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            fullset = torch.utils.data.ConcatDataset([trainset, testset])
            return fullset
