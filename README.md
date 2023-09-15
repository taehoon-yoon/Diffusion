# DDPM & DDIM PyTorch
### DDPM & DDIM re-implementation with various 

This code is the implementation code of the papers DDPM ([Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)) 
and DDIM ([Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)).

---
## Objective

Our code is mainly based on [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
repository, which is a Pytorch implementation of the official Tensorflow code
[Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion). But we find that there are some
differences in the implementation especially in **U-net** structure. And further we find that Pytorch implementation 
version lacks some functionality for monitoring training and does not have inference code. So we decided to
re-implement the code such that it can be helpful for someone who is first to **Diffusion** models.

---
## Training

To train the diffusion model, first thing you have to do is to configure your training settings by making configuration
file. You can find some example inside the folder ```./config```. I will explain how to configure your training using
```./config/cifar10_example.yaml``` file.

In side the ```cifar10_example.yaml``` you may find 4 primary section, ```type, unet, ddim, trainer```. We will first 
look at ```trainer``` section which is configured as follows.

```yaml
dataset: cifar10
batch_size: 128
lr: 0.0002
total_step: 600000
save_and_sample_every: 2500
num_samples: 64
fid_estimate_batch_size: 128
ddpm_fid_score_estimate_every: null
ddpm_num_fid_samples: null
tensorboard: true
clip: both
```
- dataset: You can give Dataset name which is available by ```torchvision.datasets```. You can find some Datasets provided
by torchvision in this [website](https://pytorch.org/vision/stable/datasets.html). If you want to use torchvision's
Dataset just provide the dataset name, for example ```cifar10```. Currently tested datasets are ```cifar10```. 
Or if you want to use custom datasets, 


- batch_size, lr, total_step: You can find the values used by DDPM author in the [DDPM paper](https://arxiv.org/abs/2006.11239)
