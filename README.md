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
```./config/cifar10_example.yaml``` file and ```./config/cifar10_torch_example.yaml``` file.
Inside the ```cifar10_example.yaml``` you may find 4 primary section, ```type, unet, ddim, trainer```. 

We will first look at ```trainer``` section which is configured as follows.

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
- ```dataset```: You can give Dataset name which is available by ```torchvision.datasets```. You can find some Datasets provided
by torchvision in this [website](https://pytorch.org/vision/stable/datasets.html). If you want to use torchvision's
Dataset just provide the dataset name, for example ```cifar10```. Currently, tested datasets are ```cifar10```. 
Or if you want to use custom datasets which you have prepared, you have to pass the path to the folder which is containing
images 


- ```batch_size, lr, total_step```: You can find the values used by DDPM author in the [DDPM paper](https://arxiv.org/abs/2006.11239)
Appendix B. total_step means total training step, for example DDPM author trained cifar10 model with 800K steps.


- ```save_and_sample_every```: The interval to which save the model and generated samples. For example in this case, 
for every 2500 steps trainer will save the model weights and also generate some sample images to visualize the training progress.


- ```num_samples```: When sampling the images evey ```save_and_sample_every``` steps, trainer will sample total ```num_samples``` 
images and save it to one large image containing each sampled images where one large image have (num_samples)**0.5 rows and
columns. So ```num_samples``` must be square number ex) 25, 36, 49, 64, ...


- ```fid_estimate_batch_size```: Batch size for sampling images for FID calculation. This batch size will be applied to
DDPM sampler as well as DDIM samplers. If you cannot decide the value, just setting this value equal to ```batch_size``` will
be fine.


- ```ddpm_fid_score_estimate_every```: Step interval for FID calculation using DDPM sampler. If set to null, FID score
will not be calculated with DDPM sampling. If you use DDPM sampling for FID calculation, i.e. setting this value other than null, 
it can be very time-consuming, ***so it is wise to set this value to null***, and use DDIM sampler for 
FID calculation (Using DDIM sampler is explained below). But anyway you can calculate FID score with DDPM sampler if you **insist**.


- ```ddpm_num_fid_samples```: Number of sampling images for FID calculation using DDPM sampler. If you set 
```ddpm_fid_score_estimate_every``` to null, i.e. not using DDPM sampler for FID calculation, then this value will
be just ignored.


- ```tensorboard```: If set to true, then you can monitor training progress such as loss, FID value, and sampled images,
during training, with the tensorboard.


- ```clip```: It must be one of [both, true, false]. This is related to sampling of x_{t-1} from p_{theta}(x_{t-1} | x_t).
There are two ways to sample x_{t-1}.
One way is to follow paper and this corresponds to line 4 in Algorithm 2 in DDPM paper. (```clip==False```)
Another way is to clip(or clamp) the predicted x_0 to -1 ~ 1 for better sampling result.
To clip the x_0 to out desired range, we cannot directly apply (11) to sample x_{t-1}, rather we have to
calculate predicted x_0 using (4) and then calculate mu in (7) using that predicted x_0. Which is exactly
same calculation except for clipping.
As you might easily expect, using clip leads to better sampling result since it
restricts sampled images range to -1 ~ 1. So for the better sampling result, it is strongly suggested 
setting ```clip``` to true. If ```clip==both``` then sampling is done twice, one done with
```clip==True``` and the other done with ```clip==False```.
[Reference](https://github.com/hojonathanho/diffusion/issues/5)

---

Now we will look at ```type, unet``` section which is configured as follows.

```yaml
# ```./config/cifar10_example.yaml```
type: original
unet:
  dim: 64
  image_size: 32
  dim_multiply:
  - 1
  - 2
  - 2
  - 2
  attn_resolutions:
  - 16
  dropout: 0.1
  num_res_blocks: 2
```

```yaml
# ```./config/cifar10_torch_example.yaml```
type: torch
unet:
  dim: 64
  image_size: 32
  dim_multiply:
  - 1
  - 2
  - 2
  - 2
  full_attn:
  - false
  - true
  - false
  - false
  attn_heads: 4
  attn_head_dim: 32
```

-```type```: It must be one of [original, torch]. ***original*** will use U-net structure which was originally suggested by
Jonathan Ho. So it's structure will be the one used in [Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
which is an official version written in Tensorflow. ***torch*** will use U-net structure which was suggested by 
[denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) which is a transcribed version of
official Tensorflow version. 

I have separated those two because there structure differs significantly. To name a few, following is the difference of
those two U-net structure.
1. official version use self Attention where the feature map resolution at each U-net level 
   is in ```attn_resolutions```. In the DDPM paper you can find that they used self Attention at the 16X16 resolution, 
   and this is why ```attn_resolutions``` is by default ```[16, ]```

   On the other hand, Pytorch transcribed version use Linear Attention and multi-head self Attention. They use
   multi-head self Attention at the U-net level where ```full_attn``` is true and Linear Attention at the 
   rest of the U-net level. So in this particular case, they used multi-head self Attention at the U-net level 1 (I will 
denote U-net level as 0, 1, 2, 3, ...) and the Linear Attention at the U-net level 0, 2, 3.

-```unet.dim```: This is related to the hidden channel dimension of feature map at each U-net model. You can find more
detail right below.

-```unet.dim_multiply```: ```len(dim_multiply)``` will be the depth of U-net model with at each level i, the dimension
        of channel will be ```dim * dim_multiply[i]```. If the input image shape is [H, W, 3] then at the lowest level,
        feature map shape will be [H/(2^(len(dim_multiply)-1), W/(2^(len(dim_multiply)-1), dim*dim_multiply[-1]]
        if not considering U-net down-up path connection.

-```unet.image_size```: Size of the input image. Image will be resized and cropped if ```image_size``` does not
equal to actual input image size. Expected to be ```Integer```, I have not tested for non-square images.

-```unet.attn_resolution / unet.full_attn```: Explained above. Since ```attn_resolution``` value 
must equal to the resolution value of feature map where you want to apply self Attention, you have to carefully 
calculate desired resolution value. In the case of ```full_attn```, it is related to applying particular Attention mechanism at each
level, it must satisfy ```len(full_attn) == len(dim_multiply)```

-```unet.num_res_blocks```: Number of ResnetBlock at each level. In downward path, at each level, there will be
num_res_blocks amount of ResnetBlock module and in upward path, at each level, there will be
(num_res_blocks+1) amount of ResnetBlock module.

-```unet.attn_heads, unet.attn_head_dim```: In the torch implementation it uses multi-head-self-Attention. attn_head is 
the # of head. It corresponds to h in "Attention is all you need" paper. See section 3.2.2
attn_head_dim is the dimension of each head. It corresponds to d_k in Attention paper.

---

Lastly we will look at ```ddim``` section which is configured as follows.


```yaml
ddim:
  0:
    ddim_sampling_steps: 20
    sample_every: 5000
    calculate_fid: true
    num_fid_sample: 6000
    save: true
  1:
    ddim_sampling_steps: 50
    sample_every: 50000
    calculate_fid: true
    num_fid_sample: 6000
    save: true
  2:
    ddim_sampling_steps: 20
    sample_every: 100000
    calculate_fid: true
    num_fid_sample: 60000
    save: true
```

There are 3 subsection (0, 1, 2) which means it will use 3 DDIM Sampler for sampling image 
and FID calculation during training. The name of each subsection, which are 0, 1, 2 in this case
is not important. 

-```ddim_sampling_steps```: The number of de-noising steps for DDIM sampling. In DDPM they used 1000 steps for sampling images. But
in DDIM we can control the total number of de-noising steps for generating images. If this value is set to 20,
then the speed of generating image will be 50 times faster than DDPM sampling. Preferred value would be 10~100.

-```sample_every```: This control how often sampling be done with particular sampler. If set to 5000 then
every 5000 steps, this sampler will be activated for sampling images. So if total training step is 50000, there will be 
total 10 sampling action.

-```calculata_fid```: Whether to calculate FID 

-```num_fid_sample```: Only valid if ```calculate_fid``` is set to true. This control how many sampled images to use for 
FID calculation. The speed of FID calculation for particular sampler will be inversely 
proportional to (ddim_sanmpling_steps * num_fid_sample)

-```save```: Whether to save the model based on FID value calculated by particular sampler. If set to true
then model parameter and all the information to resume training will be saved on ```.pt``` file when model achieve best
FID value for particular sampler.

---

Now we are finished setting configuration file and the training the model can be done by following command.

```commandline
python train.py -c /path_to_config_file/configuration_file.yaml 
```