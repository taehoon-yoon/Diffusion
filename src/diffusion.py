import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, time_step=1000, loss_type='l2'):
        """
        Diffusion model. It is based on Denoising Diffusion Probabilistic Models (DDPM), Jonathan Ho et al.
        :param model: U-net model for de-noising network
        :param image_size: image size
        :param time_step: Gaussian diffusion length T. In paper they used T=1000
        :param loss_type: either l1, l2, huber. Default, l2
        """
        super().__init__()
        self.unet = model
        self.channel = self.unet.channel
        self.device = self.unet.device
        self.image_size = image_size
        self.time_step = time_step
        self.loss_type = loss_type

        beta = self.linear_beta_schedule()  # (t, )  t=time_step, in DDPM paper t=1000
        alpha = 1. - beta  # (a1, a2, a3, ... at)
        alpha_bar = torch.cumprod(alpha, dim=0)  # (a1, a1*a2, a1*a2*a3, ..., a1*a2*~*at)
        alpha_bar_prev = F.pad(alpha_bar[:-1], pad=(1, 0), value=1.)  # (1, a1, a1*a2, ..., a1*a2*~*a(t-1))

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)

        # calculation for q(x_t | x_0) consult (4) in DDPM paper.
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - alpha_bar))

        # calculation for q(x_{t-1} | x_t, x_0) consult (7) in DDPM paper.
        self.register_buffer('beta_tilde', beta * ((1. - alpha_bar_prev) / (1. - alpha_bar)))
        self.register_buffer('mean_tilde_x0_coeff', beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar))
        self.register_buffer('mean_tilde_xt_coeff', torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        # calculation for x0 consult (9) in DDPM paper.
        self.register_buffer('sqrt_recip_alpha_bar', torch.sqrt(1. / alpha_bar))
        self.register_buffer('sqrt_recip_alpha_bar_min_1', torch.sqrt(1. / alpha_bar - 1))

        # calculation for (11) in DDPM paper.
        self.register_buffer('sqrt_recip_alpha', torch.sqrt(1. / alpha))
        self.register_buffer('beta_over_sqrt_one_minus_alpha_bar', beta / torch.sqrt(1. - alpha_bar))

    # Forward Process / Diffusion Process ##############################################################################
    def q_sample(self, x0, t, noise):
        """
        Sampling x_t, according to q(x_t | x_0). Consult (4) in DDPM paper.
        :param x0: (b, c, h, w), original image
        :param t: (b, ), timestep t
        :param noise: (b, c, h, w), We calculate q(x_t | x_0) using re-parameterization trick.
        :return: x_t with shape=(b, c, h, w), which is a noised image at timestep t
        """
        # Get x_t ~ q(x_t | x_0) using re-parameterization trick
        return self.sqrt_alpha_bar[t][:, None, None, None] * x0 + \
               self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * noise

    def forward(self, img):
        """
        Calculate L_simple according to (14) in DDPM paper
        :param img: (b, c, h, w), original image
        :return: L_simple
        """
        b, c, h, w = img.shape
        assert h == self.image_size and w == self.image_size, f'height and width of image must be {self.image_size}'
        t = torch.randint(0, self.time_step, (b,), device=img.device).long()  # (b, )
        noise = torch.randn_like(img)  # corresponds to epsilon in (14)
        noised_image = self.q_sample(img, t, noise)  # argument inside epsilon_theta
        predicted_noise = self.unet(noised_image, t)  # epsilon_theta in (14)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    ####################################################################################################################

    # Reverse Process / De-noising Process #############################################################################
    @torch.inference_mode()
    def p_sample(self, xt, t, clip=True):
        """
        Sample x_{t-1} from p_{theta}(x_{t-1} | x_t).
        There are two ways to sample x_{t-1}.
        One way is to follow paper and this corresponds to line 4 in Algorithm 2 in DDPM paper. (clip==False)
        Another way is to clip(or clamp) the predicted x_0 to -1 ~ 1 for better sampling result.
        To clip the x_0 to out desired range, we cannot directly apply (11) to sample x_{t-1}, rather we have to
        calculate predicted x_0 using (4) and then calculate mu in (7) using that predicted x_0. Which is exactly
        same calculation except for clipping.
        As you might easily expect, using clip leads to better sampling result since it
        restricts sampled images range to -1 ~ 1. Ref: https://github.com/hojonathanho/diffusion/issues/5

        :param xt: ( b, c, h, w), noised image at time step t
        :param t: ( b, )
        :param clip: [True, False] Whether to clip predicted x_0 to our desired range -1 ~ 1.
        :return: de-noised image at time step t-1
        """
        batched_time = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
        pred_noise = self.unet(xt, batched_time)  # corresponds to epsilon_{theta}
        if clip:
            x0 = self.sqrt_recip_alpha_bar[t] * xt - self.sqrt_recip_alpha_bar_min_1[t] * pred_noise
            x0.clamp_(-1., 1.)
            mean = self.mean_tilde_x0_coeff[t] * x0 + self.mean_tilde_xt_coeff[t] * xt
        else:
            mean = self.sqrt_recip_alpha[t] * (xt - self.beta_over_sqrt_one_minus_alpha_bar[t] * pred_noise)
        variance = self.beta_tilde[t]
        noise = torch.randn_like(xt) if t > 0 else 0.  # corresponds to z, consult 4: in Algorithm 2.
        x_t_minus_1 = mean + torch.sqrt(variance) * noise
        return x_t_minus_1

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timestep=False, clip=True):
        """

        :param batch_size: # of image to generate.
        :param return_all_timestep: Whether to return all images during de-noising process. So it will return the
        images from time step T ~ time step 0
        :param clip: [True, False]. Explanation in p_sample function.
        :return: Generated image of shape (b, 3, h, w) if return_all_timestep==False else (b, T, 3, h, w)
        """
        xT = torch.randn([batch_size, self.channel, self.image_size, self.image_size], device=self.device)
        denoised_intermediates = [xT]
        xt = xT
        for t in tqdm(reversed(range(0, self.time_step)), desc='DDPM Sampling', total=self.time_step, leave=False):
            x_t_minus_1 = self.p_sample(xt, t, clip)
            denoised_intermediates.append(x_t_minus_1)
            xt = x_t_minus_1

        images = xt if not return_all_timestep else torch.stack(denoised_intermediates, dim=1)
        images = (images + 1.0) * 0.5  # scale to 0~1
        return images

    ####################################################################################################################

    def linear_beta_schedule(self):
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / self.time_step
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.time_step, dtype=torch.float32)


class DDIM_Sampler(nn.Module):
    def __init__(self, ddpm_diffusion_model, ddim_sampling_steps=100, eta=0, sample_every=5000, fixed_noise=False,
                 calculate_fid=False, num_fid_sample=None, generate_image=True, clip=True, save=False):
        """
        Denoising Diffusion Implicit Models (DDIM), Jiaming Song et al.
        :param ddpm_diffusion_model: DDPM diffusion model.
        :param ddim_sampling_steps: Total sampling steps for DDIM sampling process. It corresponds to S in DDIM paper.
        Consult section 4.2 Accelerated Generation Processes in DDIM paper.
        :param eta: Hyperparameter to control the stochasticity, see (16) in DDIM paper.
        0: deterministic(DDIM) , 1: fully stochastic(DDPM)
        :param sample_every: The interval for calling this DDIM sampler. It is only valid during training.
        For example if sample_every=5000 then during training, every 5000steps, trainer will call this DDIM sampler.
        :param fixed_noise: If set to True, then this Sampler will always use same starting noise for image generation.
        :param calculate_fid: Whether to calculate FID score for this sampler.
        :param num_fid_sample: # of generating samples for FID calculation.
        If calculate_fid==True and num_fid_sample==None, then it will automatically set # of generating image to the
        total number of image in original dataset.
        :param generate_image: Whether to save the generated image to folder.
        :param clip: [True, False, 'both'] 'both' will sample twice for clip==True and clip==False.
        Details in ddim_p_sample function.
        :param save: Whether to save the diffusion model based on the FID score calculated by this sampler.
        So calculate_fid must be set to True, if you want to set this parameter to be True.
        """
        super().__init__()
        self.ddim_steps = ddim_sampling_steps
        self.eta = eta
        self.sample_every = sample_every
        self.fixed_noise = fixed_noise
        self.calculate_fid = calculate_fid
        self.num_fid_sample = num_fid_sample
        self.generate_image = generate_image
        self.channel = self.ddpm_model.channel
        self.image_size = self.ddpm_model.image_size
        self.device = self.ddpm_model.device
        self.clip = clip
        self.save = save
        self.sampler_name = None
        self.save_path = None
        ddpm_steps = ddpm_diffusion_model.time_step
        assert self.ddim_steps <= ddpm_steps, 'DDIM sampling step must be smaller or equal to DDPM sampling step'
        assert clip in [True, False, 'both'], "clip must be one of [True, False, 'both']"
        if self.save:
            assert self.calculate_fid is True, 'To save model based on FID score, you must set [calculate_fid] to True'
        self.register_buffer('best_fid', torch.tensor([1e10], dtype=torch.float32))

        alpha_bar = ddpm_diffusion_model.alpha_bar
        # One thing you mush notice is that although sampling time is indexed as [1,...T] in paper,
        # since in computer program we index from [0,...T-1] rather than [1,...T],
        # value of tau ranges from [-1, ...T-1] where t=-1 indicate initial state (Data distribution)

        # [tau_1, tau_2, ... tau_S] sec 4.2
        self.register_buffer('tau', torch.linspace(-1, ddpm_steps - 1, steps=self.ddim_steps + 1, dtype=torch.long)[1:])

        alpha_tau_i = alpha_bar[self.tau]
        alpha_tau_i_min_1 = F.pad(alpha_bar[self.tau[:-1]], pad=(1, 0), value=1.)  # alpha_0 = 1

        # (16) in DDIM
        self.register_buffer('sigma', eta * (((1 - alpha_tau_i_min_1) / (1 - alpha_tau_i) *
                                              (1 - alpha_tau_i / alpha_tau_i_min_1)).sqrt()))
        # (12) in DDIM
        self.register_buffer('coeff', (1 - alpha_tau_i_min_1 - self.sigma ** 2).sqrt())
        self.register_buffer('sqrt_alpha_i_min_1', alpha_tau_i_min_1.sqrt())

        assert self.coeff[0] == 0.0 and self.sqrt_alpha_i_min_1[0] == 1.0, 'DDIM parameter error'

    @torch.inference_mode()
    def ddim_p_sample(self, model, xt, i, clip=True):
        """
        Sample x_{tau_(i-1)} from p(x_{tau_(i-1)} | x_{tau_i}), consult (56) in DDIM paper.
        Calculation is done using (12) in DDIM paper where t-1 has to be changed to tau_(i-1) and t has to be
        changed to tau_i in (12), for accelerated generation process where total # of de-noising step is S.

        :param model: Diffusion model
        :param xt: noisy image at time step tau_i
        :param i: i is the index of array tau which is an sub-sequence of [1, ..., T] of length S. See sec. 4.2
        :param clip: Like in GaussianDiffusion p_sample, we can clip(or clamp) the predicted x_0 to -1 ~ 1
        for better sampling result. If you see (12) in DDIM paper, sampling x_(t-1) depends on epsilon_theta which is
        U-net network predicted noise at time step t. If we want to clip the "predicted x0", we have to
        re-calculate the epsilon_theta to make "predicted x0" lie in -1 ~ 1. This is exactly what is going on
        if you set clip==True.
        :return: de-noised image at time step tau_(i-1)
        """
        t = self.tau[i]
        batched_time = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
        pred_noise = model.unet(xt, batched_time)  # corresponds to epsilon_{theta}
        x0 = model.sqrt_recip_alpha_bar[t] * xt - model.sqrt_recip_alpha_bar_min_1[t] * pred_noise
        if clip:
            x0.clamp_(-1., 1.)
            pred_noise = (model.sqrt_recip_alpha_bar[t] * xt - x0) / model.sqrt_recip_alpha_bar_min_1[t]

        # x0 corresponds to "predicted x0" and pred_noise corresponds to epsilon_theta(xt) in (12) DDIM
        # Thus self.coeff[i] * pred_noise corresponds to "direction pointing to xt" in (12)
        mean = self.sqrt_alpha_i_min_1[i] * x0 + self.coeff[i] * pred_noise
        noise = torch.randn_like(xt) if i > 0 else 0.
        # self.sigma[i] * noise corresponds to "random noise" in (12)
        x_t_minus_1 = mean + self.sigma[i] * noise
        return x_t_minus_1

    @torch.inference_mode()
    def sample(self, diffusion_model, batch_size, noise=None, return_all_timestep=False, clip=True):
        """

        :param diffusion_model: Diffusion model
        :param batch_size: # of image to generate.
        :param noise: If set to True, then this Sampler will always use same starting noise for image generation.
        :param return_all_timestep: Whether to return all images during de-noising process. So it will return the
        images from time step tau_S ~ time step tau_0
        :param clip: See ddim_p_sample function
        :return: Generated image of shape (b, 3, h, w) if return_all_timestep==False else (b, S, 3, h, w)
        """
        clip = clip if clip is not None else self.clip
        xT = torch.randn([batch_size, self.channel, self.image_size, self.image_size], device=self.device) \
            if noise is None else noise.to(self.device)
        denoised_intermediates = [xT]
        xt = xT
        for i in tqdm(reversed(range(0, self.ddim_steps)), desc='DDIM Sampling', total=self.ddim_steps, leave=False):
            x_t_minus_1 = self.ddim_p_sample(diffusion_model, xt, i, clip)
            denoised_intermediates.append(x_t_minus_1)
            xt = x_t_minus_1

        images = xt if not return_all_timestep else torch.stack(denoised_intermediates, dim=1)
        images = (images + 1.0) * 0.5  # scale to 0~1
        return images
