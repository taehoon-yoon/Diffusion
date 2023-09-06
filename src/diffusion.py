import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, time_step=1000, loss_type='l2'):
        super().__init__()
        self.model = model
        self.channel = self.model.channel
        self.device = self.model.device
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
        :param x0: (b, c, h, w)
        :param t: (b, )
        :param noise: (b, c, h, w)
        :return: x_t with shape=(b, c, h, w)
        """
        return self.sqrt_alpha_bar[t][:, None, None, None] * x0 + \
               self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * noise

    def forward(self, img):
        b, c, h, w = img.shape
        assert h == self.image_size and w == self.image_size, f'height and width of image must be {self.image_size}'
        t = torch.randint(0, self.time_step, (b,), device=img.device).long()  # (b, )
        noise = torch.randn_like(img)
        noised_image = self.q_sample(img, t, noise)
        predicted_noise = self.model(noised_image, t)

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

    # Reverse Process ##################################################################################################
    @torch.inference_mode()
    def p_sample(self, xt, t, clip=True):
        """
        Sample x_{t-1} from p_{theta}(x_{t-1} | x_t).
        There are two ways to sample x_{t-1}.

        https://github.com/hojonathanho/diffusion/issues/5
        :param xt: ( b, c, h, w)
        :param t: ( b, )
        :param clip: [True, False]
        :return:
        """

        pred_noise = self.model(xt, t)  # corresponds to epsilon_{theta}
        if clip:
            x0 = self.sqrt_recip_alpha_bar[t][:, None, None, None] * xt - \
                 self.sqrt_recip_alpha_bar_min_1[t][:, None, None, None] * pred_noise
            x0.clamp_(-1., 1.)
            mean = self.mean_tilde_x0_coeff[t][:, None, None, None] * x0 + \
                   self.mean_tilde_xt_coeff[t][:, None, None, None] * xt
        else:
            mean = self.sqrt_recip_alpha[t][:, None, None, None] * \
                   (xt - self.beta_over_sqrt_one_minus_alpha_bar[t][:, None, None, None]*pred_noise)
        variance = self.beta_tilde[t][:, None, None, None]
        noise = torch.randn_like(xt) if t[0] > 0 else 0.  # corresponds to z, consult 4: in Algorithm 2.
        x_t_minus_1 = mean + torch.sqrt(variance) * noise
        return x_t_minus_1

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timestep=False, clip=True):
        xT = torch.randn([batch_size, self.channel, self.image_size, self.image_size], device=self.device)
        denoised_intermediates = [xT]
        xt = xT
        for t in tqdm(reversed(range(0, self.time_step)), desc='Sampling', total=self.time_step, leave=False):
            batched_time = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x_t_minus_1 = self.p_sample(xt, batched_time, clip)
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
