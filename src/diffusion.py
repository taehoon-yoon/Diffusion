import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, time_step=1000, loss_type='l2'):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.time_step = time_step
        self.loss_type = loss_type

        beta = self.linear_beta_schedule()  # (t, )  t=time_step, in DDPM paper t=1000
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)  # (t, )
        alpha_bar_prev = F.pad(alpha_bar[:-1], pad=(1, 0), value=1.)  # (t, )

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)

        # calculation for q(x_t | x_0) consult (4) in DDPM paper.
        self.register_buffer('sqrt_alpha_ber', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - alpha_bar))

        # calculation for q(x_{t-1} | x_t, x_0) consult (7) in DDPM paper.
        self.register_buffer('beta_tilde', beta * ((1. - alpha_bar_prev) / (1. - alpha_bar)))
        self.register_buffer('mean_tilde_x0_coeff', beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar))
        self.register_buffer('mean_tilde_xt_coeff', torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

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

    def linear_beta_schedule(self):
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / self.time_step
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.time_step, dtype=torch.float32)
