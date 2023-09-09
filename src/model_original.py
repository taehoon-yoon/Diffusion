import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from functools import partial
from .utils import PositionalEncoding


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=(3, 3), padding=1)
        self.block1 = nn.Sequential(self.norm1, self.activation1, self.conv1)

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if time_emb_dim is not None else None

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        self.activation2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3), padding=1)
        self.block2 = nn.Sequential(self.norm2, self.activation2, self.dropout, self.conv2)

        self.residual_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, 1)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        hidden = self.block1(x)
        if time_emb is not None:
            # add in timestep embedding
            hidden = hidden + self.mlp(time_emb)[..., None, None]  # (B, dim_out, 1, 1)
        hidden = self.block2(hidden)
        return hidden + self.residual_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, groups=32):
        super().__init__()
        self.scale = dim ** (-0.5)  # 1 / sqrt(d_k)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.to_qkv = nn.Conv2d(dim, dim*3, kernel_size=(1, 1))
        self.to_out = nn.Conv2d(dim, dim, kernel_size=(1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        # You can think (h*w) as sequence length where c is d_k in <Attention is all you need>
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), qkv)

        """
        q, k, v shape: (batch, seq_length, d_k)  seq_length = height*width, d_k == c == dim
        similarity shape: (batch, seq_length, seq_length)
        attention_score shape: (batch, seq_length, seq_length)
        attention shape: (batch, seq_length, d_k)
        out shape: (batch, d_k, height, width)  d_k == c == dim
        return shape: (batch, dim, height, width)
        """

        similarity = torch.einsum('b i c, b j c -> b i j', q, k)  # Q(K^T)
        attention_score = torch.softmax(similarity*self.scale, dim=-1)  # softmax(Q(K^T) / sqrt(d_k))
        attention = torch.einsum('b i j, b j c -> b i c', attention_score, v)
        # attention(Q, K, V) = [softmax(Q(K^T) / sqrt(d_k))]V -> Scaled Dot-Product Attention
        out = rearrange(attention, 'b i c -> b c h w', h=h, w=w)
        return self.to_out(out) + x


def downSample(dim_in):
    return nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2))


def upSample(dim_in):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), padding=1))


class Unet(nn.Module):
    def __init__(self, dim, image_size, dim_multiply=(1, 2, 4, 8), channel=3, num_res_blocks=2,
                 attn_resolutions=(16, ), dropout=0, device='cuda', groups=32):
        super().__init__()

        # Attributes
        self.dim = dim
        self.channel = channel
        self.time_emb_dim = 4 * self.dim
        self.num_resolutions = len(dim_multiply)
        self.device = device
        self.resolution = [int(image_size/(2**i)) for i in range(self.num_resolutions)]
        self.hidden_dims = [self.dim, *map(lambda x: x * self.dim, dim_multiply)]

        # Time embedding
        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, nn.Linear(self.dim, self.time_emb_dim),
            nn.SiLU(), nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # Layer definition
        self.down_path = nn.ModuleList([])
        self.middle_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])
        self.concat_dim = list()
        self.input_time_emb_down = list()
        self.input_time_emb_middle = list()
        self.input_time_emb_up = list()

        # Downward Path layer definition
        self.input_time_emb_down.append(False)
        self.down_path.append(nn.Conv2d(channel, self.dim, kernel_size=(3, 3), padding=1))
        self.concat_dim.append(self.dim)

        for level in range(self.num_resolutions):
            d_in, d_out = self.hidden_dims[level], self.hidden_dims[level+1]
            for block in range(num_res_blocks):
                self.input_time_emb_down.append(True)
                self.down_path.append(ResnetBlock(d_in, d_out, time_emb_dim=self.time_emb_dim, dropout=dropout))
                if self.resolution[level] in attn_resolutions:
                    self.concat_dim.append(None)
                    self.input_time_emb_down.append(False)
                    self.down_path.append(Attention(d_out))
                self.concat_dim.append(d_out)
            if level != self.num_resolutions-1:
                self.input_time_emb_down.append(False)
                self.down_path.append(downSample(d_out))
                self.concat_dim.append(d_out)

        # Middle layer definition
        mid_dim = self.hidden_dims[-1]
        self.input_time_emb_middle.append(True)
        self.middle_path.append(ResnetBlock(mid_dim, mid_dim, self.time_emb_dim, dropout))
        self.input_time_emb_middle.append(False)
        self.middle_path.append(Attention(mid_dim))
        self.input_time_emb_middle.append(True)
        self.middle_path.append(ResnetBlock(mid_dim, mid_dim, self.time_emb_dim, dropout))

        # Upward Path layer definition
        concat_d = self.concat_dim.copy()
        for level in reversed(range(self.num_resolutions)):
            d_out = self.hidden_dims[level + 1]
            for block in range(num_res_blocks + 1):
                self.input_time_emb_up.append(True)
                self.up_path.append(ResnetBlock(concat_d.pop()+d_out, d_out, self.time_emb_dim, dropout))
                if self.resolution[level] in attn_resolutions:
                    self.input_time_emb_up.append(False)
                    self.up_path.append(Attention(d_out))
            if level != 0:
                self.input_time_emb_up.append(False)
                self.up_path.append(upSample(d_out))

        assert not concat_d, 'Error in concatenation between downward path and upward path.'

        # Output layer
        final_ch = self.hidden_dims[1]
        self.final_norm = nn.GroupNorm(groups, final_ch)
        self.final_activation = nn.SiLU()
        self.final_conv = nn.Conv2d(final_ch, channel, kernel_size=(3, 3), padding=1)

    def forward(self, x, time):
        t = self.time_mlp(time)

        # Downward
        concat = list()
        for concat_dim, time_emb_bool, layer in zip(self.concat_dim, self.input_time_emb_down, self.down_path):
            x = layer(x, t) if time_emb_bool else layer(x)
            if concat_dim is not None:
                concat.append(x)

        # Middle
        for time_emb_bool, layer in zip(self.input_time_emb_middle, self.middle_path):
            x = layer(x, t) if time_emb_bool else layer(x)

        # Upward
        for concat_dim, time_emb_bool, layer in zip(reversed(self.concat_dim), self.input_time_emb_up, self.up_path):
            if concat_dim is not None:
                x = torch.cat((x, concat.pop()), dim=1)
            x = layer(x, t) if time_emb_bool else layer(x)

        # Final
        x = self.final_activation(self.final_norm(x))
        return self.final_conv(x)

