import torch
import torch.nn as nn
from einops import rearrange
from .utils import PositionalEncoding


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32):
        super().__init__()

        self.dim, self.dim_out = dim, dim_out

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

        self.dim, self.dim_out = dim, dim

        self.scale = dim ** (-0.5)  # 1 / sqrt(d_k)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1))
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
        attention_score = torch.softmax(similarity * self.scale, dim=-1)  # softmax(Q(K^T) / sqrt(d_k))
        attention = torch.einsum('b i j, b j c -> b i c', attention_score, v)
        # attention(Q, K, V) = [softmax(Q(K^T) / sqrt(d_k))]V -> Scaled Dot-Product Attention
        out = rearrange(attention, 'b (h w) c -> b c h w', h=h, w=w)
        return self.to_out(out) + x


class ResnetAttentionBlock(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32):
        super().__init__()

        self.dim, self.dim_out = dim, dim_out

        self.resnet = ResnetBlock(dim, dim_out, time_emb_dim, dropout, groups)
        self.attention = Attention(dim_out, groups)

    def forward(self, x, time_emb=None):
        x = self.resnet(x, time_emb)
        return self.attention(x)


class downSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.dim, self.dim_out = dim_in, dim_in

        self.downsameple = nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        return self.downsameple(x)


class upSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.dim, self.dim_out = dim_in, dim_in

        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), padding=1))

    def forward(self, x):
        return self.upsample(x)


class Unet(nn.Module):
    def __init__(self, dim, image_size, dim_multiply=(1, 2, 4, 8), channel=3, num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0, device='cuda', groups=32):
        """
        U-net for noise prediction. Code is based on Denoising Diffusion Probabilistic Models
        https://github.com/hojonathanho/diffusion
        :param dim: See below
        :param dim_multiply: len(dim_multiply) will be the depth of U-net model with at each level i, the dimension
        of channel will be dim * dim_multiply[i]. If the input image shape is [H, W, 3] then at the lowest level,
        feature map shape will be [H/(2^(len(dim_multiply)-1), W/(2^(len(dim_multiply)-1), dim*dim_multiply[-1]]
        if not considering U-net down-up path connection.
        :param image_size: input image size
        :param channel: 3
        :param num_res_blocks: # of ResnetBlock at each level. In downward path, at each level, there will be
        num_res_blocks amount of ResnetBlock module and in upward path, at each level, there will be
        (num_res_blocks+1) amount of ResnetBlock module
        :param attn_resolutions: The feature map resolution where we will apply Attention. In DDPM paper, author
        used Attention module when resolution of feature map is 16.
        :param dropout: dropout. If set to 0 then no dropout.
        :param device: either 'cuda' or 'cpu'
        :param groups: number of groups for Group normalization.
        """
        super().__init__()
        assert dim % groups == 0, 'parameter [groups] must be divisible by parameter [dim]'

        # Attributes
        self.dim = dim
        self.channel = channel
        self.time_emb_dim = 4 * self.dim
        self.num_resolutions = len(dim_multiply)
        self.device = device
        self.resolution = [int(image_size / (2 ** i)) for i in range(self.num_resolutions)]
        self.hidden_dims = [self.dim, *map(lambda x: x * self.dim, dim_multiply)]
        self.num_res_blocks = num_res_blocks

        # Time embedding
        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, nn.Linear(self.dim, self.time_emb_dim),
            nn.SiLU(), nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # Layer definition
        self.down_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])
        concat_dim = list()

        # Downward Path layer definition
        self.init_conv = nn.Conv2d(channel, self.dim, kernel_size=(3, 3), padding=1)
        concat_dim.append(self.dim)

        for level in range(self.num_resolutions):
            d_in, d_out = self.hidden_dims[level], self.hidden_dims[level + 1]
            for block in range(num_res_blocks):
                d_in_ = d_in if block == 0 else d_out
                if self.resolution[level] in attn_resolutions:
                    self.down_path.append(ResnetAttentionBlock(d_in_, d_out, self.time_emb_dim, dropout, groups))
                else:
                    self.down_path.append(ResnetBlock(d_in_, d_out, self.time_emb_dim, dropout, groups))
                concat_dim.append(d_out)
            if level != self.num_resolutions - 1:
                self.down_path.append(downSample(d_out))
                concat_dim.append(d_out)

        # Middle layer definition
        mid_dim = self.hidden_dims[-1]
        self.middle_resnet_attention = ResnetAttentionBlock(mid_dim, mid_dim, self.time_emb_dim, dropout, groups)
        self.middle_resnet = ResnetBlock(mid_dim, mid_dim, self.time_emb_dim, dropout, groups)

        # Upward Path layer definition
        for level in reversed(range(self.num_resolutions)):
            d_out = self.hidden_dims[level + 1]
            for block in range(num_res_blocks + 1):
                d_in = self.hidden_dims[level + 2] if block == 0 and level != self.num_resolutions - 1 else d_out
                d_in = d_in + concat_dim.pop()
                if self.resolution[level] in attn_resolutions:
                    self.up_path.append(ResnetAttentionBlock(d_in, d_out, self.time_emb_dim, dropout, groups))
                else:
                    self.up_path.append(ResnetBlock(d_in, d_out, self.time_emb_dim, dropout, groups))
            if level != 0:
                self.up_path.append(upSample(d_out))

        assert not concat_dim, 'Error in concatenation between downward path and upward path.'

        # Output layer
        final_ch = self.hidden_dims[1]
        self.final_norm = nn.GroupNorm(groups, final_ch)
        self.final_activation = nn.SiLU()
        self.final_conv = nn.Conv2d(final_ch, channel, kernel_size=(3, 3), padding=1)

    def forward(self, x, time):
        """
        return predicted noise given x_t and t
        """
        t = self.time_mlp(time)
        # Downward
        concat = list()
        x = self.init_conv(x)
        concat.append(x)
        for layer in self.down_path:
            x = layer(x, t) if not isinstance(layer, (upSample, downSample)) else layer(x)
            concat.append(x)

        # Middle
        x = self.middle_resnet_attention(x, t)
        x = self.middle_resnet(x, t)

        # Upward
        for layer in self.up_path:
            if not isinstance(layer, upSample):
                x = torch.cat((x, concat.pop()), dim=1)
            x = layer(x, t) if not isinstance(layer, (upSample, downSample)) else layer(x)

        # Final
        x = self.final_activation(self.final_norm(x))
        return self.final_conv(x)

    def print_model_structure(self):
        for i in self.down_path:
            if i.__class__.__name__ == 'downSample':
                print('-' * 20)
            if i.__class__.__name__ == "Conv2d":

                print(i.__class__.__name__)
            else:
                print(i.__class__.__name__, i.dim, i.dim_out)
        print('\n')
        print('=' * 20)
        print('\n')
        for i in self.up_path:
            if i.__class__.__name__ == 'upSample':
                print('-' * 20)
            if i.__class__.__name__ == "Conv2d":
                print(i.__class__.__name__)
            else:
                print(i.__class__.__name__, i.dim, i.dim_out)
