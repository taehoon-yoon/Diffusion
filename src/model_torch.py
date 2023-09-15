import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from functools import partial
from .utils import PositionalEncoding


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        """
        Input shape=(B, dim, H, W)
        Output shape=(B, dim_out, H, W)

        :param dim: input channel
        :param dim_out: output channel
        :param groups: number of groups for Group normalization.
        """
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=(3, 3), padding=1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        self.activation = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        return self.activation(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, group=8):
        """
        In abstract, it is composed of two Convolutional layer with residual connection,
        with information of time encoding is passed to first Convolutional layer.

        Input shape=(B, dim, H, W)
        Output shape=(B, dim_out, H, W)

        :param dim: input channel
        :param dim_out: output channel
        :param time_emb_dim: Embedding dimension for time.
        :param group: number of groups for Group normalization.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None
        self.block1 = Block(dim, dim_out, group)
        self.block2 = Block(dim_out, dim_out, group)
        self.residual_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, 1)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """

        :param x: (B, dim, H, W)
        :param time_emb: (B, time_emb_dim)
        :return: (B, dim_out, H, W)
        """
        scale_shift = None
        if time_emb is not None:
            scale_shift = self.mlp(time_emb)[..., None, None]  # (B, dim_out*2, 1, 1)
            scale_shift = scale_shift.chunk(2, dim=1)  # len 2 with each element shape (B, dim_out, 1, 1)
        hidden = self.block1(x, scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.residual_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, head=4, dim_head=32):
        super().__init__()
        self.head = head
        hidden_dim = head * dim_head

        self.scale = dim_head ** (-0.5)  # 1 / sqrt(d_k)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=(1, 1), bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1))

    def forward(self, x):
        b, c, i, j = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        # h=self.head, f=dim_head, i=height, j=width.
        # You can think (i*j) as sequence length where f is d_k in <Attention is all you need>
        q, k, v = map(lambda t: rearrange(t, 'b (h f) i j -> b h (i j) f', h=self.head), qkv)

        """
        q, k, v shape: (batch, # of head, seq_length, d_k)  seq_length = height * width
        similarity shape: (batch, # of head, seq_length, seq_length)
        attention_score shape: (batch, # of head, seq_length, seq_length)
        attention shape: (batch, # of head, seq_length, d_k)
        out shape: (batch, hidden_dim, height, width)
        return shape: (batch, dim, height, width)
        """
        # n, m is likewise sequence length.
        similarity = torch.einsum('b h n f, b h m f -> b h n m', q, k)  # Q(K^T)
        attention_score = torch.softmax(similarity * self.scale, dim=-1)  # softmax(Q(K^T) / sqrt(d_k))
        attention = torch.einsum('b h n m, b h m f -> b h n f', attention_score, v)
        # attention(Q, K, V) = [softmax(Q(K^T) / sqrt(d_k))]V -> Scaled Dot-Product Attention

        out = rearrange(attention, 'b h (i j) f -> b (h f) i j', i=i, j=j)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, head=4, dim_head=32):
        super().__init__()
        self.head = head
        hidden_dim = head * dim_head

        self.scale = dim_head ** (-0.5)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=(1, 1), bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1)), RMSNorm(dim))

    def forward(self, x):
        b, c, i, j = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        # h=self.head, f=dim_head, i=height, j=width.
        # You can think (i*j) as sequence length where f is d_k in <Attention is all you need>
        q, k, v = map(lambda t: rearrange(t, 'b (h f) i j -> b h f (i j)', h=self.head), qkv)

        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum('b h f m, b h e m -> b h f e', k, v)
        linear_attention = torch.einsum('b h f e, b h f n -> b h e n', context, q)
        out = rearrange(linear_attention, 'b h e (i j) -> b (h e) i j', i=i, j=j, h=self.head)
        return self.to_out(out)


def downSample(dim_in, dim_out):
    return nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
                         nn.Conv2d(dim_in * 4, dim_out, kernel_size=(1, 1)))


def upSample(dim_in, dim_out):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), padding=1))


class Unet(nn.Module):
    def __init__(self, dim, image_size, dim_multiply=(1, 2, 4, 8), channel=3, attn_heads=4, attn_head_dim=32,
                 full_attn=(False, False, False, True), resnet_group_norm=8, device='cuda'):
        """
        U-net for noise prediction. Code is based on denoising-diffusion-pytorch
        https://github.com/lucidrains/denoising-diffusion-pytorch
        :param dim: See below
        :param dim_multiply: len(dim_multiply) will be the depth of U-net model with at each level i, the dimension
        of channel will be dim * dim_multiply[i]. If the input image shape is [H, W, 3] then at the lowest level,
        feature map shape will be [H/(2^(len(dim_multiply)-1), W/(2^(len(dim_multiply)-1), dim*dim_multiply[-1]]
        if not considering U-net down-up path connection.
        :param channel: 3
        :param attn_heads: It uses multi-head-self-Attention. attn_head is the # of head. It corresponds to h in
        "Attention is all you need" paper. See section 3.2.2
        :param attn_head_dim: It is the dimension of each head. It corresponds to d_k in Attention paper.
        :param full_attn: In pytorch implementation they used Linear Attention where full Attention(multi head self
        attention) is not applied. This param indicates at each level, whether to use full attention
        or use linear attention. So the len(full_attn) must equal to len(dim_multiply). For example if
        full_attn=(F, F, F, T) then at level 0, 1, 2 it will use Linear Attention and at level 3 it will use
        multi-head self attention(i.e. full attention)
        :param resnet_group_norm: number of groups for Group normalization.
        :param device: either 'cuda' or 'cpu'
        """
        super().__init__()
        assert len(dim_multiply) == len(full_attn), 'Length of dim_multiply and Length of full_attn must be same'

        # Attributes
        self.dim = dim
        self.channel = channel
        self.hidden_dims = [self.dim, *map(lambda x: x * self.dim, dim_multiply)]
        self.dim_in_out = list(zip(self.hidden_dims[:-1], self.hidden_dims[1:]))
        self.time_emb_dim = 4 * self.dim
        self.full_attn = full_attn
        self.depth = len(dim_multiply)
        self.device = device

        # Time embedding
        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, nn.Linear(self.dim, self.time_emb_dim),
            nn.GELU(), nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # Layer definition
        resnet_block = partial(ResnetBlock, time_emb_dim=self.time_emb_dim, group=resnet_group_norm)
        self.init_conv = nn.Conv2d(self.channel, self.dim, kernel_size=(7, 7), padding=3)
        self.down_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])

        # Downward Path layer definition
        for idx, ((dim_in, dim_out), full_attn_flag) in enumerate(zip(self.dim_in_out, self.full_attn)):
            isLast = idx == (self.depth - 1)
            attention = LinearAttention if not full_attn_flag else Attention
            self.down_path.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attention(dim_in, head=attn_heads, dim_head=attn_head_dim),
                downSample(dim_in, dim_out) if not isLast else nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), padding=1)
            ]))

        # Middle layer definition
        mid_dim = self.hidden_dims[-1]
        self.mid_resnet_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attention = Attention(mid_dim, head=attn_heads, dim_head=attn_head_dim)
        self.mid_resnet_block2 = resnet_block(mid_dim, mid_dim)

        # Upward Path layer definition
        for idx, ((dim_in, dim_out), full_attn_flag) in enumerate(
                zip(reversed(self.dim_in_out), reversed(self.full_attn))):
            isLast = idx == (self.depth - 1)
            attention = LinearAttention if not full_attn_flag else Attention
            self.up_path.append(nn.ModuleList([
                resnet_block(dim_in + dim_out, dim_out),
                resnet_block(dim_in + dim_out, dim_out),
                attention(dim_out, head=attn_heads, dim_head=attn_head_dim),
                upSample(dim_out, dim_in) if not isLast else nn.Conv2d(dim_out, dim_in, kernel_size=(3, 3), padding=1)
            ]))

        self.final_resnet_block = resnet_block(2 * self.dim, self.dim)
        self.final_conv = nn.Conv2d(self.dim, self.channel, kernel_size=(1, 1))

    def forward(self, x, time):
        """
        return predicted noise given x_t and t
        """
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        concat = list()

        for block1, block2, attn, downsample in self.down_path:
            x = block1(x, t)
            concat.append(x)

            x = block2(x, t)
            x = attn(x) + x
            concat.append(x)

            x = downsample(x)

        x = self.mid_resnet_block1(x, t)
        x = self.mid_attention(x) + x
        x = self.mid_resnet_block2(x, t)

        for block1, block2, attn, upsample in self.up_path:
            x = torch.cat((x, concat.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, concat.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_resnet_block(x, t)
        return self.final_conv(x)
