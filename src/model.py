import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
        :param time_emb_dim: d_model for Positional Encoding
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
        :param time_emb: (B, d_model)
        :return: (B, dim_out, H, W)
        """
        scale_shift = None
        if time_emb is not None:
            scale_shift = self.mlp(scale_shift)[..., None, None]  # (B, dim_out*2, 1, 1)
            scale_shift = scale_shift.chunk(2, dim=1)  # len 2 with each element shape (B, dim_out, 1, 1)
        hidden = self.block1(x, scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.residual_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, head=4, dim_head=32, dropout=None):
        super().__init__()
        self.head = head
        hidden_dim = head * dim_head

        self.scale = dim_head ** (-0.5)  # 1 / sqrt(d_k)
        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
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
        attention_score = self.dropout(attention_score) if self.dropout is not None else attention_score
        attention = torch.einsum('b h n m, b h m f -> b h n f', attention_score * self.scale, v)
        # attention(Q, K, V) = softmax(Q(K^T) / sqrt(d_k))V / Scaled Dot-Product Attention

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
        out = rearrange(linear_attention, 'b h e (i j) -> b (h e) i j', i=i, j=j, e=self.head)
        return self.to_out(out)
