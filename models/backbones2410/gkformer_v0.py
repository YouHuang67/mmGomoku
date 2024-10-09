from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from engine.utils import rearrange, repeat, reduce
from projects.cppboard import Board


class RotaryEmbedding(nn.Module):

    def __init__(self,
                 dim,
                 size,
                 ratio=0.5,
                 theta=10000):
        super(RotaryEmbedding, self).__init__()
        freqs = theta ** (-torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        freqs = repeat(freqs, 'f -> (f r)', r=2) * ratio
        ys, xs = torch.meshgrid(
            torch.arange(size), torch.arange(size), indexing='ij')
        coords = torch.stack((ys, xs), dim=-1).float()
        coords = rearrange(coords, 'h w c -> (h w) c')
        freqs = torch.einsum('..., f -> ... f', coords, freqs)
        freqs = rearrange(freqs, '... d f -> ... (d f)', d=2)
        self.register_buffer('freqs', freqs)
        self.cls_freq = nn.Parameter(nn.Embedding(1, 2 * dim).weight)

    def forward(self, x):
        """
        :param x: (..., D)
        :return:  (..., D)
        """
        freqs = torch.cat([self.freqs, self.cls_freq], dim=0)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        rt_x = x.new_empty(x.shape)
        rt_x[..., 0::2] = x[..., 1::2]
        rt_x[..., 1::2] = -x[..., 0::2]
        return x * freqs_cos + rt_x * freqs_sin


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(WindowAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        hidden_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, rope):
        """
        params x: (B, L, C)
        return:   (B, L, C)
        """
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, 'b l (n3 n d) -> n3 b n l d', n3=3, n=self.num_heads
        ).unbind(dim=0)

        q = rope(q).contiguous()
        k = rope(k).contiguous()
        v = v.contiguous()

        attn = ((self.scale * q) @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape_as(x)

        x = self.proj(x)
        return x


class Block(nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_ratio=1.0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_dim=embed_dim, hidden_dim=mlp_hidden_dim)

    def forward(self, x, **kwargs):
        """
        params x: (B, L, C)
        return:   (B, L, C)
        """
        x = x + self.attn(self.norm1(x), **kwargs)
        x = x + self.mlp(self.norm2(x))
        return x


@MODELS.register_module()
class GKFormerV0(nn.Module):

    def __init__(self,

                 depth,
                 embed_dim,
                 num_heads,
                 mlp_ratio,

                 rope_ratio=0.5,
                 rope_theta=10000,

                 in_dim=3,
                 size=Board.BOARD_SIZE):

        super(GKFormerV0, self).__init__()
        self.in_dim = in_dim
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.size = size

        ys = torch.linspace(0, 1, 2 * size + 1)[1::2]
        xs = torch.linspace(0, 1, 2 * size + 1)[1::2]
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([ys.flatten(), xs.flatten()], dim=-1)
        self.register_buffer('coords', rearrange(coords, 'l c -> () l c'))
        self.in_proj = nn.Linear(in_dim + 2, embed_dim)

        self.cls_token = nn.Embedding(1, embed_dim)

        self.rope = RotaryEmbedding(
            dim=embed_dim // num_heads // 2,
            size=size,
            ratio=rope_ratio,
            theta=rope_theta
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.action_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        """
        :param x: (B, 3, H, W)
        :return: (B, H, W), (B, )
        """
        B, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = torch.cat([x, self.coords.repeat(B, 1, 1)], dim=-1)
        x = self.in_proj(x)
        cls_token = repeat(self.cls_token.weight, '() d -> b () d', b=B)
        x = torch.cat([x, cls_token], dim=1)

        for block in self.blocks:
            x = block(x, rope=self.rope)
        x, cls_token = x.split([H * W, 1], dim=1)
        action_probs = self.action_mlp(x)
        action_probs = rearrange(action_probs, 'b (h w) () -> b h w', h=H, w=W)
        values = self.value_mlp(cls_token)
        values = values.flatten()
        return dict(action_probs=action_probs, values=values)


if __name__ == '__main__':
    model = GKFormerV0(16, 64, 4, 1.0)
    out = model(torch.randint(2, size=(2, 3, 15, 15)).float())
    print(out['action_probs'].shape, out['values'].shape)
