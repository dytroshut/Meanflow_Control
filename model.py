import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    def forward(self, t):
        t = t * 1000.0
        half_dim = self.nfreq
        freqs = torch.exp(-math.log(10000.0) * torch.arange(start=0, end=half_dim, 
                                                            dtype=torch.float32, device=t.device) / half_dim)
        args = t * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)

class FourierEmbedding(nn.Module):
    def __init__(self, in_dim: int, num_features: int, scale: float = 16.0):
        super().__init__()
        # CRITICAL FIX: Higher scale (16.0) is required for torus thickness/coverage
        B = torch.randn(in_dim, num_features) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResBlockAdaLN(nn.Module):
    def __init__(self, width, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 6 * width))
        self.conv1 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.conv2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        h = self.conv1(h); h = F.silu(h); h = self.dropout(h)
        x = x + gate_msa * h
        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        h = self.conv2(h); h = F.silu(h); h = self.dropout(h)
        x = x + gate_mlp * h
        return x

class ResMLP3D(nn.Module):
    def __init__(self, dim=3, width=512, depth=8, fourier_features=128, dropout=0.0):
        super().__init__()
        temb_dim = width * 4
        self.t_embedder = TimestepEmbedder(temb_dim)
        self.r_embedder = TimestepEmbedder(temb_dim)
        self.fourier = FourierEmbedding(dim, fourier_features)
        self.in_proj = nn.Linear(dim + fourier_features * 2, width)
        self.blocks = nn.ModuleList([ResBlockAdaLN(width, temb_dim, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(temb_dim, 2 * width))
        self.out_proj = nn.Linear(width, dim)
        
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, z, t, r):
        if t.ndim == 1: t = t[:, None]
        if r.ndim == 1: r = r[:, None]
        c = self.t_embedder(t) + self.r_embedder(r)
        x = self.in_proj(torch.cat([z, self.fourier(z)], dim=1))
        for block in self.blocks: x = block(x, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale) + shift
        return self.out_proj(x)