"""Flamingo Paper Deep Dive"""

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


# ============================================
# PART 1: Perceiver Resampler Implementation
# ============================================


class PerceiverAttention(nn.Module):
    """
    Cross-attention layer for Perceiver Resampler

    KEY INNOVATION: concatenates latent (K,V) with input (K,V) which allows latents to attend to both:
    1. The visual features (cross-attention)
    2. Each other (self-attention component)
    """

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim_head**0.5
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        # Linear Projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents (torch.Tensor): [batch, num_latents, dim] - queries
            x (torch.Tensor): [batch, num_media_tokens, dim] - keys/values

        Returns:
            [batch, num_latents, dim]
        """
        # b, n = latents.shape[0], latents.shape[1]
        h = self.heads

        # layer norm
        latents = self.norm_latents(latents)
        x = self.norm_media(x)

        # get q
        q = self.to_q(latents)

        # get k,v
        latents_kv = self.to_kv(latents).chunk(2, dim=-1)
        media_kv = self.to_kv(x).chunk(2, dim=-1)

        # concat k,v
        k = torch.cat([latents_kv[0], media_kv[0]], dim=1)
        v = torch.cat([latents_kv[1], media_kv[1]], dim=1)

        # reshape for multi-head
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        # attention
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class FeedForward(nn.Module):
    """FeedForward Network"""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler

    KEY INNOVATION: uses learned latent queries to compress variable-length visual features into a fixed number of tokens

    Args:
        dim: Hidden dimensions
        depth: number of transformer layers
        num_latents: number of latent learned queries (64 as per paper)
        heads: number of attention heads
        dim_head = Dimension per head
        ff_mult = feed forward multiplier

    """

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 6,
        num_latents: int = 64,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_latents = num_latents

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # transformer layers
        for _ in range(depth):
            self.attn_layers = nn.ModuleList(
                [PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads) for _ in range(depth)]
            )
            self.ff_layers = nn.ModuleList([FeedForward(dim=dim, mult=ff_mult) for _ in range(depth)])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): Visual features [batch, num_tokens, dim]

        Returns:
            torch.Tensor: Fixed size visual tokens [batch, num_latents, dim]
        """
        batch_size = x.shape[0]

        latents = repeat(self.latents, "n d -> b n d", b=batch_size)

        for attn, ff in zip(self.attn_layers, self.ff_layers):
            # Cross-attention: latents query visual features
            latents = attn(latents, x) + latents

            # Feed-forward
            latents = ff(latents) + latents

        return self.norm(latents)


class SelfAttentionLayer(nn.Module):
    """Standard self-attention layer"""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        # Linear projections
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_tokens, dim]
        Returns:
            [batch, num_tokens, dim]
        """
        h = self.heads

        # Layer norm
        x = self.norm(x)

        # Get q, k, v
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        # Attention
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class DirectSelfAttention(nn.Module):
    """Baseline: Direct self-attention on all visual tokens (no compression)"""

    def __init__(self, dim: int = 1024, depth: int = 6, heads: int = 8, dim_head: int = 64, ff_mult: int = 4) -> None:
        super().__init__()
        self.depth = depth

        # Separate attention and feedforward layers (matching PerceiverResampler structure)
        self.attn_layers = nn.ModuleList(
            [SelfAttentionLayer(dim=dim, dim_head=dim_head, heads=heads) for _ in range(depth)]
        )
        self.ff_layers = nn.ModuleList([FeedForward(dim=dim, mult=ff_mult) for _ in range(depth)])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_tokens, dim]
        Returns:
            [batch, num_tokens, dim] - all tokens
        """
        for attn, ff in zip(self.attn_layers, self.ff_layers):
            # Self-attention
            x = attn(x) + x

            # Feed-forward
            x = ff(x) + x

        return self.norm(x)


def demo_perceiver_resampler():
    """Demo Perceiver Resampler"""
    print("\n" + "=" * 60)
    print("PART 1: Perceiver Resampler Analysis")
    print("=" * 60)

    # create perceiver
    num_latents = 64
    perceiver = PerceiverResampler(dim=1024, depth=6, num_latents=num_latents, heads=8, dim_head=64).to(device)

    batch_size = 4
    num_spatial_tokens = 24 * 24  # 24x24 grid
    dim = 1024

    visual_features = torch.randn(batch_size, num_spatial_tokens, dim).to(device)

    print(f"\nInput Shape: {visual_features.shape}")
    print(f"input tokens per image: {num_spatial_tokens}")
    print(f"complexity: O({num_spatial_tokens**2:,})")

    print("\n" + "-" * 60)
    print("Perceiver Resampler")
    print("-" * 60)
    with torch.no_grad():
        visual_tokens = perceiver(visual_features)

    print(f"\nOutput shape: {visual_tokens.shape}")
    print(f"Output tokens per image: {visual_tokens.shape[1]}")
    # param count
    params = sum(p.numel() for p in perceiver.parameters())
    print(f"Perceiver Parameters: {params:,}")

    print(f"Compression ratio: {num_spatial_tokens / visual_tokens.shape[1]:.1f}x")
    print(f"New complexity: O({visual_tokens.shape[1]**2:,})")

    # Direct Self-attention (no compression)
    print("\n" + "-" * 60)
    print("Regular Self-Attention (No Perceiver Sampler)")
    print("-" * 60)

    baseline = DirectSelfAttention(dim=dim, depth=6, heads=8, dim_head=64).to(device)

    with torch.no_grad():
        baseline_output = baseline(visual_features)

    print(f"\nOutput shape: {baseline_output.shape}")
    print(f"Output tokens per image: {baseline_output.shape[1]}")

    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Self-Attention parameters: {baseline_params:,}")
    print(f"Compression ratio: {num_spatial_tokens / baseline_output.shape[1]:.1f}x")
    print(f"New complexity: O({baseline_output.shape[1]**2:,})")

    print("\n")
    num_runs = 100
    print(f"Running benchmark for {num_runs} runs:")

    # Benchmark Perceiver Resampler
    start_time = time.time()
    for i in range(num_runs):
        with torch.no_grad():
            visual_tokens = perceiver(visual_features)
    duration = time.time() - start_time

    perceiver_time = duration / num_runs
    perceiver_imgs_per_sec = batch_size / perceiver_time

    # Benchmark Direct Self-attention
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            baseline_output = baseline(visual_features)
    baseline_duration = time.time() - start_time
    baseline_time = baseline_duration / num_runs
    baseline_imgs_per_sec = batch_size / baseline_time

    compression_ratio = num_spatial_tokens / num_latents

    print("=" * 110)
    print(
        f"{'Method':<30} {'Time (ms)':<15} {'Images/sec':<15} {'Tokens/img':<15} {'Token Compression':<22} {'Speedup':<10}"
    )
    print("-" * 110)
    print(
        f"{'Perceiver Resampler':<30} {perceiver_time*1000:<15.2f} {perceiver_imgs_per_sec:<15.2f} {num_latents:<15} {f'{compression_ratio:.1f}x':<22} {f'{1.0:.2f}x':<10}"
    )
    print(
        f"{'Direct Self-Attention':<30} {baseline_time*1000:<15.2f} {baseline_imgs_per_sec:<15.2f} {num_spatial_tokens:<15} {'1.0x':<22} {f'{baseline_time/perceiver_time:.2f}x':<10}"
    )
    print("=" * 110)
    print(f"\nPerceiver is {baseline_time/perceiver_time:.2f}x FASTER with {compression_ratio:.1f}x token compression!")

    return perceiver, visual_tokens


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("FLAMINGO EXPLORATION")
    print("=" * 50)

    perceiver, visual_tokens = demo_perceiver_resampler()
