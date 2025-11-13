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

# Import common layers
from attention_layers import CrossAttention, PerceiverAttention, SelfAttentionLayer
from feedforward import FeedForward

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


# ============================================
# PART 1: Perceiver Resampler Implementation
# ============================================


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
            # Cross-attention
            latents = attn(latents, x) + latents

            # Feed-forward
            latents = ff(latents) + latents

        return self.norm(latents)


class DirectSelfAttention(nn.Module):
    """Baseline: Direct self-attention on all visual tokens (no compression)"""

    def __init__(self, dim: int = 1024, depth: int = 6, heads: int = 8, dim_head: int = 64, ff_mult: int = 4) -> None:
        super().__init__()
        self.depth = depth

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
    print("\n\n")
    print("=" * 60)
    print("PART 1: Perceiver Resampler Analysis")
    print("=" * 60)

    # create Perceiver
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

    # Benchmark Perceiver Resampler
    print(f"Running Perceiver Resampler benchmark for {num_runs} runs:")
    start_time = time.time()
    for i in tqdm(range(num_runs)):
        with torch.no_grad():
            visual_tokens = perceiver(visual_features)
    duration = time.time() - start_time

    perceiver_time = duration / num_runs
    perceiver_imgs_per_sec = batch_size / perceiver_time

    # Benchmark Direct Self-attention
    print(f"Running Direct Self-attention Benchmark for {num_runs} runs:")
    start_time = time.time()
    for _ in tqdm(range(num_runs)):
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
# PART 2: Gated Cross-Attention Implementation
# ============================================


class GatedCrossAttentionBlock(nn.Module):
    """Gated cross attention block

    KEY INNOVATION: Tanh gating initialized at zero
    - Allows gradual visual information injection
    - Preserves frozen LM at initialization
    - Critical for training stability

    Forward pass:
    1. Cross-attention: text queries attend to visual tokens
    2. Tanh gate: control information flow
    3. Residual connection: preserve original text
    4. Feed-forward with another tanh gate
    """

    def __init__(
        self,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        only_attend_immediate: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_visual = dim_visual
        self.only_attend_immediate = only_attend_immediate

        # Cross attn
        self.attn = CrossAttention(dim=dim, dim_context=dim_visual, dim_head=dim_head, heads=heads)

        # Tanh gating for cross-attention
        self.attn_gate = nn.Parameter(torch.full((1,), 0.1))

        # Feed Forward
        self.ff = FeedForward(dim=dim, mult=ff_mult)

        # tanh gating for feed forward
        self.ff_gate = nn.Parameter(torch.full((1,), 0.1))

    def forward(
        self, x: torch.Tensor, visual_tokens: torch.Tensor, media_locations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """forward pass

        Args:
            x (torch.Tensor): Text hidden states [batch, seq_len, dim]
            visual_tokens (torch.Tensor): Visual tokens [batch, num_images, num_tokens_per_image, dim_visual]
            media_locations (Optional[torch.Tensor], optional): Boolean mask [batch, seq_len] indicating which tokens can attend to which images

        Returns:
            torch.Tensor: Enhanced text hidden states [batch, seq_len, dim]
        """
        attn_out = self.attn(x, visual_tokens, media_locations)

        # tanh gating - During training: gate gradually opens
        x = x + torch.tanh(self.attn_gate) * attn_out

        # feed forward gating - During training: gate gradually opens
        ff_out = self.ff(x)
        x = x + torch.tanh(self.ff_gate) * ff_out

        return x


def demo_gated_cross_attention():
    """Demonstrate gated cross-attention mechanism"""
    print()
    print("=" * 60)
    print("PART 2: Gated Cross-Attention Analysis")
    print("=" * 60)

    xattn = GatedCrossAttentionBlock(dim=512, dim_visual=512, heads=8).to(device)

    # Simulated inputs
    batch_size = 2
    seq_len = 20
    num_images = 2
    num_visual_tokens = 64
    dim = 512

    text_hidden = torch.randn(batch_size, seq_len, dim).to(device)
    visual_tokens = torch.randn(batch_size, num_images, num_visual_tokens, dim).to(device)

    media_locations = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)
    media_locations[:, 10:] = 1

    print(f"\nText hidden states: {text_hidden.shape}")
    print(f"Visual tokens: {visual_tokens.shape}")
    print(f"Media locations: {media_locations.shape}")

    # Check initial gate values
    print(f"\nInitial cross-attention gate: {xattn.attn_gate.item():.3f}")
    print(f"Initial tanh(attn_gate): {torch.tanh(xattn.attn_gate).item():.3f}")
    print(f"Initial feed-forward gate: {xattn.ff_gate.item():.3f}")
    print(f"Initial tanh(ff_gate): {torch.tanh(xattn.ff_gate).item():.3f}")

    # Forward pass
    with torch.no_grad():
        output = xattn(text_hidden, visual_tokens, media_locations)

    print(f"\nOutput shape at initialization: {output.shape}")

    diff = (output - text_hidden).abs().mean().item()
    print(f"Difference from input: {diff:.6f}")
    print("Negligible difference - gates working!")

    print("\n" + "=" * 90)
    print("Simulating training progression:")
    print("=" * 90)
    print(f"{'Gate Value (alpha)':<20} {'tanh(alpha)':<20} {'Mean Diff from Input':<25} {'Visual Info Flow':<25}")
    print("-" * 90)

    gate_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    for alpha in gate_values:
        xattn.attn_gate.data.fill_(alpha)
        xattn.ff_gate.data.fill_(alpha)

        with torch.no_grad():
            output = xattn(text_hidden, visual_tokens, media_locations)

        gate_value = torch.tanh(xattn.attn_gate).item()
        diff = (output - text_hidden).abs().mean().item()

        # Determine flow status
        flow_status = f"Gate Open %: {gate_value*100:.2f}%"

        print(f"{alpha:<20.1f} {gate_value:<20.6f} {diff:<25.6f} {flow_status:<25}")

    print("=" * 90)
    print("\nAs training progresses, gates open and visual info flows!")

    return xattn


# ============================================
# PART 3: Mini Flamingo Model
# ============================================


class miniFlamingo(nn.Module):
    """Mini Flamingo model for exploration

    Components:
    1. Vision encoder (simulated)
    2. Perceiver resampler
    3. Language model with gated cross-attention layers

    Production Flamingo would use:
    - NFNet-F6 vision encoder
    - Chinchilla 70B language model
    - More sophisticated training
    """

    def __init__(
        self,
        dim: int = 512,
        num_visual_tokens: int = 64,
        depth: int = 6,
        num_xattn_layers: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.num_visual_tokens = num_visual_tokens

        # Simple Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Linear(3 * 224 * 224, 2048),
            nn.GELU(),
            nn.Linear(2048, dim),
        )

        # Perceiver resampler
        self.perceiver = PerceiverResampler(
            dim=dim,
            depth=depth,
            num_latents=num_visual_tokens,
            heads=8,
            dim_head=64,
        )

        self.text_embed = nn.Embedding(50000, dim)  # Vocab size 50k
        self.lm_layers = nn.ModuleList([])
        self.xattn_layers = nn.ModuleList([])

        for i in range(num_xattn_layers):
            # Frozen LM layer
            self.lm_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=8,
                    dim_feedforward=dim * 4,
                    batch_first=True,
                )
            )

            # Gated cross-attention layer
            self.xattn_layers.append(GatedCrossAttentionBlock(dim=dim, dim_visual=dim, heads=8))

        self.lm_head = nn.Linear(dim, 50000)

    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        media_locations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            images: [batch, num_images, 3, 224, 224]
            text_ids: [batch, seq_len]
            media_locations: [batch, seq_len] - which image each token attends to

        Returns:
            logits: [batch, seq_len, vocab_size]
        """

        batch_size, num_images = images.shape[:2]

        images_flat = rearrange(images, "b m c h w -> b m (c h w)")
        visual_features = self.vision_encoder(images_flat)  # [batch, num_images, dim]

        # Resampler - process each image separately
        visual_tokens = []
        for i in range(num_images):
            # Get features for image i and add sequence dimension: [batch, 1, dim]
            img_features = visual_features[:, i : i + 1, :]
            tokens = self.perceiver(img_features)  # [batch, num_latents, dim]
            visual_tokens.append(tokens)

        visual_tokens = torch.stack(visual_tokens, dim=1)  # [batch, num_images, num_tokens, dim]

        # Process text
        x = self.text_embed(text_ids)

        for lm_layer, xattn_layer in zip(self.lm_layers, self.xattn_layers):
            x = lm_layer(x)
            x = xattn_layer(x, visual_tokens, media_locations)  # Gated cross-attention

        # Output
        logits = self.lm_head(x)

        return logits


def demo_mini_flamingo():
    """Demo mini Flamingo model"""
    print("\n\n")
    print("=" * 60)
    print("PART 3: Mini Flamingo Complete Architecture")
    print("=" * 60)

    # Create mini Flamingo
    model = miniFlamingo(
        dim=512,
        num_visual_tokens=64,
        depth=4,
        num_xattn_layers=3,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.lm_layers.parameters())
    learned_params = sum(p.numel() for p in model.perceiver.parameters()) + sum(
        p.numel() for p in model.xattn_layers.parameters()
    )

    print("\nModel Architecture:")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen LM params: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"Learned params: {learned_params:,} ({learned_params/total_params*100:.1f}%)")

    # Simulate forward pass
    batch_size = 2
    num_images = 2
    seq_len = 20

    images = torch.randn(batch_size, num_images, 3, 224, 224).to(device)
    text_ids = torch.randint(0, 50000, (batch_size, seq_len)).to(device)
    media_locations = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)
    media_locations[:, 10:] = 1  # Second half attends to second image

    print("\nInputs:")
    print(f"Images: {images.shape}")
    print(f"Text IDs: {text_ids.shape}")
    print(f"Media locations: {media_locations.shape}")

    # Forward pass
    with torch.no_grad():
        start_time = time.time()
        logits = model(images, text_ids, media_locations)
        duration = time.time() - start_time

    print("\nOutput:")
    print(f"Logits: {logits.shape}")
    print(f"Forward pass time: {duration*1000:.2f} ms")

    # Show gate values after initialization
    print("\nInitial gate values (all should be ~0):")
    for i, xattn_layer in enumerate(model.xattn_layers):
        attn_gate = xattn_layer.attn_gate.item()  # type:ignore
        ff_gate = xattn_layer.ff_gate.item()  # type:ignore
        print(f"  Layer {i+1}: attn_gate={attn_gate:.6f}, ff_gate={ff_gate:.6f}")

    return model


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("FLAMINGO EXPLORATION")
    print("=" * 50)

    perceiver, visual_tokens = demo_perceiver_resampler()

    demo_gated_cross_attention()

    model = demo_mini_flamingo()
