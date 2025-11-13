"""Common Attention Layer Implementations"""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class PerceiverAttention(nn.Module):
    """
    Cross-attention layer for Perceiver Resampler

    KEY INNOVATION: concatenates latent (K,V) with input (K,V) which allows latents to attend to both:
    1. The visual features (cross-attention)
    2. Each other (self-attention component)

    Args:
        dim: Input dimension
        dim_head: Dimension per attention head (default: 64)
        heads: Number of attention heads (default: 8)
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


class SelfAttentionLayer(nn.Module):
    """Standard self-attention layer

    Multi-head self-attention with layer normalization.

    Args:
        dim: Input dimension
        dim_head: Dimension per attention head (default: 64)
        heads: Number of attention heads (default: 8)
    """

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


class CrossAttention(nn.Module):
    """Cross Attention: queries from text, (k,v) from visual tokens

    Implements image-causal masking: each text token only attends to
    the visual tokens from its immediately preceding image.

    Args:
        dim: Query dimension (text)
        dim_context: Key/Value dimension (visual)
        dim_head: Dimension per attention head (default: 64)
        heads: Number of attention heads (default: 8)
    """

    def __init__(self, dim: int, dim_context: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim_context)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, media_locations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """forward pass

        Args:
            x (torch.Tensor): [batch, seq_len, dim] - text (queries)
            context (torch.Tensor):  [batch, num_images, num_visual_tokens, dim_context] - visual tokens (keys/values)
            media_locations (Optional[torch.Tensor], optional): [batch, seq_len] - which image each token attends to

        Returns:
            torch.Tensor: [batch, seq_len, dim]
        """
        h = self.heads
        x = self.norm(x)

        # Flatten visual tokens: [batch, num_images * num_visual_tokens, dim]
        context_flat = rearrange(context, "b m n d -> b (m n) d")
        context_flat = self.norm_context(context_flat)

        # get Q, K, V
        q = self.to_q(x)
        k, v = self.to_kv(context_flat).chunk(2, dim=-1)

        # reshape for multihead
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        # attention
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # apple causal masking
        if media_locations is not None:
            sim = self._apply_media_mask(sim, media_locations, context.shape[1], context.shape[2])

        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)

    def _apply_media_mask(
        self, sim: torch.Tensor, media_locations: torch.Tensor, num_images: int, tokens_per_image: int
    ) -> torch.Tensor:
        """Apple Image-Causal Masking

        Each text token at position i can only attend to visual tokens
        from image media_locations[i].

        Args:
            sim (torch.Tensor): Similarity matrix of shape [batch, heads, seq_len, num_visual_tokens]
            media_locations (torch.Tensor): Image index (0 to num_images-1) for each query token
            num_images (int): Total number of images in the batch
            tokens_per_image (int): Number of visual tokens extracted for each image

        Returns:
            torch.Tensor: similarity matrix with masked (disallowed) logits
        """
        # create mask
        mask = torch.zeros_like(sim)

        for batch_idx in range(sim.shape[0]):
            for token_idx in range(sim.shape[2]):
                image_idx = media_locations[batch_idx, token_idx]
                if image_idx >= 0:
                    start_idx = image_idx * tokens_per_image
                    end_idx = start_idx + tokens_per_image
                    mask[batch_idx, :, token_idx, start_idx:end_idx] = 1.0

        sim = sim.masked_fill(mask == 0, float("-inf"))

        return sim
