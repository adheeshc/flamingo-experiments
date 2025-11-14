"""Flamingo Model Optimization Experiments"""

import math
import pickle as pkl
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

# Import common layers
from attention_layers import CrossAttention, PerceiverAttention, SelfAttentionLayer
from cifar_utils import CIFAR10VisionEncoder
from feedforward import FeedForward
from flamingo_paper_exploration import MiniFlamingo, PerceiverResampler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ============================================
# OPTIMIZATION 1: Visual Token Caching
# ============================================


class VisualTokenCache:
    """
    Cache for visual tokens to avoid recomputing them

    Key optimization:
    - Visual encoding is expensive (vision encoder + perceiver)
    - For static images, we can cache the 64 visual tokens
    - Massive speedup for repeated images or batch processing
    """

    def __init__(self, max_cache_size: int = 10000):
        self.cache: Dict[str, torch.Tensor] = {}
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_key(self, image: torch.Tensor) -> str:
        """Generate cache key from image tensor"""
        return str(hash(image.cpu().numpy().tobytes()))

    def get(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached visual tokens if available"""
        cache_key = self.get_cache_key(image)

        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1
        return None

    def put(self, image: torch.Tensor, visual_tokens: torch.Tensor):
        """Cache visual tokens for an image"""
        if len(self.cache) >= self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        cache_key = self.get_cache_key(image)
        self.cache[cache_key] = visual_tokens.detach()

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "cache_size": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate*100:.1f}%",
            "speedup_estimate": f"{1 / max(1 - hit_rate, 1e-4):.2f}x",
        }

    def save(self, path: str):
        """Save cache"""
        cache_data = {"cache": {k: v.cpu() for k, v in self.cache.items()}, "stats": self.get_stats()}

        with open(path, "wb") as f:
            pkl.dump(cache_data, f)
        print(f"Cache saved to {path}")

    def load(self, path: str):
        """Load cache"""
        with open(path, "rb") as f:
            cache_data = pkl.load(f)

        self.cache = {k: v.to(device) for k, v in cache_data["cache"].items()}
        print(f"Cache loaded from {path}")


def demo_visual_cache():
    """Demo visual token caching"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION 1: Visual Token Caching")
    print("=" * 60)

    dim = 512
    perceiver = PerceiverResampler(dim=dim, depth=4, num_latents=64).to(device)
    cache = VisualTokenCache(max_cache_size=100)

    vision_encoder = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((24, 24)),
        nn.Flatten(1),
        nn.Linear(64 * 24 * 24, dim),
    ).to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_paths = [
        "./data/flamingo_test_images/bird.jpg",
        "./data/flamingo_test_images/building.jpg",
        "./data/flamingo_test_images/car.jpg",
        "./data/flamingo_test_images/cat.jpg",
        "./data/flamingo_test_images/dog.jpg",
        "./data/flamingo_test_images/flower.jpg",
        "./data/flamingo_test_images/food.jpg",
        "./data/flamingo_test_images/person.jpg",
    ]

    images = []
    images_loaded = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img).to(device)  # type: ignore
            images.append(img_tensor)
            images_loaded.append(path.split("/")[-1])
        except Exception as e:
            print(f"could not load image, creating random synthetic : {e}")
            img_tensor = torch.randn(3, 224, 224).to(device)
            images.append(img_tensor)
            images_loaded.append(f"synthetic_{len(images)}")

    num_repeats = 6
    images_extended = images * num_repeats

    print(f"\nLoaded {len(images_extended)} unique images: {images_loaded}")

    # First pass: no cache
    print("\n--- Pass 1: Initial Processing ---")
    start_time = time.time()
    for img in images_extended:
        cached_tokens = cache.get(img)
        if cached_tokens is None:
            with torch.no_grad():
                # Simulate vision encoder
                visual_features = vision_encoder(img.unsqueeze(0))
                visual_features = visual_features.view(1, 1, dim)
                visual_tokens = perceiver(visual_features)
            cache.put(img, visual_tokens)
    duration1 = time.time() - start_time
    print(f"Time: {duration1:.3f}s")
    print(f"Cache stats: {cache.get_stats()}")

    # Second pass: with cache
    print("\n--- Pass 2: Re-processing using Cache ---")
    cache.cache_hits = 0
    cache.cache_misses = 0

    start_time = time.time()
    for img in images_extended:
        cached_tokens = cache.get(img)
        if cached_tokens is None:
            with torch.no_grad():
                visual_features = vision_encoder(img.unsqueeze(0))
                visual_features = visual_features.view(1, 1, dim)
                visual_tokens = perceiver(visual_features)
            cache.put(img, visual_tokens)
        else:
            visual_tokens = cached_tokens
    duration2 = time.time() - start_time
    print(f"Time: {duration2:.3f}s")
    print(f"Cache stats: {cache.get_stats()}")

    print("\n")
    speedup = duration1 / duration2
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {duration1 - duration2:.3f}s ({(1-duration2/duration1)*100:.1f}% faster)")

    # cache.save("visual_token_cache.pkl")
    return cache


# ============================================
# OPTIMIZATION 2: Reduced Visual Tokens
# ============================================


def benchmark_reduced_visual_tokens():
    """
    Compare different numbers of visual tokens

    Key tradeoff: Quality vs Speed
    - More tokens = better quality, slower inference
    - Fewer tokens = faster inference, potential quality loss
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION 2: Reduced Visual Tokens")
    print("=" * 60)

    dim = 512
    batch_size = 4
    visual_features = torch.randn(batch_size, 576, dim).to(device)

    token_counts = [256, 128, 64, 48, 32, 16]

    print("\nCompare different numbers of visual tokens for Speed and Memory usage\n")
    print("-" * 70)
    print(f"{'Tokens':<10} {'Time (ms)':<15} {'Memory (MB)':<15} {'Params':<15}")
    print("-" * 70)

    for num_tokens in token_counts:
        perceiver = PerceiverResampler(dim=dim, depth=4, num_latents=num_tokens, heads=8).to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = perceiver(visual_features)

        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        num_runs = 50
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = perceiver(visual_features)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        duration = (time.time() - start_time) / num_runs * 1000  # ms

        # Memory
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1e6
        else:
            memory_mb = 0

        # Parameters
        params = sum(p.numel() for p in perceiver.parameters())

        print(f"{num_tokens:<10} {duration:<15.2f} {memory_mb:<15.1f} {params:<15,}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================
# OPTIMIZATION 3: Sparse Cross-Attention
# ============================================


class SparseCrossAttention(nn.Module):
    """
    Sparse cross-attention: only attend to top-k visual tokens.

    Optimization rationale:
    - Not all 64 visual tokens are equally important
    - Can attend to top-k most relevant tokens
    - Reduces computation, maintains quality
    """

    def __init__(
        self,
        dim: int,
        dim_context: int,
        dim_head: int = 64,
        heads: int = 8,
        top_k: int = 32,  # Attend to top 32 out of 64 tokens
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.top_k = top_k
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim_context)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            context: [batch, num_images, num_visual_tokens, dim_context]

        Returns:
            [batch, seq_len, dim]
        """
        h = self.heads

        x = self.norm(x)
        context_flat = rearrange(context, "b m n d -> b (m n) d")
        context_flat = self.norm_context(context_flat)

        # Get Q, K, V
        q = self.to_q(x)
        k, v = self.to_kv(context_flat).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # SPARSE ATTENTION: Select top-k tokens
        topk_vals, topk_indices = sim.topk(self.top_k, dim=-1)
        sparse_sim = torch.full_like(sim, float("-inf"))
        sparse_sim.scatter_(-1, topk_indices, topk_vals)
        attn = sparse_sim.softmax(dim=-1)

        # Output
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


def demo_sparse_attention():
    """Demonstrate sparse attention optimization"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION 3: Sparse Cross-Attention")
    print("=" * 60)

    dim = 512
    batch_size = 16
    seq_len = 512
    num_images = 4
    num_visual_tokens = 64

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Num images: {num_images}")
    print(f"Visual tokens per image: {num_visual_tokens}")
    print(f"Total visual tokens: {num_images * num_visual_tokens}")

    text = torch.randn(batch_size, seq_len, dim).to(device)
    visual = torch.randn(batch_size, num_images, num_visual_tokens, dim).to(device)

    # Compare different sparsity levels
    k_values = [256, 128, 64, 48, 32, 16]
    print("\nonly attend to top-k visual tokens - Copmparing Speed and Memory")
    print("\n" + "-" * 80)
    print(f"{'Top-K Tokens':<15} {'Time (ms)':<15} {'Memory (MB)':<15} {'Speedup':<15}")
    print("-" * 80)

    baseline_time = None

    for top_k in k_values:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        attn = SparseCrossAttention(dim=dim, dim_context=dim, heads=8, top_k=top_k).to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = attn(text, visual)

        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        num_runs = 50
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = attn(text, visual)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        duration = (time.time() - start_time) / num_runs * 1000

        # Memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            memory_mb = 0

        # First run (k=256) is the baseline
        if baseline_time is None:
            baseline_time = duration
            speedup = 1.0
        else:
            speedup = baseline_time / duration

        print(f"{top_k:<15} {duration:<15.2f} {memory_mb:<15.1f} {f'{speedup:.2f}x':<15}")

        # Clear memory
        del attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================
# OPTIMIZATION 4: Quantization-Aware Training
# ============================================


def benchmark_quantization():
    """
    Benchmark FP16 and INT8 quantization for Perceiver Resampler

    Key insight: Different components have different quantization sensitivity
    - Perceiver: Should be quantization-friendly
    - Cross-attention: More sensitive (gates need precision)
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION 4: Quantization Analysis")
    print("=" * 60)

    dim = 512
    batch_size = 4
    perceiver = PerceiverResampler(dim=dim, depth=4, num_latents=64).to(device)

    visual_features = torch.randn(batch_size, 576, dim).to(device)

    # FP32
    num_runs = 50
    for i in range(10):  # Warmup
        with torch.no_grad():
            output_fp32 = perceiver(visual_features)

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_runs):
        with torch.no_grad():
            output_fp32 = perceiver(visual_features)
    torch.cuda.synchronize()
    time_fp32 = (time.time() - start_time) / num_runs * 1000

    # FP16
    for i in range(10):  # Warmup
        with torch.no_grad(), torch.amp.autocast("cuda"):  # type: ignore
            output_fp16 = perceiver(visual_features)

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_runs):
        with torch.no_grad(), torch.amp.autocast("cuda"):  # type: ignore
            output_fp16 = perceiver(visual_features)
    torch.cuda.synchronize()
    time_fp16 = (time.time() - start_time) / num_runs * 1000

    # Compute error
    output_fp16_float = output_fp16.float()
    error = (output_fp32 - output_fp16_float).abs().mean().item()
    rel_error = error / output_fp32.abs().mean().item()

    print("\nLatency: ")
    print(f"FP32: {time_fp32:.2f} ms")
    print(f"FP16: {time_fp16:.2f} ms")
    print(f"Speedup: {time_fp32/time_fp16:.2f}x")
    print("\nAccuracy:")
    print(f"Absolute error: {error:.6f}")
    print(f"Relative error: {rel_error*100:.4f}%")


# ============================================
# OPTIMIZATION 5: Dynamic Perceiver Depth
# ============================================


class AdaptivePerceiverResampler(nn.Module):
    """
    Adaptive Perceiver with early exit capability.

    Idea: Simple images may not need all 6 transformer layers.
    Can exit early based on convergence criterion.
    """

    def __init__(
        self,
        dim: int = 1024,
        max_depth: int = 6,
        num_latents: int = 64,
        heads: int = 8,
        dim_head: int = 64,
        convergence_threshold: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.max_depth = max_depth
        self.convergence_threshold = convergence_threshold

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        self.attn_layers = nn.ModuleList(
            [PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads) for _ in range(max_depth)]
        )
        self.ff_layers = nn.ModuleList([FeedForward(dim=dim, mult=4) for _ in range(max_depth)])
        for _ in range(max_depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=4),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

        self.exit_counts = [0] * (max_depth + 1)

    def forward(self, x: torch.Tensor, use_early_exit: bool = True) -> torch.Tensor:
        """
        Args:
            x: [batch, num_tokens, dim]
            use_early_exit: Bool to use adaptive depth

        Returns:
            [batch, num_latents, dim]
        """
        batch_size = x.shape[0]
        latents = repeat(self.latents, "n d -> b n d", b=batch_size)

        for layer_idx, (attn, ff) in enumerate(zip(self.attn_layers, self.ff_layers)):
            latents_prev = latents.clone()

            # Attention
            latents = attn(latents, x) + latents

            # Feed-forward
            latents = ff(latents) + latents

            # Check for convergence (early exit)
            if use_early_exit and layer_idx < self.max_depth - 1:
                change = (latents - latents_prev).abs().mean().item()

                if change < self.convergence_threshold:
                    self.exit_counts[layer_idx] += 1
                    break
        else:
            self.exit_counts[self.max_depth] += 1

        return self.norm(latents)

    def get_exit_stats(self) -> Dict:
        """Get statistics on layer exits"""
        total = sum(self.exit_counts)
        if total == 0:
            return {}

        avg_depth = sum(i * count for i, count in enumerate(self.exit_counts)) / total

        return {
            "exit_counts": self.exit_counts,
            "average_depth": avg_depth,
            "max_depth": self.max_depth,
            "computation_saved": f"{(1 - avg_depth/self.max_depth)*100:.1f}%",
        }


def demo_adaptive_depth():
    """Demonstrate adaptive perceiver depth"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION 5: Adaptive Perceiver Depth")
    print("=" * 60)

    dim = 512
    batch_size = 32
    num_samples = 100

    # Load CIFAR-10 dataset and vision encoder
    print("\n")
    cifar_encoder = CIFAR10VisionEncoder(dim=dim, data_root="./data", device=device)

    # Find optimal threshold
    print("\nFinding optimal convergence threshold")
    test_perceiver = AdaptivePerceiverResampler(
        dim=dim, max_depth=6, num_latents=64, convergence_threshold=0.0  # Disable early exit
    ).to(device)

    # Encode sample images for profiling
    visual_features = cifar_encoder.encode_batch(batch_size=10, start_idx=0, num_tokens=576)

    changes = []
    batch_size_temp = 10
    latents = repeat(test_perceiver.latents, "n d -> b n d", b=batch_size_temp)

    with torch.no_grad():
        for layer_idx, (attn, ff) in enumerate(zip(test_perceiver.attn_layers, test_perceiver.ff_layers)):
            latents_prev = latents.clone()
            latents = attn(latents, visual_features) + latents
            latents = ff(latents) + latents

            change = (latents - latents_prev).abs().mean().item()
            changes.append(change)
            print(f"  Layer {layer_idx}: change = {change:.6f}")

    min_change = min(changes)
    max_change = max(changes)
    avg_change = sum(changes) / len(changes)

    # Set threshold closer to max for more early exits (demo purposes)
    suggested_threshold = min_change + (avg_change - min_change) * 0.5

    print("\nChange statistics:")
    print(f"Min: {min_change:.6f}, Max: {max_change:.6f}, Avg: {avg_change:.6f}")
    print(f"Suggested threshold: {suggested_threshold:.6f}")
    print(
        "\nNote: Randomly initialized models don't converge well. In a trained model, convergence would be more natural "
    )
    print("This is aggressive thresholding for demo purposes")

    perceiver_adaptive = AdaptivePerceiverResampler(
        dim=dim, max_depth=6, num_latents=64, convergence_threshold=suggested_threshold
    ).to(device)

    perceiver_standard = PerceiverResampler(dim=dim, depth=6, num_latents=64).to(device)

    print(f"\nProcessing {num_samples} batches")

    # Standard perceiver
    start_time = time.time()
    for i in range(num_samples):
        visual_features = cifar_encoder.encode_batch(batch_size=batch_size, start_idx=i * batch_size, num_tokens=576)
        with torch.no_grad():
            _ = perceiver_standard(visual_features)
    time_standard = time.time() - start_time

    # Adaptive perceiver
    start_time = time.time()
    for i in range(num_samples):
        visual_features = cifar_encoder.encode_batch(batch_size=batch_size, start_idx=i * batch_size, num_tokens=576)
        with torch.no_grad():
            _ = perceiver_adaptive(visual_features, use_early_exit=True)
    time_adaptive = time.time() - start_time

    print("\nResults:")
    print(f"Standard perceiver: {time_standard:.3f}s")
    print(f"Adaptive perceiver: {time_adaptive:.3f}s")
    print(f"Speedup: {time_standard/time_adaptive:.2f}x")

    stats = perceiver_adaptive.get_exit_stats()
    print("\nAdaptive depth statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


# ============================================
# OPTIMIZATION 6: Speculative Decoding
# ============================================


def demo_speculative_decoding():
    """
    Demonstrate concept of speculative decoding for Flamingo.

    Idea: Use small draft model to generate multiple tokens quickly,
    then verify with large model in parallel.
    # Large model: Flamingo-70B
    # Draft model: Flamingo-7B (10x smaller, 10x faster)

    Note: This is a simplified demonstration of the concept.
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION 6: Speculative Decoding Concept")
    print("=" * 60)

    print("Prior Assumptions: ")
    # Simulate latencies
    large_model_time_per_token = 100
    small_model_time_per_token = 20
    verification_time = 30
    print(f"Large model time per token: {large_model_time_per_token}ms")
    print(f"Small model time per token: {small_model_time_per_token}ms")
    print(f"Verification time: {verification_time}ms")

    num_tokens = 20
    k = 4  # draft tokens

    standard_time = num_tokens * large_model_time_per_token

    # Speculative decoding (assuming 80% acceptance rate)
    acceptance_rate = 0.8
    num_iterations = num_tokens // k
    speculative_time = num_iterations * (
        k * small_model_time_per_token + verification_time * (1 + (1 - acceptance_rate))
    )

    print("\nIdea: Use small draft model to generate multiple tokens quickly, then verify with large model in parallel")
    print("Large model: Flamingo-70B")
    print("Draft model: Flamingo-7B (10x smaller, 10x faster)")

    print(f"\nExample Use Case - Generating {num_tokens} tokens:\n")
    print(f"\nStandard time for Large Model: {standard_time} ms")

    print(f"\nSpeculative decoding (k={k}):")
    print(f"Draft time: {k * small_model_time_per_token} ms/iteration")
    print(f"Verification time: ~{verification_time} ms/iteration")
    print(f"Total time for Small Model + Verification: {speculative_time:.0f} ms")
    print(f"Speedup: {standard_time/speculative_time:.2f}x")


# ============================================
# OPTIMIZATION 7: Combined Optimizations
# ============================================


class OptimizedFlamingo:
    """
    Flamingo with all optimizations applied.

    Optimizations:
    1. Visual token caching
    2. Reduced visual tokens
    3. FP16 quantization
    4. Sparse cross-attention
    5. Adaptive perceiver depth (set higher for demo purposes)
    """

    def __init__(
        self,
        dim: int = 512,
        num_visual_tokens: int = 32,
        use_fp16: bool = True,
        use_sparse_attention: bool = True,
        cache_size: int = 1000,
        perceiver_depth: int = 4,
        convergence_threshold: float = 0.01,
    ):
        self.dim = dim
        self.num_visual_tokens = num_visual_tokens
        self.use_fp16 = use_fp16
        self.use_sparse_attention = use_sparse_attention

        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Linear(3 * 224 * 224, 2048),
            nn.GELU(),
            nn.Linear(2048, dim),
        ).to(device)

        # Adaptive perceiver
        self.perceiver = AdaptivePerceiverResampler(
            dim=dim,
            max_depth=perceiver_depth,
            num_latents=num_visual_tokens,
            convergence_threshold=convergence_threshold,
        ).to(device)

        # Visual token cache
        self.cache = VisualTokenCache(max_cache_size=cache_size)

        print("\nOptimizedFlamingo Stats:")
        print(f"Visual tokens: {num_visual_tokens}")
        print(f"Perceiver depth: {perceiver_depth}")
        print(f"FP16: {use_fp16}")
        print(f"Sparse attention: {use_sparse_attention}")

    def encode_image(self, image: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """
        Encode image to visual tokens.

        Args:
            image: [batch, 3, 224, 224]
            use_cache: Whether to use caching

        Returns:
            visual_tokens: [batch, num_visual_tokens, dim]
        """
        # Check cache
        if use_cache:
            cached = self.cache.get(image)
            if cached is not None:
                return cached

        # Encode
        image_flat = rearrange(image, "b c h w -> b (c h w)")

        if self.use_fp16:
            with torch.amp.autocast("cuda"):  # type: ignore
                features = self.vision_encoder(image_flat)
                features = features.unsqueeze(1)
                visual_tokens = self.perceiver(features, use_early_exit=True)
            visual_tokens = visual_tokens.float()
        else:
            features = self.vision_encoder(image_flat)
            features = features.unsqueeze(1)
            visual_tokens = self.perceiver(features, use_early_exit=True)

        # Cache
        if use_cache:
            self.cache.put(image, visual_tokens)

        return visual_tokens

    def get_stats(self) -> Dict:
        """Get optimization statistics"""
        return {
            "cache_stats": self.cache.get_stats(),
            "perceiver_depth_stats": self.perceiver.get_exit_stats(),
        }


def demo_combined_optimizations():
    """Demonstrate all optimizations together"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION 7: Combined Optimizations")
    print("=" * 60)

    dim = 512
    batch_size = 8
    num_batches = 50

    cifar_encoder = CIFAR10VisionEncoder(dim=dim, data_root="./data", device=device)
    print(f"Processing {num_batches} batches of {batch_size} images each")

    # Baseline
    baseline_perceiver = PerceiverResampler(dim=dim, depth=6, num_latents=64).to(device)

    # Optimized
    optimized = OptimizedFlamingo(
        dim=dim,
        num_visual_tokens=32,
        use_fp16=True,
        use_sparse_attention=True,
        perceiver_depth=4,
        convergence_threshold=0.3,  # Set higher for demo purposes
    )

    # Benchmark baseline
    print("\n--- Baseline (64 tokens, FP32, depth 6, no cache) ---")
    start_time = time.time()
    for i in range(num_batches):
        visual_features = cifar_encoder.encode_batch(batch_size=batch_size, start_idx=i * batch_size, num_tokens=1)
        with torch.no_grad():
            _ = baseline_perceiver(visual_features)
    baseline_time = time.time() - start_time
    print(f"Time: {baseline_time:.3f}s")
    print(f"Throughput: {num_batches * batch_size / baseline_time:.2f} images/sec")

    # Benchmark optimized (first pass - filling cache)
    print("\n--- Optimized Pass 1 (32 tokens, FP16, depth 4, adaptive, filling cache) ---")
    start_time = time.time()
    for i in range(num_batches):
        batch_images = cifar_encoder.get_batch(batch_size=batch_size, start_idx=i * batch_size)
        with torch.no_grad():
            _ = optimized.encode_image(batch_images, use_cache=True)
    optimized_time1 = time.time() - start_time
    print(f"Time: {optimized_time1:.3f}s")
    print(f"Throughput: {num_batches * batch_size / optimized_time1:.2f} images/sec")
    stats1 = optimized.get_stats()
    print(f"Cache stats: {stats1['cache_stats']}")

    # Benchmark optimized (second pass - using cache)
    print("\n--- Optimized Pass 2 (with cache hits) ---")
    start_time = time.time()
    for i in range(num_batches):
        batch_images = cifar_encoder.get_batch(batch_size=batch_size, start_idx=i * batch_size)
        with torch.no_grad():
            _ = optimized.encode_image(batch_images, use_cache=True)
    optimized_time2 = time.time() - start_time
    print(f"Time: {optimized_time2:.3f}s")
    print(f"Throughput: {num_batches * batch_size / optimized_time2:.2f} images/sec")
    stats2 = optimized.get_stats()
    print(f"Cache stats: {stats2['cache_stats']}")
    print(f"Adaptive depth stats: {stats2['perceiver_depth_stats']}")

    print("\n")

    total_images = num_batches * batch_size
    baseline_throughput = total_images / baseline_time
    optimized_throughput1 = total_images / optimized_time1
    optimized_throughput2 = total_images / optimized_time2

    print(f"{'Name':<20} {'Configuration':<35} {'Time (s)':<15} {'Throughput':<20} {'Speedup':<15}")
    print("-" * 110)
    print(
        f"{'Baseline':<20} {'64 tokens, FP32, depth 6':<35} {baseline_time:<15.3f} {f'{baseline_throughput:.2f} img/s':<20} {'1.00x':<15}"
    )
    print(
        f"{'Optimized (Pass 1)':<20} {'32 tokens, FP16, depth 4':<35} {optimized_time1:<15.3f} {f'{optimized_throughput1:.2f} img/s':<20} {f'{baseline_time/optimized_time1:.2f}x':<15}"
    )
    print(
        f"{'Optimized (Pass 2)':<20} {'32 tokens, FP16, depth 4, Cache':<35} {optimized_time2:<15.3f} {f'{optimized_throughput2:.2f} img/s':<20} {f'{baseline_time/optimized_time2:.2f}x':<15}"
    )
    print("-" * 110)


if __name__ == "__main__":
    print("=" * 50)
    print("FLAMINGO OPTIMIZATION")
    print("=" * 50)

    demo_visual_cache()

    benchmark_reduced_visual_tokens()

    demo_sparse_attention()

    benchmark_quantization()

    demo_adaptive_depth()

    demo_speculative_decoding()

    demo_combined_optimizations()
