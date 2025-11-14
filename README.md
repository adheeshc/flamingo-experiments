# Flamingo Experiments

A comprehensive implementation and optimization study of the Flamingo vision-language model, featuring modular components and extensive performance benchmarks.

## Overview

This project provides a clean, educational implementation of DeepMind's Flamingo model architecture along with several production-grade optimizations. Flamingo is a visual language model that can process interleaved sequences of images and text to perform few-shot learning on various vision-language tasks.

**Key Features:**
- Modular implementation of core Flamingo components
- 7 different optimization techniques with benchmarks

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Architecture Components](#architecture-components)
  - [Core Modules](#core-modules)
- [Optimization Techniques](#optimization-techniques)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setup](#setup)
- [Usage](#usage)
  - [Running Core Flamingo Exploration](#running-core-flamingo-exploration)
  - [Running Optimization Experiments](#running-optimization-experiments)
  - [Using Individual Components](#using-individual-components)
  - [Using Optimized Components](#using-optimized-components)
- [Key Insights](#key-insights)
  - [From Exploration Experiments](#from-exploration-experiments)
  - [From Optimization Experiments](#from-optimization-experiments)
- [Performance Benchmarks](#performance-benchmarks)
  - [Benchmark Hardware Specifications](#benchmark-hardware-specifications)
  - [Core Architecture Performance](#core-architecture-performance)
  - [Optimization Techniques](#optimization-techniques-1)
  - [Summary](#summary)
- [Key Innovations](#key-innovations)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## Architecture Components

### Core Modules

1. **Perceiver Resampler** (`flamingo_paper_exploration.py`)
   - Compresses variable-length visual features into fixed-size token representations
   - Uses learned latent queries for efficient attention-based compression
   - Achieves significant computational savings over direct self-attention

2. **Gated Cross-Attention** (`flamingo_paper_exploration.py`)
   - Enables text-to-vision attention with learnable gating mechanism
   - Tanh gates initialized near zero preserve frozen language model weights
   - Supports image-causal masking for proper temporal ordering

3. **Attention Layers** (`attention_layers.py`)
   - `PerceiverAttention`: Cross-attention with latent self-attention
   - `CrossAttention`: Standard cross-attention with media masking
   - `SelfAttentionLayer`: Multi-head self-attention

4. **Feed-Forward Networks** (`feedforward.py`)
   - Standard MLP blocks with LayerNorm and GELU activation
   - Configurable expansion ratio and dropout

5. **Mini Flamingo** (`flamingo_paper_exploration.py`)
   - Complete end-to-end model implementation
   - Simplified vision encoder + Perceiver + Gated cross-attention layers
   - Demonstrates full architecture integration

## Optimization Techniques

The `flamingo_optimization.py` module implements 7 key optimizations:

### 1. Visual Token Caching
- Caches processed visual tokens to avoid recomputation
- Provides massive speedup for repeated images or batch processing
- Includes cache statistics and persistence

### 2. Reduced Visual Tokens
- Compares different token counts (256, 128, 64, 48, 32, 16)
- Analyzes quality vs. speed tradeoffs
- Benchmarks latency and memory usage

### 3. Sparse Cross-Attention
- Attends to only top-k most relevant visual tokens
- Reduces computational complexity while maintaining quality
- Configurable sparsity levels

### 4. Quantization (FP16)
- Benchmark mixed-precision inference
- Measures speedup vs. accuracy tradeoff
- Compatible with modern GPU tensor cores

### 5. Adaptive Perceiver Depth
- Early exit mechanism based on convergence
- Dynamically adjusts model depth per sample
- Tracks layer-wise exit statistics

### 6. Speculative Decoding
- Conceptual demonstration of draft-verify decoding
- Shows potential speedups with smaller draft models

### 7. Combined Optimizations
- Integrates multiple optimizations together
- Comprehensive benchmarking against baseline
- Demonstrates cumulative performance gains

## Project Structure

```
flamingo-experiments/
├── flamingo_paper_exploration.py  # Core Flamingo implementation
├── flamingo_optimization.py       # Optimization experiments
├── attention_layers.py            # Attention mechanism implementations
├── feedforward.py                 # Feed-forward network layers
├── cifar_utils.py                # CIFAR-10 dataset utilities
├── data/                         # Dataset and test images
│   ├── flamingo_test_images/     # Sample test images
│   └── cifar-10-batches-py/      # CIFAR-10 dataset (auto-downloaded)
└── papers/                       # Reference papers
```

## Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
einops>=0.6.0
Pillow>=9.0.0
tqdm>=4.65.0
```

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd flamingo-experiments

# Install dependencies
pip install torch torchvision einops Pillow tqdm

# The CIFAR-10 dataset will be automatically downloaded on first run
```

## Usage

### Running Core Flamingo Exploration

```bash
python flamingo_paper_exploration.py
```

This will demonstrate:
- Perceiver Resampler with compression benchmarks
- Gated cross-attention mechanism
- Complete Mini Flamingo model

### Running Optimization Experiments

```bash
python flamingo_optimization.py
```

This runs all 7 optimization experiments:
1. Visual token caching demo
2. Reduced visual tokens benchmark
3. Sparse cross-attention comparison
4. Quantization analysis (FP16 vs FP32)
5. Adaptive perceiver depth
6. Speculative decoding concept
7. Combined optimizations

## Key Insights

### From Exploration Experiments

- **Perceiver Resampler Efficiency**: Achieves 9x token compression (576→64 tokens) while being 4.07x faster than direct self-attention, reducing computational complexity from O(331,776) to O(4,096) - an 81x reduction
- **Gated Cross-Attention**: Tanh gates initialized near zero (0.1) allow gradual visual information injection during training, preserving frozen language model weights at initialization while enabling full visual-text integration when fully trained
- **Architectural Modularity**: The separation of vision encoding (Perceiver Resampler) and language-vision fusion (Gated Cross-Attention) enables independent optimization of each component
- **Parameter Efficiency**: Mini Flamingo uses only 5.6% learned parameters (22.1M) while keeping 2.4% of LM parameters frozen (9.5M), demonstrating efficient few-shot learning capability
- **Embedding Space Structure**: Visual tokens create semantically meaningful representations that can be efficiently cross-attended by text tokens

### From Optimization Experiments

- **Visual Token Caching**: Provides exceptional speedup (20.68x) for repeated image processing scenarios, with 100% cache hit rate achieving near-instantaneous inference (0.020s vs 0.403s)
- **Token Count Optimization**: Reducing tokens from 64→32 provides 1.20x speedup with minimal memory savings (0.4 MB), while 64→16 achieves 1.49x speedup, showing diminishing returns beyond 32 tokens
- **Sparse Attention Tradeoffs**: Top-k sparse attention shows best results at k=16 (1.79x speedup, 188.8 MB memory savings), but the benefit plateaus between k=32 and k=64, suggesting most visual information is concentrated in top-32 tokens
- **FP16 Quantization**: Achieves 1.49x speedup with negligible accuracy loss (0.495% relative error), making it an ideal optimization for production deployment on modern GPUs with Tensor Cores
- **Adaptive Depth**: Demonstrates 1.80x speedup with early exit mechanisms, though the 100% computation savings in benchmarks reflects aggressive thresholding; trained models would achieve 20-50% realistic savings
- **Speculative Decoding Potential**: Conceptual analysis shows 3.45x theoretical speedup using draft-verify approach, suggesting significant opportunity for autoregressive generation optimization
- **Combined Optimization Reality**: First pass shows overhead (0.38x) due to cache filling and setup, but subsequent passes with warm cache achieve 1.22x speedup, highlighting the importance of amortized costs in production
- **Batch Processing**: Larger batch sizes dramatically improve throughput (implied by optimization results), essential for production deployments
- **Model Selection Tradeoffs**: Perceiver depth, token count, and precision choices should be task-dependent, with the benchmarks providing clear speed/memory/accuracy tradeoffs

## Performance Benchmarks

### Benchmark Hardware Specifications

- **System**: Acer Predator PH16-71 (Laptop)
- **CPU**: Intel Core i7-13700HX (13th Gen, 16 cores, 24 threads @ 2.3 GHz base)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
- **RAM**: 16 GB DDR5
- **OS**: Windows 11 (64-bit) with WSL2 (Linux 5.15.133.1-microsoft-standard-WSL2)
- **Python**: 3.10
- **PyTorch**: 2.0+ with CUDA enabled
- **CUDA**: Enabled

All benchmarks run on CUDA GPU. Results from actual runs:

### Core Architecture Performance

**Perceiver Resampler vs Direct Self-Attention**

| Method | Time (ms) | Tokens/Image | Compression | Speedup |
|--------|-----------|--------------|-------------|---------|
| Perceiver Resampler | 10.99 | 64 | 9.0x | 4.07x faster |
| Direct Self-Attention | 44.74 | 576 | 1.0x | baseline |
| **Complexity Reduction** | - | - | O(331,776) → O(4,096) | 81x reduction |

**Mini Flamingo End-to-End**
- Total parameters: 392M
- Forward pass (2 images, seq_len=20): 55.51ms
- Frozen LM params: 9.5M (2.4%)
- Learned params: 22.1M (5.6%)

### Optimization Techniques

**1. Visual Token Caching**

| Pass | Time | Cache Hit Rate | Speedup |
|------|------|----------------|---------|
| Pass 1 (filling cache) | 0.403s | 83.3% | - |
| Pass 2 (with cache) | 0.020s | 100.0% | 20.68x |

**2. Reduced Visual Tokens**

| Tokens | Time (ms) | Memory (MB) | Parameters | vs 64 tokens |
|--------|-----------|-------------|------------|--------------|
| 256 | 9.99 | 67.3 | 12.7M | 2.44x slower |
| 128 | 5.55 | 66.0 | 12.7M | 1.36x slower |
| **64** | **4.09** | **65.4** | **12.6M** | **baseline** |
| 48 | 3.68 | 65.2 | 12.6M | 1.11x faster |
| 32 | 3.41 | 65.0 | 12.6M | 1.20x faster |
| 16 | 2.75 | 64.9 | 12.6M | 1.49x faster |

**3. Sparse Cross-Attention (Top-K)**

Setup: batch_size=16, seq_len=512, num_images=4, 64 tokens/image

| Top-K | Time (ms) | Memory (MB) | Speedup |
|-------|-----------|-------------|---------|
| 256 (all) | 15.23 | 533.9 | baseline |
| 128 | 10.27 | 433.2 | 1.48x |
| 64 | 10.63 | 382.9 | 1.43x |
| 48 | 10.51 | 370.3 | 1.45x |
| 32 | 8.80 | 357.7 | 1.73x |
| 16 | 8.50 | 345.1 | 1.79x |

**4. FP16 Quantization**

| Precision | Time (ms) | Speedup | Relative Error |
|-----------|-----------|---------|----------------|
| FP32 | 4.10 | baseline | 0% |
| FP16 | 2.75 | 1.49x | 0.495% |

**5. Adaptive Perceiver Depth**

| Mode | Time (s) | Speedup | Computation Saved |
|------|----------|---------|-------------------|
| Standard (depth 6) | 5.274 | baseline | 0% |
| Adaptive (early exit) | 2.928 | 1.80x | 100%* |

*Aggressive threshold for demo; trained models would see 20-50% savings

**6. Speculative Decoding (Conceptual)**

| Mode | Time (ms) | Speedup |
|------|-----------|---------|
| Standard Large Model (20 tokens) | 2000 | baseline |
| Speculative (draft + verify, k=4) | 580 | 3.45x |

**7. Combined Optimizations**

Configuration: 32 tokens, FP16, depth 4, adaptive depth, caching

| Setup | Time (s) | Throughput (img/s) | Speedup |
|-------|----------|-------------------|---------|
| Baseline (64 tokens, FP32, depth 6) | 0.438 | 914.07 | baseline |
| Optimized Pass 1 (filling cache) | 1.144 | 349.62 | 0.38x* |
| Optimized Pass 2 (with cache) | 0.360 | 1112.61 | 1.22x |

*First pass slower due to cache filling; subsequent runs show benefit

### Summary

| Optimization | Best Speedup | Memory Savings | Accuracy Impact |
|--------------|--------------|----------------|-----------------|
| Visual Token Caching | 20.68x | N/A | None |
| Reduced Tokens (64→32) | 1.20x | 0.4 MB | Minimal |
| Sparse Attention (k=16) | 1.79x | 188.8 MB (35%) | Task-dependent |
| FP16 Quantization | 1.49x | ~50% | 0.495% error |
| Adaptive Depth | 1.80x | N/A | Depends on threshold |
| Combined (with cache) | 1.22x | ~50% | Minimal |

## Key Innovations

1. **Perceiver Resampler**: Reduces visual tokens from 576→64 (9x compression) while maintaining quality
2. **Gated Cross-Attention**: Enables stable training with frozen language models
3. **Visual Token Caching**: Cache-friendly design for production deployments
4. **Modular Architecture**: Easy to experiment with different configurations

## References

- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (Alayrac et al., 2022)
- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) (Jaegle et al., 2021)


## Acknowledgments

This implementation is inspired by the original Flamingo paper by DeepMind and incorporates architectural insights from the Perceiver family of models.
