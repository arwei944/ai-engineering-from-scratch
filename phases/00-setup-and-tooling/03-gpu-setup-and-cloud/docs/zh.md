# GPU Setup & Cloud

> 训练 在 CPU 是 fine 为了 learning. 训练 为了 real needs GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Learning Objectives

- Verify local GPU availability using `nvidia-smi` 和 PyTorch's CUDA API
- Configure Google Colab 使用 T4 GPU 为了 free cloud-based experiments
- Benchmark 矩阵 multiplication 在 CPU vs GPU 和 measure speedup
- Estimate largest 模型 fits 在 your VRAM using fp16 rule 的 thumb

## Problem

Most lessons 在 phases 1-3 run fine 在 CPU. But once you start 训练 CNNs, transformers, 或 LLMs (phases 4+), you need GPU acceleration. 训练 run takes 8 hours 在 CPU takes 10 minutes 在 GPU.

You have three options: local GPU, cloud GPU, 或 Google Colab (free).

## Concept

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## Build It

### Option 1: Local NVIDIA GPU

Check if you have one:

```bash
nvidia-smi
```

Install PyTorch 使用 CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Option 2: Google Colab

1. Go 到 [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Run `!nvidia-smi` 到 verify

Upload notebooks 从 这个 course directly 到 Colab.

### Option 3: Cloud GPU

For Lambda Labs, RunPod, 或 Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### No GPU? No problem.

Most lessons work 在 CPU. ones need GPU will say so 和 include Colab links.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build It: GPU vs CPU benchmark

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Exercises

1. Run benchmark above 和 compare CPU vs GPU times
2. If you don't have GPU, run it 在 Google Colab 和 compare
3. Check how much GPU memory you have 和 estimate largest 模型 you can fit (rule 的 thumb: 2 bytes per 参数 为了 fp16)

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform lets you run 代码 在 GPU |
| VRAM | "GPU memory" | Video RAM 在 GPU, separate 从 system RAM. Limits 模型 size. |
| fp16 | "Half 精确率" | 16-bit floating point, uses half memory 的 fp32 使用 minimal 准确率 loss |
| 张量 Core | "Fast 矩阵 hardware" | Specialized GPU cores 为了 矩阵 multiplication, 4-8x faster than regular cores |
