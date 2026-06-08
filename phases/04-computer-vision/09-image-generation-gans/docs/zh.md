# 图像 Generation — GAN

> A GAN is two 神经网络 in a fixed game. One draws, one critiques. They get better together until the drawings fool the critic.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 3 Lesson 06 (Optimizers), Phase 3 Lesson 07 (正则化)
**Time:** ~75 minutes

## Learning Objectives

- Explain the minimax game between generator and discriminator and why the equilibrium corresponds to p_model = p_data
- Implement a DCGAN in PyTorch and get it to generate coherent 32x32 synthetic 图像 in under 60 lines
- Stabilise GAN 训练 with the three standard tricks: non-saturating loss, spectral norm, TTUR (two-timescale update rule)
- Read 训练 curves that distinguish healthy convergence from mode collapse, oscillation, and discriminator-wins-completely

## The Problem

Classification teaches a network to map 图像 to labels. Generation inverts the problem: sample new 图像 that look like they came from the same distribution. 有 no "correct" 输出 you can diff against; there is only a distribution you want to mimic.

The standard loss 函数 (MSE, cross-entropy) cannot measure "did this sample come from the real distribution." Minimising per-像素 error produces blurry averages, not realistic samples. The breakthrough was to learn the loss: train a second network whose job is to tell real from fake, and use its judgement to push the generator.

GAN (Goodfellow et al., 2014) defined that framework. By 2018 StyleGAN was producing 1024x1024 faces indistinguishable from photographs. 扩散模型 模型 have since taken the throne on quality and controllability, but every trick that makes 扩散模型 practical — normalisation choices, latent spaces, 特征 losses — was first understood on GAN.

## The Concept

### The two networks

```mermaid
flowchart LR
    Z["z ~ N(0, I)<br/>noise"] --> G["Generator<br/>transposed convs"]
    G --> FAKE["Fake image"]
    REAL["Real image"] --> D["Discriminator<br/>conv classifier"]
    FAKE --> D
    D --> OUT["P(real)"]

    style G fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

The **generator** G takes a 向量 of noise `z` and 输出 an 图像. The **discriminator** D takes an 图像 and 输出 a single scalar: the probability that the 图像 is real.

### The game

G wants D to be wrong. D wants to be right. Formally:

```
min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Read right to left: D is maximising accuracy on real (`log D(real)`) and fake (`log (1 - D(fake))`) 图像. G is minimising D's accuracy on fakes — it wants `D(G(z))` to be high.

Goodfellow proved that this minimax has a global equilibrium where `p_G = p_data`, D 输出 0.5 everywhere, and the Jensen-Shannon divergence between generated and real distributions is zero. The hard part is getting there.

### Non-saturating loss

The form above is numerically unstable. Early in 训练, `D(G(z))` is near zero for every fake, so `log(1 - D(G(z)))` has vanishing 梯度 with respect to G. The fix: flip G's loss.

```
L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
L_G = -E_z[log D(G(z))]                          # non-saturating
```

Now when `D(G(z))` is near zero, G's loss is large and its 梯度 is informative. Every modern GAN trains with this variant.

### DCGAN architecture rules

Radford, Metz, Chintala (2015) distilled years of failed experiments into five rules that make GAN 训练 stable:

1. Replace 池化 with strided convs (both nets).
2. Use 批次 norm in both generator and discriminator, except 输出 of G and 输入 of D.
3. Remove fully connected 层 on deeper architectures.
4. G uses ReLU on all 层 except 输出 (tanh for 输出 in [-1, 1]).
5. D uses LeakyReLU (negative_slope=0.2) on all 层.

Every modern conv-based GAN (StyleGAN, BigGAN, GigaGAN) still starts from these rules and replaces pieces one at a time.

### Failure modes and their signatures

```mermaid
flowchart LR
    M1["Mode collapse<br/>G produces a narrow<br/>set of outputs"] --> S1["D loss low,<br/>G loss oscillating,<br/>sample variety drops"]
    M2["Vanishing gradients<br/>D wins completely"] --> S2["D accuracy ~100%,<br/>G loss huge and static"]
    M3["Oscillation<br/>G and D keep trading<br/>wins forever"] --> S3["Both losses swing<br/>wildly with no downward trend"]

    style M1 fill:#fecaca,stroke:#dc2626
    style M2 fill:#fecaca,stroke:#dc2626
    style M3 fill:#fecaca,stroke:#dc2626
```

- **Mode collapse**: G finds one 图像 that fools D and produces only that. Fix: add minibatch discrimination, spectral norm, or label-conditioning.
- **Discriminator wins**: D becomes too strong too fast, G's 梯度 vanish. Fix: smaller D, lower D 学习率, or apply label smoothing on the real labels.
- **Oscillation**: the two nets trade wins without ever approaching equilibrium. Fix: TTUR (D learns faster than G by a factor of 2-4), or switch to Wasserstein loss.

### Evaluation

GAN have no ground truth, so how do you know they are working?

- **Sample inspection** — just look at 64 samples at the end of every 轮次. Non-negotiable.
- **FID (Fréchet Inception Distance)** — distance between Inception-v3 特征 distributions of real and generated sets. Lower is better. Community standard.
- **Inception Score** — older, more brittle; prefer FID.
- **精确率/召回率 for generative 模型** — measures quality (精确率) and coverage (召回率) separately. More informative than FID alone.

For a small synthetic-数据 run, sample inspection is enough.

## Build It

### Step 1: Generator

A small DCGAN generator that takes 64-dim noise and produces a 32x32 图像.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

Four transposed convs, each with `kernel_size=4, 步长=2, 填充=1` so they cleanly double spatial size. 输出 activations in [-1, 1] via tanh.

### Step 2: Discriminator

Mirror of the generator. LeakyReLU, strided convs, ends with a scalar logit.

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 2, feat * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x).view(-1)
```

The last conv reduces a `4x4` 特征图 to `1x1`. 输出 is a single scalar per 图像; apply sigmoid only during loss computation.

### Step 3: 训练 step

Alternate: update D once, then G once, every 批次.

```python
import torch.nn.functional as F

def train_step(G, D, real, z, opt_g, opt_d, device):
    real = real.to(device)
    bs = real.size(0)

    # D step
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
    loss_d.backward()
    opt_d.step()

    # G step
    opt_g.zero_grad()
    d_fake = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()
```

`G(z).detach()` in the D step is critical: we do not want 梯度 flowing into G during its update. Forgetting that is the classic beginner bug.

### Step 4: Full 训练 loop on synthetic shapes

```python
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def synthetic_images(num=2000, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((num, 3, size, size), dtype=np.float32) - 1.0
    for i in range(num):
        r = rng.uniform(6, 12)
        cx, cy = rng.uniform(r, size - r, size=2)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.5, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = synthetic_images()
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

G = Generator(z_dim=64, img_channels=3, feat=32).to(device)
D = Discriminator(img_channels=3, feat=32).to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for (batch,) in loader:
        z = torch.randn(batch.size(0), 64, device=device)
        ld, lg = train_step(G, D, batch, z, opt_g, opt_d, device)
    print(f"epoch {epoch}  D {ld:.3f}  G {lg:.3f}")
```

`Adam(lr=2e-4, betas=(0.5, 0.999))` is the DCGAN default — the low beta1 keeps the momentum term from stabilising the adversarial game too much.

### Step 5: Sampling

```python
@torch.no_grad()
def sample(G, n=16, z_dim=64, device="cpu"):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1)
```

Always switch to eval mode before sampling. For DCGAN this matters because 批次 norm running stats are used instead of the 批次's stats.

### Step 6: Spectral normalisation

A drop-in replacement for BN in the discriminator that guarantees the network is 1-Lipschitz. Fixes most "D wins too hard" failures.

```python
from torch.nn.utils import spectral_norm

def build_sn_discriminator(img_channels=3, feat=64):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(img_channels, feat, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat, feat * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 4, 1, 4, 1, 0)),
    )
```

Swap `Discriminator` for `build_sn_discriminator()` and you often do not need the TTUR trick. Spectral norm is the easiest single robustness upgrade you can apply.

## Use It

For serious generation, use pretrained 权重 or switch to 扩散模型. Two standard libraries:

- `torch_fidelity` computes FID / IS on your generator without writing custom eval 代码.
- `pytorch-gan-zoo` (legacy) and `StudioGAN` ship tested implementations of DCGAN, WGAN-GP, SN-GAN, StyleGAN, and BigGAN.

In 2026, GAN are still the best choice for: real-time 图像 generation (latency <10 ms), style transfer, 图像-to-图像 translation with precise control (Pix2Pix, CycleGAN). 扩散模型 wins on photorealism and text conditioning.

## Ship It

This lesson produces:

- `输出/prompt-gan-训练-triage.md` — a prompt that reads a 训练 curve description and picks the failure mode (mode collapse, D-wins, oscillation) plus the single recommended fix.
- `输出/skill-dcgan-scaffold.md` — a skill that writes a DCGAN scaffold from `z_dim`, target `image_size`, and `num_channels`, including 训练 loop and sample saver.

## Exercises

1. **(Easy)** Train the DCGAN above on the synthetic circle 数据集 and save a grid of 16 samples at the end of each 轮次. By which 轮次 do the generated circles become clearly circular?
2. **(Medium)** Replace the discriminator's 批次 norm with spectral norm. Train both versions side by side. Which one converges faster? Which one has lower variance across three seeds?
3. **(Hard)** Implement a conditional DCGAN: feed the class label into both G and D (concat one-hot to the noise in G, concat a class embedding 通道 in D). Train on the synthetic "circles vs squares" 数据集 from lesson 7 and show that class conditioning works by sampling with specific labels.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Generator (G) | "The draws-stuff net" | Maps noise to 图像; trained to fool the discriminator |
| Discriminator (D) | "The critic" | Binary classifier; trained to distinguish real from generated 图像 |
| Minimax | "The game" | min over G, max over D of an adversarial loss; equilibrium is p_G = p_data |
| Non-saturating loss | "The numerically sane version" | G's loss is -log(D(G(z))) instead of log(1 - D(G(z))) to avoid vanishing 梯度 early in 训练 |
| Mode collapse | "Generator makes one thing" | G produces only a small subset of the 数据 distribution; fix with SN, minibatch discrimination, or larger 批次 |
| TTUR | "Two learning rates" | D learns faster than G, typically by a factor of 2-4; stabilises 训练 |
| Spectral norm | "1-Lipschitz 层" | A 权重-normalisation that bounds each 层's Lipschitz constant; stops D from becoming arbitrarily steep |
| FID | "Fréchet Inception Distance" | Distance between Inception-v3 特征 distributions of real and generated sets; the standard evaluation metric |

## Further Reading

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661) — the paper that started it all
- [DCGAN (Radford, Metz, Chintala, 2015)](https://arxiv.org/abs/1511.06434) — the architecture rules that made GAN trainable
- [Spectral Normalization for GAN (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957) — the single most useful stabilisation trick
- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423) — the SOTA GAN; reads like a greatest-hits album of every trick from the last decade
