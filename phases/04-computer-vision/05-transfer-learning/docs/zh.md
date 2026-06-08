# 迁移学习 & Fine-Tuning

> Somebody else spent a million GPU hours teaching a network what edges, textures, and object parts look like. 你应该 borrow those 特征 before 训练 your own.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (图像 Classification)
**Time:** ~75 minutes

## Learning Objectives

- Distinguish 特征 extraction from fine-tuning and pick the right one based on 数据集 size, domain distance, and compute budget
- Load a pretrained 骨干网络, replace its classifier head, and train only the head to a working baseline in under 20 lines
- Progressively unfreeze 层 with discriminative learning rates so early generic 特征 get smaller updates than late task-specific ones
- Diagnose the three common failures: 特征 drift from too-high LR on unfrozen blocks, BN statistics collapse on tiny 数据集, and catastrophic forgetting

## The Problem

训练 a ResNet-50 on ImageNet costs around 2,000 GPU-hours. Very few teams have that budget for every task they ship. What almost every team actually ships is a pretrained 骨干网络 with a new head trained on a few hundred or few thousand task-specific 图像.

这是 not a shortcut. The first conv block of any ImageNet-trained CNN learns edges and Gabor-like 滤波器. The next few blocks learn textures and simple motifs. The middle blocks learn object parts. The final blocks learn combinations that start to look like the 1,000 ImageNet categories. The first 90% of that hierarchy transfers almost unchanged to medical imaging, industrial inspection, satellite 数据, and every other vision task — because nature has a limited vocabulary of edges and textures. The last 10% is what you actually train.

Getting transfer right has three bugs waiting for you: destroying pretrained 特征 with a too-high 学习率, starving the 模型 of information by freezing too much, and letting BatchNorm's running statistics drift toward a tiny 数据集 that the rest of the network never learnt from. This lesson walks each of them on purpose.

## The Concept

### 特征 extraction vs fine-tuning

Two regimes, picked by how much you trust the pretrained 特征 and how much 数据 you have.

```mermaid
flowchart TB
    subgraph FE["Feature extraction — backbone frozen"]
        FE1["Pretrained backbone<br/>(no gradient)"] --> FE2["New head<br/>(trained)"]
    end
    subgraph FT["Fine-tuning — end-to-end"]
        FT1["Pretrained backbone<br/>(tiny LR)"] --> FT2["New head<br/>(normal LR)"]
    end

    style FE1 fill:#e5e7eb,stroke:#6b7280
    style FE2 fill:#dcfce7,stroke:#16a34a
    style FT1 fill:#fef3c7,stroke:#d97706
    style FT2 fill:#dcfce7,stroke:#16a34a
```

Rules of thumb:

| 数据集 size | Domain distance | Recipe |
|--------------|-----------------|--------|
| < 1k 图像 | close to ImageNet | Freeze 骨干网络, train head only |
| 1k-10k | close | Freeze first 2-3 stages, fine-tune the rest |
| 10k-100k | any | Fine-tune end-to-end with discriminative LR |
| 100k+ | far | Fine-tune everything; consider 训练 from scratch if domain is far enough |

"Close to ImageNet" roughly means natural RGB photos with object-like content. Medical CT scans, overhead satellite imagery, and microscopy are far domains — the 特征 still help, but you will need to let more 层 adapt.

### Why freezing works at all

The ImageNet 特征 a CNN learns are not specialised to the 1,000 categories. They are specialised to the statistics of natural 图像: edges at specific orientations, textures, contrast patterns, shape primitives. Those statistics are stable across almost every visual domain a human can name. 那是 why a 模型 trained on ImageNet and evaluated zero-shot on CIFAR-10 with just a new linear head (no fine-tuning of the 骨干网络) reaches 80%+ accuracy. The head is learning which of the already-learnt 特征 to 权重 for this task.

### Discriminative learning rates

When you do unfreeze, early 层 should train slower than late 层. Early 层 encode generic 特征 that you want to preserve; late 层 encode task-specific structure that you need to move a lot.

```
Typical recipe:

  stage 0 (stem + first group): lr = base_lr / 100    (mostly fixed)
  stage 1:                       lr = base_lr / 10
  stage 2:                       lr = base_lr / 3
  stage 3 (last backbone group): lr = base_lr
  head:                          lr = base_lr  (or slightly higher)
```

In PyTorch this is just a list of 参数 groups passed to the 优化器. One 模型, five learning rates, zero extra 代码.

### The BatchNorm problem

BN 层 hold `running_mean` and `running_var` buffers that were computed on ImageNet. If your task has a different 像素 distribution — different lighting, different sensor, different colour space — those buffers are wrong. Three options in order of preference:

1. **Fine-tune with BN in train mode.** Let BN update its running statistics along with everything else. Default choice when the task 数据集 is medium-sized (>= 5k examples).
2. **Freeze BN in eval mode.** Keep the ImageNet statistics and train only the 权重. Correct when your 数据集 is small enough that BN's moving average would be noisy.
3. **Replace BN with GroupNorm.** Removes the moving-average problem entirely. Used in detection and 分割 backbones where 批次 size per GPU is tiny.

Getting this wrong silently tanks accuracy by 5-15%.

### Head design

The classifier head is 1-3 linear 层 plus an optional dropout. Every torchvision 骨干网络 ships a default head that you replace:

```
backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)          # ResNet
backbone.classifier[1] = nn.Linear(..., num_classes)                    # EfficientNet, MobileNet
backbone.heads.head = nn.Linear(..., num_classes)                       # torchvision ViT
```

For small 数据集, a single linear 层 is usually enough. Adding a hidden 层 (Linear -> ReLU -> Dropout -> Linear) helps when the task distribution is farther from the 骨干网络's 训练 distribution.

### 层-wise LR decay

A smoother version of discriminative LR used in modern fine-tuning (BEiT, DINOv2, ViT-B fine-tunes). Instead of grouping 层 into stages, give every 层 a slightly smaller LR than the one above it:

```
lr_layer_k = base_lr * decay^(L - k)
```

With decay = 0.75 and L = 12 transformer blocks, the first block trains at `0.75^11 ≈ 0.04x` the head's LR. Matters more for transformer fine-tunes than for CNNs, where stage-grouped LRs are usually enough.

### What to evaluate

Transfer-learning runs need two numbers you would not track on a scratch run:

- **Pretrained-only accuracy** — the head's accuracy with the 骨干网络 frozen. 这是 your floor.
- **Fine-tuned accuracy** — the same 模型 after end-to-end 训练. 这是 your ceiling.

If fine-tuned is less than pretrained-only, you have a learning-rate or BN bug. Always print both.

## Build It

### Step 1: Load a pretrained 骨干网络 and inspect it

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print(backbone)
print()
print("classifier head:", backbone.fc)
print("feature dim:", backbone.fc.in_features)
```

`ResNet18` has four stages (`layer1..layer4`) plus a stem and a `fc` head. Every torchvision classification 骨干网络 has an analogous structure.

### Step 2: 特征 extraction — freeze everything, replace the head

```python
def make_feature_extractor(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = make_feature_extractor(num_classes=10)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable: {trainable:>10,}")
print(f"frozen:    {frozen:>10,}")
```

Only `模型.fc` is trainable. The 骨干网络 is a frozen 特征 extractor.

### Step 3: Discriminative fine-tuning

A utility that builds 参数 groups with stage-specific learning rates.

```python
def discriminative_param_groups(model, base_lr=1e-3, decay=0.3):
    stages = [
        ["conv1", "bn1"],
        ["layer1"],
        ["layer2"],
        ["layer3"],
        ["layer4"],
        ["fc"],
    ]
    groups = []
    for i, names in enumerate(stages):
        lr = base_lr * (decay ** (len(stages) - 1 - i))
        params = [p for n, p in model.named_parameters()
                  if any(n.startswith(k) for k in names)]
        if params:
            groups.append({"params": params, "lr": lr, "name": "_".join(names)})
    return groups

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
for p in model.parameters():
    p.requires_grad = True

groups = discriminative_param_groups(model)
for g in groups:
    print(f"{g['name']:>10s}  lr={g['lr']:.2e}  params={sum(p.numel() for p in g['params']):>8,}")
```

`decay=0.3` means each stage trains at 30% of the rate of the next one. `fc` gets `base_lr`, `layer4` gets `0.3 * base_lr`, `conv1` gets `0.3^5 * base_lr ≈ 0.00243 * base_lr`. Extreme sounding; empirically it works.

### Step 4: BatchNorm handling

Helper to freeze BN running statistics without freezing its 权重.

```python
def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    return model
```

Call it after you set `模型.train()` at the start of every 轮次. `模型.train()` flips everything to 训练 mode; this reverses it only for BN 层.

### Step 5: A minimal end-to-end fine-tuning loop

```python
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

def fine_tune(model, train_loader, val_loader, device, epochs=5, base_lr=1e-3, freeze_bn=False):
    model = model.to(device)
    groups = discriminative_param_groups(model, base_lr=base_lr)
    optimizer = SGD(groups, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        if freeze_bn:
            freeze_bn_stats(model)
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_total += x.size(0)
            tr_correct += (logits.argmax(-1) == y).sum().item()
        scheduler.step()

        model.eval()
        va_total, va_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                va_total += x.size(0)
                va_correct += (pred == y).sum().item()
        print(f"epoch {epoch}  train {tr_loss/tr_total:.3f}/{tr_correct/tr_total:.3f}  "
              f"val {va_correct/va_total:.3f}")
    return model
```

Five 轮次 with the above recipe on CIFAR-10 takes `ResNet18-IMAGENET1K_V1` from ~70% zero-shot linear-probe accuracy to ~93% fine-tuned accuracy. The head alone would plateau around 86% without ever touching the 骨干网络.

### Step 6: Progressive unfreezing

A schedule that unfreezes one stage per 轮次 from the end toward the beginning. Mitigates 特征 drift at the cost of some extra 轮次.

```python
def progressive_unfreeze_schedule(model):
    stages = ["layer4", "layer3", "layer2", "layer1"]
    yielded = set()

    def start():
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    def unfreeze(epoch):
        if epoch < len(stages):
            name = stages[epoch]
            yielded.add(name)
            for n, p in model.named_parameters():
                if n.startswith(name):
                    p.requires_grad = True
            return name
        return None

    return start, unfreeze
```

Call `start()` once before the first 轮次. Call `unfreeze(轮次)` at the start of each 轮次. Rebuild the 优化器 whenever the set of trainable 参数 changes, otherwise the frozen params still hold cached moments that confuse it.

## Use It

For most real tasks, `torchvision.模型` + three lines is enough. The heavier machinery above matters when you run into the problems that library defaults cannot fix.

```python
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

Two other production-grade defaults:

- `timm` ships ~800 pretrained vision backbones with a consistent API (`timm.create_model("resnet50", pretrained=True, num_classes=10)`). For any fine-tune beyond the torchvision zoo, it is the standard.
- For transformers, `transformers.AutoModelForImageClassification.from_pretrained(name, num_labels=N)` gives you ViT / BEiT / DeiT with the same loading semantics as text 模型.

## Ship It

This lesson produces:

- `输出/prompt-fine-tune-planner.md` — a prompt that picks 特征-extraction vs progressive vs end-to-end fine-tuning based on 数据集 size, domain distance, and compute budget.
- `输出/skill-freeze-inspector.md` — a skill that, given a PyTorch 模型, reports which 参数 are trainable, which BatchNorm 层 are in eval mode, and whether the 优化器 is actually being fed the trainable 参数.

## Exercises

1. **(Easy)** Train a `ResNet18` as a linear probe (骨干网络 frozen) and as a full fine-tune on the same synthetic-CIFAR 数据集. Report both accuracies side by side. Explain which gap tells you the 特征 transfer well and which tells you they do not.
2. **(Medium)** Introduce a bug on purpose: set `base_lr = 1e-1` on the 骨干网络 stage instead of the head. Show the 训练 loss explode, then recover by applying the `discriminative_param_groups` helper. Record the LR at which each stage starts diverging.
3. **(Hard)** Take a medical imaging 数据集 (e.g. CheXpert-small, PatchCamelyon, or HAM10000) and compare three regimes: (a) ImageNet-pretrained frozen 骨干网络 + linear head; (b) ImageNet-pretrained fine-tune end-to-end; (c) scratch 训练. Report accuracy and compute cost for each. At what 数据集 size does scratch 训练 become competitive?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| 特征 extraction | "Freeze and train head" | 骨干网络 参数 frozen, only the new classifier head receives 梯度 |
| Fine-tuning | "Retrain end-to-end" | All 参数 trainable, usually with much smaller LR than scratch 训练 |
| Discriminative LR | "Smaller LR for early 层" | 优化器 参数 groups where early-stage LR is a fraction of late-stage LR |
| 层-wise LR decay | "Smooth LR 梯度" | Per-层 LR multiplied by decay^(L - k); common in transformer fine-tunes |
| Catastrophic forgetting | "The 模型 lost ImageNet" | A too-high LR overwrites pretrained 特征 before the new task signal is learnt |
| BN statistics drift | "Running mean is wrong" | BatchNorm running_mean/var computed on a different distribution than the current task, silently hurting accuracy |
| Linear probe | "Frozen 骨干网络 + linear head" | Evaluation of pretrained 特征 — accuracy of the best linear classifier on top of the frozen representation |
| Catastrophic collapse | "Everything predicts one class" | Happens when fine-tuning with an LR high enough to destroy 特征 before 梯度 from the head can stabilise |

## Further Reading

- [How transferable are 特征 in deep 神经网络? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792) — the paper that quantified 特征 transferability across 层
- [Universal Language 模型 Fine-tuning (ULMFiT, Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146) — the original discriminative LR / progressive unfreezing recipe; the ideas transfer directly to vision
- [timm documentation](https://huggingface.co/docs/timm) — the reference for modern vision backbones and the exact fine-tune defaults they were trained with
- [A Simple Framework for Linear-Probe Evaluation (Kornblith et al., 2019)](https://arxiv.org/abs/1805.08974) — why linear-probe accuracy matters and how to report it correctly
