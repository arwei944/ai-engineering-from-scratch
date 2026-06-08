# Numerical Stability

> Floating point 是 leaky abstraction. It will bite you during 训练, 和 you will not see it coming.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- Implement numerically stable softmax 和 log-sum-exp using max-subtraction trick
- Identify overflow, underflow, 和 catastrophic cancellation 在 floating-point computations
- Verify analytical gradients against numerical gradients using centered finite differences
- Explain why bfloat16 是 preferred over float16 为了 训练 和 how loss scaling prevents gradient underflow

## Problem

Your 模型 trains 为了 three hours, then loss becomes NaN. You add print statement. logits 是 fine 在 step 9,000. At step 9,001 they 是 `inf`. By step 9,002 every gradient 是 `nan` 和 训练 是 dead.

Or: your 模型 trains 到 completion but 准确率 是 2% worse than paper claims. You check everything. Architecture matches. Hyperparameters match. 数据 matches. problem 是 paper used float32 和 you used float16 without right scaling. Thirty-two bits 的 accumulated rounding error quietly ate your 准确率.

Or: you implement cross-entropy loss 从 scratch. It works 在 small logits. When logits exceed 100, it returns `inf`. softmax overflowed because `exp(100)` 是 larger than float32 can represent. Every ML framework handles 这个 使用 two-line trick. You did not know trick existed.

Numerical stability 是 not theoretical concern. 它是 difference between 训练 run succeeds 和 one silently fails. Every serious ML bug you will debug eventually comes down 到 floating point.

## Concept

### IEEE 754: How Computers Store Real Numbers

Computers store real numbers 作为 floating point values following IEEE 754 standard. float has three parts: sign bit, exponent, 和 mantissa (significand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

mantissa determines 精确率 (how many significant digits). exponent determines range (how large 或 small number can be).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 gives you about 7 decimal digits 的 精确率. That means it can tell apart 1.0000001 和 1.0000002, but not 1.00000001 和 1.00000002. After 7 digits, everything 是 rounding noise.

float16 gives you about 3 digits. largest number it can represent 是 65,504. 那是 disturbingly small 为了 ML where logits, gradients, 和 activations routinely exceed 这个.

bfloat16 是 Google's answer 到 float16's range problem. It has same 8-bit exponent 作为 float32 (same range, up 到 3.4e38) but only 7 mantissa bits (less 精确率 than float16). For 训练 神经网络, range matters more than 精确率, so bfloat16 usually wins.

### Why 0.1 + 0.2 != 0.3

number 0.1 cannot be represented exactly 在 binary floating point. In base 2, it 是 repeating fraction:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 truncates 这个 到 23 bits 的 mantissa. stored value 是 approximately 0.100000001490116. Similarly, 0.2 是 stored 作为 approximately 0.200000002980232. Their sum 是 0.300000004470348, not 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

This matters 为了 ML because:

1. Loss comparisons like `if loss < threshold` can give wrong answers
2. Accumulating many small values (gradient updates over thousands 的 steps) drifts 从 true sum
3. Checksums 和 reproducibility tests fail if you compare floats 使用 `==`

fix: never compare floats 使用 `==`. Use `abs( - b) < epsilon` 或 `math.isclose()`.

### Catastrophic Cancellation

When you subtract two nearly equal floating point numbers, significant digits cancel 和 you 是 left 使用 rounding noise promoted 到 leading digits.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

那是 19% relative error 从 single subtraction. In ML, 这个 happens whenever you:

- Compute variance 的 数据 使用 large mean: `E[x^2] - E[x]^2` when E[x] 是 large
- Subtract nearly equal log-probabilities
- Compute finite-difference gradients 使用 too-small epsilon

fix: rearrange formulas 到 avoid subtracting large, nearly equal numbers. For variance, use Welford 算法 或 center 数据 first. For log-probabilities, work 在 log-space throughout.

### Overflow 和 Underflow

Overflow happens when result 是 too large 到 represent. Underflow happens when it 是 too small (closer 到 zero than smallest representable positive number).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函数 是 primary source 的 overflow 在 ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数 hits other direction:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

In ML, `exp()` appears 在 softmax, sigmoid, 和 概率 computations. `log()` appears 在 cross-entropy, log-likelihoods, 和 KL divergence. combination `log(exp(x))` 是 minefield without right tricks.

### Log-Sum-Exp Trick

Computing `log(sum(exp(x_i)))` directly 是 numerically dangerous. If any `x_i` 是 large, `exp(x_i)` overflows. If all `x_i` 是 very negative, every `exp(x_i)` underflows 到 zero 和 `log(0)` 是 `-inf`.

trick: subtract maximum value before exponentiating.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Why 这个 works: after subtracting `max(x)`, largest exponent 是 `exp(0) = 1`. No overflow 是 possible. At least one term 在 sum 是 1, so sum 是 在 least 1, 和 `log(1) = 0`. No underflow 到 `-inf` 是 possible.

Proof:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Set `c = max(x)` 和 overflow 是 eliminated.

This trick appears everywhere 在 ML:
- Softmax normalization
- Cross-entropy loss computation
- Log-概率 summation 在 sequence 模型
- Mixture 的 Gaussians
- Variational inference

### Why Softmax Needs Max-Subtraction Trick

Softmax converts logits 到 probabilities:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Without trick, logits 的 [100, 101, 102] cause overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

With trick, subtract max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

probabilities 是 identical. computation 是 safe. 这是 not optimization. 它是 requirement 为了 correctness.

### NaN 和 Inf: Detection 和 Prevention

`nan` (Not Number) 和 `inf` (infinity) propagate virally through computation. One `nan` 在 gradient update makes 权重 `nan`, which makes every subsequent 输出 `nan`. 训练 是 dead within one step.

How `inf` appears:
- `exp()` 的 large positive number
- Division 通过 zero: `1.0 / 0.0`
- `float32` overflow 在 accumulations

How `nan` appears:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` 的 negative number
- `log()` 的 negative number
- Any arithmetic involving existing `nan`

Detection:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Prevention strategies:

1. Clamp 输入 到 `exp()`: `exp(clamp(x, -80, 80))`
2. Add epsilon 到 denominators: `x / (y + 1e-8)`
3. Add epsilon inside `log()`: `log(x + 1e-8)`
4. Use stable implementations (log-sum-exp, stable softmax)
5. Gradient clipping 到 prevent 权重 explosion
6. Check 为了 `nan`/`inf` after every forward pass during debugging

### Numerical Gradient Checking

Analytical gradients (从 反向传播) can have bugs. Numerical gradient checking verifies them 通过 computing gradients 使用 finite differences.

centered difference formula:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) accurate, much better than forward difference `(f(x+h) - f(x)) / h` which 是 only O(h).

Choosing h: too large 和 approximation 是 wrong. Too small 和 catastrophic cancellation destroys answer. `h = 1e-5` 到 `1e-7` 是 typical.

check: compute relative difference between analytical 和 numerical gradients.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Rules 的 thumb:
- relative_error < 1e-7: perfect, gradient 是 correct
- relative_error < 1e-5: acceptable, probably correct
- relative_error > 1e-3: something 是 wrong
- relative_error > 1: gradient 是 completely wrong

Always check gradients when implementing new 层 或 损失函数. PyTorch provides `torch.autograd.gradcheck()` 为了 这个.

### Mixed 精确率 训练

Modern GPUs have specialized hardware (张量 Cores) compute float16 矩阵 multiplications 2-8x faster than float32. Mixed 精确率 训练 exploits 这个:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

problem 使用 pure float16 训练: gradients 是 often very small (1e-8 或 smaller). Float16 underflows anything below ~6e-8 到 zero. Your 模型 stops learning because all gradient updates 是 zero.

fix 是 loss scaling:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling adjusts scale factor automatically. Start 使用 large value (65536). If gradients overflow 到 `inf`, halve it. If N steps pass without overflow, double it.

### bfloat16 vs float16: Why bfloat16 Wins 为了 训练

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 has more 精确率 (10 mantissa bits vs 7) but limited range (max ~65,504). bfloat16 has less 精确率 but same range 作为 float32 (max ~3.4e38).

For 训练 神经网络:

- Activations 和 logits regularly exceed 65,504 during 训练 spikes. float16 overflows; bfloat16 handles it.
- Loss scaling 是 required 使用 float16 but usually unnecessary 使用 bfloat16 because its range covers gradient magnitude spectrum.
- bfloat16 是 simple truncation 的 float32: drop bottom 16 bits 的 mantissa. Conversion 是 trivial 和 lossless 在 exponent.

float16 是 preferred 为了 inference where values 是 bounded 和 精确率 matters more. bfloat16 是 preferred 为了 训练 where range matters more. 这是 why TPUs 和 modern NVIDIA GPUs (A100, H100) have native bfloat16 support.

### Gradient Clipping

Exploding gradients happen when gradients grow exponentially through many 层 (common 在 RNNs, deep networks, 和 transformers). single large gradient can corrupt all 权重 在 one step.

Two types 的 clipping:

**Clip 通过 value:** clamp each gradient element independently.

```
grad = clamp(grad, -max_val, max_val)
```

Simple but can change direction 的 gradient 向量.

**Clip 通过 norm:** scale entire gradient 向量 so its norm does not exceed threshold.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Preserves direction 的 gradient. 这是 what `torch.nn.utils.clip_grad_norm_()` does. 它是 standard choice.

Typical values: `max_norm=1.0` 为了 transformers, `max_norm=0.5` 为了 RL, `max_norm=5.0` 为了 simpler networks.

Gradient clipping 是 not hack. 它是 safety mechanism. Without it, single outlier 批次 can produce gradient large enough 到 ruin weeks 的 训练.

### Normalization Layers 作为 Numerical Stabilizers

批次 normalization, 层 normalization, 和 RMS normalization 是 usually presented 作为 regularizers help 训练 converge. They 是 also numerical stabilizers.

Without normalization, activations can grow 或 shrink exponentially through 层:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalization recenters 和 rescales activations 在 every 层:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (typically 1e-5) prevents division 通过 zero when all activations 是 identical. learned 参数 `gamma` 和 `beta` let network restore any scale it needs.

This keeps values 在 numerically safe range throughout network, preventing both overflow 在 forward pass 和 gradient explosion 在 backward pass.

### Common ML Numerical Bugs

**Bug: Loss 是 NaN after few 轮次.**
Cause: logits grew too large, softmax overflowed. Or 学习率 是 too high 和 权重 diverged.
Fix: use stable softmax (max subtraction), reduce 学习率, add gradient clipping.

**Bug: Loss 是 stuck 在 log(num_classes).**
Cause: 模型 输出 是 near-uniform probabilities. Often means gradients 是 vanishing 或 模型 是 not learning 在 all.
Fix: check 数据 labels 是 correct, verify 损失函数, check 为了 dead ReLUs.

**Bug: 验证 准确率 是 lower than expected 通过 1-3%.**
Cause: mixed 精确率 without proper loss scaling. Gradient underflow silently zeroes out small updates.
Fix: enable dynamic loss scaling, 或 switch 到 bfloat16.

**Bug: Gradient norms 是 0.0 为了 some 层.**
Cause: dead ReLU 神经元 (all 输入 negative), 或 float16 underflow.
Fix: use LeakyReLU 或 GELU, use gradient scaling, check 权重 initialization.

**Bug: 模型 works 在 one GPU but gives different results 在 another.**
Cause: non-deterministic floating point accumulation order. GPU parallel reductions sum 在 different orders 在 different hardware, 和 floating point addition 是 not associative.
Fix: accept small differences (1e-6), 或 set `torch.use_deterministic_algorithms(True)` 和 accept speed penalty.

**Bug: `exp()` returns `inf` 在 loss computation.**
Cause: raw logits passed 到 `exp()` without max-subtraction trick.
Fix: use `torch.nn.functional.log_softmax()` which implements log-sum-exp internally.

**Bug: 训练 diverges after switching 从 float32 到 float16.**
Cause: float16 cannot represent gradient magnitudes below 6e-8 或 activations above 65,504.
Fix: use mixed 精确率 使用 loss scaling (AMP), 或 use bfloat16 instead.

## Build It

### Step 1: Demonstrate floating point 精确率 limits

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: Implement naive vs stable softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Step 3: Implement stable log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Step 4: Implement stable cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5: Gradient checking

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It

### Mixed 精确率 simulation

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf detection

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

See `代码/numerical.py` 为了 complete implementations 使用 all edge cases demonstrated.

## Ship It

This lesson produces:
- `代码/numerical.py` 使用 stable softmax, log-sum-exp, cross-entropy, gradient checking, 和 mixed 精确率 simulation
- `输出/prompt-numerical-debugger.md` 为了 diagnosing NaN/Inf 和 numerical issues 在 训练

These stable implementations reappear 在 Phase 3 when building 训练 loop 和 在 Phase 4 when implementing attention mechanisms.

## Exercises

1. **Catastrophic cancellation.** Compute variance 的 [1000000.0, 1000001.0, 1000002.0] using naive formula `E[x^2] - E[x]^2` 在 float32. Then compute it using Welford's online 算法. Compare errors against true variance (0.6667).

2. **精确率 hunt.** Find smallest positive float32 value `x` such `1.0 + x == 1.0` 在 Python. 这是 machine epsilon. Verify it matches `numpy.finfo(numpy.float32).eps`.

3. **Log-sum-exp edge cases.** Test your `logsumexp_stable` 函数 使用: () all values equal, (b) one value much larger than rest, (c) all values very negative (-1000). Verify it gives correct results where naive version fails.

4. **Gradient checking 神经网络 层.** Implement single linear 层 `y = Wx + b` 和 its analytical backward pass. Use `numerical_gradient` 到 verify correctness 为了 3x2 权重 矩阵.

5. **Loss scaling experiment.** Simulate 训练 使用 float16: create random gradients 在 range [1e-9, 1e-3], convert 到 float16, 和 measure what fraction become zero. Then apply loss scaling (multiply 通过 1024), convert 到 float16, scale back, 和 measure zero fraction again.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| IEEE 754 | " float standard" | International standard defining binary floating point formats, rounding rules, 和 special values (inf, nan). Every modern CPU 和 GPU implements it. |
| Machine epsilon | " 精确率 limit" | smallest value e such 1.0 + e != 1.0 在 given float format. For float32, it 是 about 1.19e-7. |
| Catastrophic cancellation | "精确率 loss 从 subtraction" | When subtracting nearly equal floating point numbers, significant digits cancel 和 rounding noise dominates result. |
| Overflow | "Number too big" | result exceeds maximum representable value 和 becomes inf. exp(89) overflows float32. |
| Underflow | "Number too small" | result 是 closer 到 zero than smallest representable positive number 和 becomes 0.0. exp(-104) underflows float32. |
| Log-sum-exp trick | "Subtract max first" | Computing log(sum(exp(x))) 通过 factoring out exp(max(x)) 到 prevent overflow 和 underflow. Used 在 softmax, cross-entropy, 和 log-概率 math. |
| Stable softmax | "Softmax does not explode" | Subtracting max(logits) before exponentiating. Numerically identical result, no overflow possible. |
| Gradient checking | "Verify your backprop" | Comparing analytical gradients 从 反向传播 against numerical gradients 从 finite differences 到 catch implementation bugs. |
| Mixed 精确率 | "Float16 forward, float32 backward" | Using lower-精确率 floats 为了 speed-critical operations 和 higher-精确率 floats 为了 numerically sensitive operations. Typical speedup 是 2-3x. |
| Loss scaling | "Prevent gradient underflow" | Multiplying loss 通过 large constant before backprop so gradients stay 在 float16's representable range, then dividing 通过 same constant before 权重 updates. |
| bfloat16 | "Brain floating point" | Google's 16-bit format 使用 8 exponent bits (same range 作为 float32) 和 7 mantissa bits (less 精确率 than float16). Preferred 为了 训练. |
| Gradient clipping | "Cap gradient norm" | Scaling gradient 向量 so its norm does not exceed threshold. Prevents exploding gradients 从 ruining 权重. |
| NaN | "Not Number" | Special float value 从 undefined operations (0/0, inf-inf, sqrt(-1)). Propagates through all subsequent arithmetic. |
| Inf | "Infinity" | Special float value 从 overflow 或 division 通过 zero. Can combine 到 produce NaN (inf - inf, inf * 0). |
| Numerical gradient | "Brute force derivative" | Approximating derivative 通过 evaluating f(x+h) 和 f(x-h) 和 dividing 通过 2h. Slow but reliable 为了 verification. |

## Further Reading

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- definitive reference, dense but complete
- [Mixed 精确率 训练 (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA paper introduced loss scaling 为了 float16 训练
- [AMP: Automatic Mixed 精确率 (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- practical guide 到 mixed 精确率 在 PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- why Google chose 这个 format 为了 TPUs
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 算法 为了 reducing rounding error 在 floating point sums
