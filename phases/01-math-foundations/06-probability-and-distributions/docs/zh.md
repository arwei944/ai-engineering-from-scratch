# 概率 和 Distributions

> 概率 是 language AI uses 到 express uncertainty.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Learning Objectives

- Implement PMFs 和 PDFs 从 scratch 为了 Bernoulli, categorical, Poisson, uniform, 和 normal distributions
- Compute expected value, variance, 和 use Central Limit Theorem 到 explain why Gaussians dominate
- Build softmax 和 log-softmax 函数 使用 numerical stability trick (subtract max logit)
- Calculate cross-entropy loss 从 logits 和 connect it 到 negative log-likelihood

## Problem

classifier 输出 `[0.03, 0.91, 0.06]`. language 模型 picks next word 从 50,000 candidates. diffusion 模型 generates images 通过 sampling 从 learned distributions. All 的 这些 是 概率 在 action.

Every prediction 模型 makes 是 概率 distribution. Every 损失函数 measures how far predicted distribution 是 从 true one. Every 训练 step adjusts 参数 到 make one distribution look more like another. Without 概率, you cannot read single ML paper, debug single 模型, 或 understand why your 训练 loss 是 NaN.

## Concept

### Events, Sample Spaces, 和 概率

sample space S 是 set 的 all possible outcomes. event 是 subset 的 sample space. 概率 maps events 到 numbers between 0 和 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Three axioms define all 的 概率:
1. P() >= 0 为了 any event
2. P(S) = 1 (something always happens)
3. P( 或 B) = P() + P(B) when 和 B cannot both occur

Everything else (Bayes' theorem, expectations, distributions) follows 从 这些 three rules.

### Conditional 概率 和 Independence

P(|B) 是 概率 的 given B happened.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Two events 是 independent when knowing one tells you nothing about other:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Coin flips 是 independent. Drawing cards without replacement 是 not.

### 概率 Mass Functions vs 概率 Density Functions

Discrete random variables have 概率 mass 函数 (PMF). Each outcome has specific 概率 you can read off directly.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Continuous random variables have 概率 density 函数 (PDF). density 在 single point 是 not 概率. 概率 comes 从 integrating density over interval.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

This distinction matters 在 ML. 分类 输出 是 PMFs (discrete choices). VAE latent spaces use PDFs (continuous).

### Common Distributions

**Bernoulli:** one trial, two outcomes. Models binary 分类.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:** one trial, k outcomes. Models multi-class 分类 (softmax 输出).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** all outcomes equally likely. Used 为了 random initialization.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** bell curve. Parameterized 通过 mean (mu) 和 variance (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** counts 的 rare events 在 fixed interval. Models event rates.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Expected Value 和 Variance

Expected value 是 weighted average outcome.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Variance measures spread around mean.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

In ML, expected value appears 作为 损失函数 (average loss over 数据 distribution). Variance tells you about 模型 stability. High variance 在 gradients means noisy 训练.

### Joint 和 Marginal Distributions

joint distribution P(X, Y) describes two random variables together.

Joint PMF example (X = weather, Y = umbrella):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

marginal distribution sums out other variable:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

row 和 column totals 在 table above 是 marginals.

### Why Normal Distribution Shows Up Everywhere

Central Limit Theorem: sum (或 average) 的 many independent random variables converges 到 normal distribution, regardless 的 original distribution.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

这是 why:
- Measurement errors 是 approximately normal (many small independent sources)
- 权重 initializations 在 神经网络 use normal distributions
- Gradient noise 在 SGD 是 approximately normal (sum 的 many sample gradients)
- normal distribution 是 maximum entropy distribution 为了 given mean 和 variance

### Log Probabilities

Raw probabilities cause numerical problems. Multiplying many small probabilities together quickly underflows 到 zero.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log probabilities fix 这个. Multiplications become additions.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Rules:
- log( * b) = log() + log(b)
- log probabilities 是 always <= 0 (since 0 < P <= 1)
- More negative = less likely
- Cross-entropy loss 是 negative log 概率 的 correct class

### Softmax 作为 概率 Distribution

Neural networks 输出 raw scores (logits). Softmax converts them into valid 概率 distribution.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

softmax trick: subtract max logit before exponentiating 到 prevent overflow.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax combines softmax 和 log 为了 numerical stability. PyTorch uses 这个 internally 为了 cross-entropy loss.

### Sampling

Sampling means drawing random values 从 distribution. In ML:
- Dropout randomly samples which 神经元 到 zero out
- 数据 augmentation samples random transformations
- Language 模型 sample next token 从 predicted distribution
- Diffusion 模型 sample noise 和 progressively denoise

Sampling 从 arbitrary distributions requires techniques like inverse transform sampling, rejection sampling, 或 reparameterization trick (used 在 VAEs).

## Build It

### Step 1: 概率 basics

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Step 2: PMF 和 PDF 从 scratch

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Step 3: Expected value 和 variance

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Step 4: Sampling 从 distributions

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Step 5: Softmax 和 log probabilities

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Step 6: Central Limit Theorem demonstration

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Step 7: Visualization

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Full implementations 使用 all visualizations 是 在 `代码/概率.py`.

## Use It

With NumPy 和 SciPy, everything above 是 one-liners:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

You built 这些 从 scratch. Now you know what library calls 是 doing.

## Exercises

1. Implement inverse transform sampling 为了 exponential distribution. Verify 通过 sampling 10,000 values 和 comparing histogram 到 true PDF.

2. Build joint distribution table 为了 two loaded dice. Compute marginal distributions 和 check whether dice 是 independent.

3. Compute cross-entropy loss 为了 5-class classifier 输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]` when correct class 是 index 3. Then verify your answer 使用 PyTorch's `nn.CrossEntropyLoss`.

4. Write 函数 takes list 的 log probabilities 和 returns most likely sequence, total log 概率, 和 equivalent raw 概率. Test it 使用 sentence 的 50 words where each word has 概率 0.01.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sample space | "All possibilities" | set S 的 every possible outcome 的 experiment |
| PMF | " 概率 函数" | 函数 gives exact 概率 的 each discrete outcome, summing 到 1 |
| PDF | " 概率 curve" | density 函数 为了 continuous variables. Integrate it over interval 到 get 概率 |
| Conditional 概率 | "概率 given something" | P(\|B) = P( 和 B) / P(B). foundation 的 Bayesian thinking 和 Bayes' theorem |
| Independence | "They don't affect each other" | P( 和 B) = P() * P(B). Knowing one event tells you nothing about other |
| Expected value | " average" | 概率-weighted sum 的 all outcomes. 损失函数 是 expected value |
| Variance | "How spread out" | expected squared deviation 从 mean. High variance = noisy, unstable estimates |
| Normal distribution | " bell curve" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Appears everywhere due 到 CLT |
| Central Limit Theorem | "Averages become normal" | mean 的 many independent samples converges 到 normal distribution regardless 的 source |
| Joint distribution | "Two variables together" | P(X, Y) describes 概率 的 every combination 的 X 和 Y outcomes |
| Marginal distribution | "Sum out other variable" | P(X) = sum_y P(X, Y). Recovers one variable's distribution 从 joint |
| Log 概率 | "Log 的 概率" | log P(x). Turns products into sums, preventing numerical underflow 在 long sequences |
| Softmax | "Turn scores into probabilities" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Maps real-valued logits 到 valid 概率 distribution |
| Cross-entropy | " 损失函数" | -sum(p_true * log(p_predicted)). Measures how different two distributions 是. Lower 是 better |
| Logits | "Raw 模型 输出" | Unnormalized scores before softmax. Named after logistic 函数 |
| Sampling | "Drawing random values" | Generating values according 到 概率 distribution. How 模型 generate 输出 |

## Further Reading

- [3Blue1Brown: But what 是 Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - visual proof 的 why averages become normal
- [Stanford CS229 概率 Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - concise reference covering everything here 和 more
- [ Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - why numerical stability matters 和 how 到 achieve it
