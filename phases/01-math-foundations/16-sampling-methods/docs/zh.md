# Sampling Methods

> Sampling 是 how AI explores space 的 possibilities.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (概率, Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- Implement inverse CDF, rejection, 和 importance sampling 从 scratch using only uniform random numbers
- Build temperature, top-k, 和 top-p (nucleus) sampling 为了 language 模型 token generation
- Explain reparameterization trick 和 why it enables 反向传播 through sampling 在 VAEs
- Run Metropolis-Hastings MCMC 到 sample 从 unnormalized target distribution

## Problem

language 模型 finishes processing your prompt 和 produces 向量 的 50,000 logits. One 为了 every token 在 its vocabulary. Now it has 到 pick one. How?

If it always picks highest-概率 token, every response 是 identical. Deterministic. Boring. If it picks uniformly 在 random, 输出 是 gibberish. answer lives somewhere between 这些 extremes, 和 somewhere 是 controlled 通过 sampling.

Sampling 是 not limited 到 text generation. Reinforcement learning estimates policy gradients 通过 sampling trajectories. VAEs learn latent representations 通过 sampling 从 learned distributions 和 backpropagating through randomness. Diffusion 模型 generate images 通过 sampling noise 和 iteratively denoising. Monte Carlo methods estimate integrals have no closed-form solution. MCMC 算法 explore high-dimensional posterior distributions 是 impossible 到 enumerate.

Every generative AI system 是 sampling system. sampling strategy determines quality, diversity, 和 controllability 的 输出. This lesson builds every major sampling method 从 scratch, starting 从 uniform random numbers 和 ending 使用 techniques power modern LLMs 和 generative 模型.

## Concept

### Why Sampling Matters

Sampling appears 在 four fundamental roles across AI 和 machine learning:

**Generation.** Language 模型, diffusion 模型, 和 GANs all produce 输出 通过 sampling. sampling 算法 directly controls creativity, coherence, 和 diversity. Temperature, top-k, 和 nucleus sampling 是 knobs engineers turn daily.

**训练.** Stochastic 梯度下降 samples mini-批次. Dropout samples 神经元 到 deactivate. 数据 augmentation samples random transformations. Importance sampling reweights samples 到 reduce gradient variance 在 reinforcement learning (PPO, TRPO).

**Estimation.** Many quantities 在 ML have no closed-form solution. expected loss over 数据 distribution, partition 函数 的 energy-based 模型, evidence 在 Bayesian inference. Monte Carlo estimation approximates all 的 这些 通过 averaging over samples.

**Exploration.** MCMC 算法 explore posterior distributions 在 Bayesian inference. Evolutionary strategies sample 参数 perturbations. Thompson sampling balances exploration 和 exploitation 在 bandits.

core challenge: you can only sample directly 从 simple distributions (uniform, normal). For everything else, you need method 到 convert simple samples into samples 从 your target distribution.

### Uniform Random Sampling

Every sampling method starts here. uniform random number generator produces values 在 [0, 1) where every sub-interval 的 equal length has equal 概率.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

To sample uniformly 从 discrete set 的 n items, generate U 和 return floor(n * U). To sample 从 continuous range [, b], compute + (b - ) * U.

key insight: single uniform random number contains exactly right amount 的 randomness 到 produce one sample 从 any distribution. trick 是 finding right transformation.

### Inverse CDF Method (Inverse Transform Sampling)

cumulative distribution 函数 (CDF) maps values 到 probabilities:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

inverse CDF maps probabilities back 到 values. If U ~ Uniform(0, 1), then X = F_inverse(U) follows target distribution.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Exponential distribution example:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

This works perfectly when you can write down F_inverse 在 closed form. For normal distribution, there 是 no closed-form inverse CDF, so we use other methods (Box-Muller, 或 numerical approximation).

**Discrete version:** For discrete distributions, build CDF 作为 cumulative sum, generate U, 和 find first index where cumulative sum exceeds U. 这是 how `sample_categorical` works 在 Lesson 06.

### Rejection Sampling

When you cannot invert CDF but can evaluate target PDF up 到 constant, rejection sampling works.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

tighter bound M, higher acceptance rate. In low dimensions (1-3), rejection sampling works well. In high dimensions, acceptance rate drops exponentially because most 的 proposal volume gets rejected. 这是 curse 的 dimensionality 为了 rejection sampling.

**Example: sampling 从 truncated normal.** Use uniform proposal over truncated range. envelope M 是 maximum 的 normal PDF 在 range.

**Example: sampling 从 semicircle.** Propose uniformly 在 bounding rectangle. Accept if point falls inside semicircle. 这是 how Monte Carlo computes pi: acceptance rate equals area ratio pi/4.

### Importance Sampling

Sometimes you do not need samples 从 target distribution p(x). 你需要 到 estimate expectation under p(x), 和 you have samples 从 different distribution q(x).

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这是 critical 在 reinforcement learning. In PPO (Proximal Policy Optimization), you collect trajectories under old policy pi_old but want 到 optimize new policy pi_new. importance 权重 是 pi_new(|s) / pi_old(|s). PPO clips 这些 权重 到 prevent new policy 从 diverging too far 从 old one.

variance 的 importance sampling estimator depends 在 how similar q 是 到 p. If q 是 very different 从 p, few samples get enormous 权重 和 dominate estimate. Self-normalized importance sampling divides 通过 sum 的 权重 到 reduce 这个 problem:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Estimation

Monte Carlo estimation approximates integrals 通过 averaging random samples. law 的 large numbers guarantees 收敛.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

error rate 是 dimension-independent. 这是 why Monte Carlo methods dominate 在 high dimensions where grid-based integration 是 impossible.

**Estimating pi:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Estimating expectations:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC constructs Markov chain whose stationary distribution 是 target distribution p(x). After enough steps, samples 从 chain 是 (approximately) samples 从 p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

For symmetric proposals (q(x'|x) = q(x|x')), ratio simplifies 到 p(x')/p(x). 这是 original Metropolis 算法.

**Why it works.** acceptance rule ensures detailed balance: 概率 的 being 在 x 和 moving 到 x' equals 概率 的 being 在 x' 和 moving 到 x. Detailed balance implies p(x) 是 stationary distribution 的 chain.

**Practical considerations:**
- Burn-在: discard early samples before chain reaches equilibrium
- Thinning: keep every k-th sample 到 reduce autocorrelation
- Proposal scale: too small 和 chain moves slowly (high acceptance, slow exploration); too large 和 most proposals 是 rejected (low acceptance, stuck 在 place)
- optimal acceptance rate 为了 Gaussian proposal 在 high dimensions 是 approximately 0.234

### Gibbs Sampling

Gibbs sampling 是 special case 的 MCMC 为了 multivariate distributions. Instead 的 proposing move 在 all dimensions 在 once, it updates one variable 在 time 从 its conditional distribution.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling requires you can sample 从 each conditional distribution p(x_i | x_{-i}). 这是 straightforward 为了 many 模型:
- Bayesian networks: conditionals follow 从 graph structure
- Gaussian mixtures: conditionals 是 Gaussian
- Ising 模型: each spin's conditional depends only 在 its neighbors

acceptance rate 是 always 1 (every proposal 是 accepted) because sampling 从 exact conditional automatically satisfies detailed balance.

**Limitation.** When variables 是 highly correlated, Gibbs sampling mixes slowly because updating one variable 在 time cannot make large diagonal moves through distribution.

### Temperature Sampling (Used 在 LLMs)

Language 模型 输出 logits z_1, ..., z_V 为了 each token 在 vocabulary. Softmax converts 这些 到 probabilities. Temperature rescales logits before softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Why it works.** Dividing logits 通过 T < 1 amplifies differences between logits. If z_1 = 2 和 z_2 = 1, dividing 通过 T = 0.5 gives z_1/T = 4 和 z_2/T = 2, making gap larger. After softmax, highest-logit token gets much larger share.

**In practice:**
- T = 0.0: greedy decoding, best 为了 factual Q&
- T = 0.3-0.7: slightly creative, good 为了 代码 generation
- T = 0.7-1.0: balanced, good 为了 general conversation
- T = 1.0-1.5: creative writing, brainstorming
- T > 1.5: increasingly random, rarely useful

Temperature does not change which tokens 是 possible. It changes 概率 mass allocated 到 each token.

### Top-k Sampling

Top-k sampling restricts candidate set 到 k tokens 使用 highest probabilities, then renormalizes 和 samples 从 restricted set.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k prevents 模型 从 selecting extremely unlikely tokens (typos, nonsense) exist 在 long tail 的 vocabulary distribution. problem: k 是 fixed regardless 的 context. When 模型 是 confident (one token has 95% 概率), k = 40 still allows 39 alternatives. When 模型 是 uncertain (概率 是 spread across 1000 tokens), k = 40 cuts off plausible options.

### Top-p (Nucleus) Sampling

Top-p sampling dynamically adjusts candidate set size. Instead 的 keeping fixed number 的 tokens, it keeps smallest set 的 tokens whose cumulative 概率 exceeds p.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

When 模型 是 confident, nucleus sampling keeps few tokens (maybe 2-3). When 模型 是 uncertain, it keeps many (maybe 200). This adaptive behavior 是 why nucleus sampling generally produces better text than top-k.

**Common combinations:**
- Temperature 0.7 + top-p 0.9: good general-purpose setting
- Temperature 0.0 (greedy): best 为了 deterministic tasks
- Temperature 1.0 + top-k 50: Fan et al. (2018) original paper setting

Top-k 和 top-p can be combined. Apply top-k first, then top-p 在 remaining set.

### Reparameterization Trick (Used 在 VAEs)

Variational autoencoders (VAEs) learn 通过 encoding 输入 into distribution 在 latent space, sampling 从 distribution, 和 decoding sample back. problem: you cannot backpropagate through sampling operation.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

reparameterization trick separates randomness 从 参数:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

This works because N(mu, sigma^2) has same distribution 作为 mu + sigma * N(0, 1). key insight: move randomness 到 参数-free source (epsilon), then express sample 作为 differentiable transformation 的 参数.

**In VAE 训练 loop:**
1. Encoder 输出 mu 和 log(sigma^2) 为了 each 输入
2. Sample epsilon ~ N(0, 1)
3. Compute z = mu + sigma * epsilon
4. Decode z 到 reconstruct 输入
5. Backpropagate through steps 4, 3, 2, 1 (possible because step 3 是 differentiable)

Without reparameterization trick, VAEs cannot be trained 使用 standard 反向传播. This single insight made VAEs practical.

### Gumbel-Softmax (Differentiable Categorical Sampling)

reparameterization trick works 为了 continuous distributions (Gaussian). For discrete categorical distributions, we need different approach. Gumbel-Softmax provides differentiable approximation 到 categorical sampling.

** Gumbel-Max trick (non-differentiable):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (differentiable approximation):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax produces continuous relaxation 的 discrete sample. 输出 是 概率 向量 (soft one-hot) instead 的 hard one-hot. Gradients flow through softmax. During forward pass 在 训练, you can use "straight-through" estimator: use hard argmax 为了 forward pass but soft Gumbel-Softmax gradients 为了 backward pass.

**Applications:**
- Discrete latent variables 在 VAEs
- Neural architecture search (choosing discrete operations)
- Hard attention mechanisms
- Reinforcement learning 使用 discrete actions

### Stratified Sampling

Standard Monte Carlo sampling can leave gaps 在 sample space 通过 chance. Stratified sampling forces even coverage 通过 dividing space into strata 和 sampling 从 each.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified sampling always has lower 或 equal variance compared 到 standard Monte Carlo:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Applications:**
- Numerical integration (quasi-Monte Carlo)
- 训练 数据 splits (ensuring class balance 在 each fold)
- Importance sampling 使用 stratification (combining both techniques)
- NeRF (Neural Radiance Fields) uses stratified sampling along camera rays

### Connection 到 Diffusion Models

Diffusion 模型 generate images through sampling process. forward process adds Gaussian noise 到 image over T steps until it becomes pure noise. reverse process learns 到 denoise, recovering original image step 通过 step.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

connection 到 methods 在 这个 lesson:
- Each denoising step uses reparameterization trick (sample noise, apply deterministic transform)
- noise schedule {alpha_t} controls form 的 temperature annealing
- 训练 uses Monte Carlo estimation 到 approximate ELBO (evidence lower bound)
- Ancestral sampling 在 diffusion 模型 是 Markov chain (each step depends only 在 current state)

entire image generation process 是 iterative sampling: start 从 noise, 和 在 each step, sample slightly less noisy version conditioned 在 learned denoising 模型.

## Build It

### Step 1: Uniform 和 inverse CDF sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Generate 10,000 exponential samples 和 verify mean 是 1/lambda.

### Step 2: Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Use rejection sampling 到 draw 从 truncated normal distribution. Verify shape 通过 histogramming samples.

### Step 3: Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Estimate E[X^2] under normal distribution using uniform proposal. Compare 到 known answer (mu^2 + sigma^2).

### Step 4: Monte Carlo estimation 的 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Step 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Sample 从 bimodal distribution (mixture 的 two Gaussians). Visualize chain's trajectory.

### Step 6: Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Step 7: Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Show how temperature changes 输出 distribution 为了 set 的 token logits.

### Step 8: Top-k 和 top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Step 9: Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Demonstrate gradients flow through reparameterized sample but not through direct sampling.

### Step 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Show how decreasing temperature makes 输出 approach one-hot 向量.

Full implementations 使用 all visualizations 是 在 `代码/sampling.py`.

## Use It

With NumPy 和 SciPy, production versions:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

For MCMC 在 scale, use dedicated libraries:
- PyMC: full Bayesian modeling 使用 NUTS (adaptive HMC)
- emcee: ensemble MCMC sampler
- NumPyro/JAX: GPU-accelerated MCMC

You built 这些 从 scratch. Now you know what library calls 是 doing.

## Exercises

1. Implement inverse CDF sampling 为了 Cauchy distribution. CDF 是 F(x) = 0.5 + arctan(x)/pi. Generate 10,000 samples 和 plot histogram against true PDF. Notice heavy tails (extreme values far 从 center).

2. Use rejection sampling 到 generate samples 从 Beta(2, 5) distribution using Uniform(0, 1) proposal. Plot accepted samples against true Beta PDF. What 是 theoretical acceptance rate?

3. Estimate integral 的 sin(x) 从 0 到 pi using Monte Carlo 使用 1,000, 10,000, 和 100,000 samples. Compare error 在 each level. Verify error scales 作为 O(1/sqrt(N)).

4. Implement Metropolis-Hastings 到 sample 从 2D distribution p(x, y) proportional 到 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Plot samples 和 chain trajectory. Experiment 使用 different proposal standard deviations.

5. Build complete text generation demo: given vocabulary 的 10 words 使用 logits, generate sequences 的 20 tokens using () greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9. Compare diversity 的 输出 across 5 runs.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sampling | "Drawing random values" | Generating values according 到 概率 distribution. mechanism behind all generative AI |
| Uniform distribution | "All equally likely" | Every value 在 [, b] has equal 概率 density 1/(b-). starting point 为了 all sampling methods |
| Inverse CDF | "概率 transform" | F_inverse(U) converts uniform sample into sample 从 any distribution 使用 known CDF. Exact 和 efficient |
| Rejection sampling | "Propose 和 accept/reject" | Generate 从 simple proposal, accept 使用 概率 proportional 到 target/proposal ratio. Exact but wastes samples |
| Importance sampling | "Reweight samples" | Estimate expectations under p(x) using samples 从 q(x) 通过 weighting each sample 通过 p(x)/q(x). Core 到 PPO 在 RL |
| Monte Carlo | "Average random samples" | Approximate integrals 作为 sample averages. Error O(1/sqrt(N)) regardless 的 dimension |
| MCMC | "Random walk converges" | Construct Markov chain whose stationary distribution 是 target. Metropolis-Hastings 是 foundational 算法 |
| Metropolis-Hastings | "Accept uphill, sometimes downhill" | Propose moves, accept based 在 density ratio. Detailed balance ensures 收敛 到 target distribution |
| Gibbs sampling | "One variable 在 time" | Update each variable 从 its conditional distribution holding others fixed. 100% acceptance rate |
| Temperature | "Confidence knob" | Divides logits 通过 T before softmax. T<1 sharpens (more confident), T>1 flattens (more diverse) |
| Top-k sampling | "Keep k best" | Zero out all but k highest-概率 tokens, renormalize, sample. Fixed candidate set size |
| Nucleus sampling (top-p) | "Keep probable ones" | Keep smallest set 的 tokens whose cumulative 概率 exceeds p. Adaptive candidate set size |
| Reparameterization trick | "Move randomness outside" | Write z = mu + sigma * epsilon where epsilon ~ N(0,1). Makes sampling differentiable. Essential 为了 VAE 训练 |
| Gumbel-Softmax | "Soft categorical sampling" | Differentiable approximation 到 categorical sampling using Gumbel noise + softmax 使用 temperature |
| Stratified sampling | "Forced coverage" | Divide sample space into strata, sample 从 each. Always lower variance than naive Monte Carlo |
| Burn-在 | "Warm-up period" | Initial MCMC samples discarded before chain reaches its stationary distribution |
| Detailed balance | "Reversibility condition" | p(x) * T(x->y) = p(y) * T(y->x). Sufficient condition 为了 p 到 be stationary distribution 的 Markov chain |
| Diffusion sampling | "Iterative denoising" | Generate 数据 通过 starting 从 noise 和 applying learned denoising steps. Each step 是 conditional sampling operation |

## Further Reading

- [Holbrook (2023): Metropolis-Hastings 算法](https://arxiv.org/abs/2304.07010) - detailed tutorial 在 MCMC foundations
- [Jang, Gu, Poole (2017): Categorical Reparameterization 使用 Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - original Gumbel-Softmax paper
- [Holtzman et al. (2020): Curious Case 的 Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus (top-p) sampling paper
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - VAE paper introducing reparameterization trick
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM connects sampling 到 image generation
