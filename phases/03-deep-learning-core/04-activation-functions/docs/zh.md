# 激活函数

> Without nonlinearity, your 100-层 network is a fancy matrix multiply. Activations are the gates that let 神经网络s think in curves.

**类型:** 实现
**语言:** Python
**Prerequisites:** Lesson 03.03 (反向传播)
**Time:** ~75 minutes

## 学习目标

- Implement sigmoid, tanh, ReLU, Leaky ReLU, GELU, Swish, and softmax with their derivatives from scratch
- Diagnose the vanishing 梯度 problem by measuring activation magnitudes through 10+ 层s with different activations
- Detect dead neurons in a ReLU network and explain why GELU avoids this failure mode
- Select the correct 激活函数 for a given architecture (transformer, CNN, RNN, output 层)

## The Problem

Stack two linear transformations: y = W2(W1x + b1) + b2. Expand it: y = W2W1x + W2b1 + b2. That's just y = Ax + c -- a single linear transformation. No matter how many linear 层s you stack, the result collapses to one matrix multiply. Your 100-层 network has the same representational power as a single 层.

This is not a theoretical curiosity. It means a deep linear network literally cannot learn XOR, cannot classify a spiral dataset, cannot recognize a face. Without 激活函数s, depth is an illusion.

Activation functions break the linearity. They warp the output of each 层 through a nonlinear function, giving the network the ability to bend decision boundaries, approximate arbitrary functions, and actually learn. But pick the wrong activation and your 梯度s vanish to zero (sigmoid in deep networks), explode to infinity (unbounded activations without careful initialization), or your neurons die permanently (ReLU with large negative 偏置es). The choice of 激活函数 directly determines whether your network learns at all.

## The Concept

### Why Nonlinearity Is Necessary

Matrix multiplication is composable. Multiplying a vector by matrix A then matrix B is identical to multiplying by AB. This means stacking ten linear 层s is mathematically equivalent to one linear 层 with one big matrix. All those parameters, all that depth -- wasted. You need something to break the chain. That's what 激活函数s do.

Here is the proof. A linear 层 computes f(x) = Wx + b. Stack two:

```
层 1: h = W1 * x + b1
层 2: y = W2 * h + b2
```

Substitute:

```
y = W2 * (W1 * x + b1) + b2
y = (W2 * W1) * x + (W2 * b1 + b2)
y = A * x + c
```

One 层. Insert a nonlinear activation g() between 层s:

```
h = g(W1 * x + b1)
y = W2 * h + b2
```

Now the substitution breaks. W2 * g(W1 * x + b1) + b2 cannot be reduced to a single linear transformation. The network can represent nonlinear functions. Each additional 层 with an activation adds representational capacity.

### Sigmoid

The original 激活函数 for 神经网络s.

```
sigmoid(x) = 1 / (1 + e^(-x))
```

Output range: (0, 1). Smooth, differentiable, maps any real number to a probability-like value.

The derivative:

```
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
```

The maximum value of this derivative is 0.25, occurring at x = 0. In 反向传播, 梯度s multiply through 层s. Ten 层s of sigmoid means the 梯度 gets multiplied by at most 0.25 ten times:

```
0.25^10 = 0.000000953674
```

Less than one millionth of the original signal. This is the vanishing 梯度 problem. 梯度s in early 层s become so small that 权重s barely update. The network appears to learn -- loss decreases in later 层s -- but the first 层s are frozen. Deep sigmoid networks simply do not train.

Additional problem: sigmoid outputs are always positive (0 to 1), which means 梯度s on 权重s are always the same sign. This causes zig-zagging during 梯度下降.

### Tanh

The centered version of sigmoid.

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

Output range: (-1, 1). Zero-centered, which eliminates the zig-zag problem.

The derivative:

```
tanh'(x) = 1 - tanh(x)^2
```

Maximum derivative is 1.0 at x = 0 -- four times better than sigmoid. But the vanishing 梯度 problem still exists. For large positive or negative inputs, the derivative approaches zero. Ten 层s still crush the 梯度, just less aggressively.

### ReLU: The Breakthrough

Rectified Linear Unit. Popularized for deep learning by Nair and Hinton in 2010 (the function itself dates to Fukushima's 1969 work), it changed everything.

```
relu(x) = max(0, x)
```

Output range: [0, infinity). The derivative is trivially simple:

```
relu'(x) = 1  if x > 0
            0  if x <= 0
```

No vanishing 梯度 for positive inputs. The 梯度 is exactly 1, passed straight through. This is why deep networks became trainable -- ReLU preserves 梯度 magnitude across 层s.

But there is a failure mode: the dead neuron problem. If a neuron's 权重ed input is always negative (due to a large negative 偏置 or unfortunate 权重 initialization), its output is always zero, its 梯度 is always zero, and it never updates. It is permanently dead. In practice, 10-40% of neurons in a ReLU network can die during training.

### Leaky ReLU

The simplest fix for dead neurons.

```
leaky_relu(x) = x        if x > 0
                alpha * x if x <= 0
```

Where alpha is a small constant, typically 0.01. The negative side has a small slope instead of zero, so dead neurons still get a 梯度 signal and can recover.

### GELU: The Modern Default

Gaussian Error Linear Unit. Introduced by Hendrycks and Gimpel in 2016. Default activation in BERT, GPT, and most modern transformers.

```
gelu(x) = x * Phi(x)
```

Where Phi(x) is the cumulative distribution function of the standard normal distribution. The approximation used in practice:

```
gelu(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

GELU is smooth everywhere, allows small negative values (unlike ReLU which hard-clips to zero), and has a probabilistic interpretation: it 权重s each input by how likely it is to be positive under a Gaussian distribution. This smooth gating outperforms ReLU in transformer architectures because it provides better 梯度 flow and avoids the dead neuron problem entirely.

### Swish / SiLU

Self-gated activation discovered by Ramachandran et al. in 2017 through automated search.

```
swish(x) = x * sigmoid(x)
```

Swish is formally x * sigmoid(x). Google discovered it through automated search over 激活函数 space -- a 神经网络 designing parts of 神经网络s.

Like GELU, it is smooth, non-monotonic, and allows small negative values. The difference is subtle: Swish uses sigmoid for gating while GELU uses the Gaussian CDF. In practice, performance is nearly identical. Swish is used in EfficientNet and some vision models. GELU dominates in language models.

### Softmax: The Output Activation

Not used in hidden 层s. Softmax converts a vector of raw scores (logits) into a probability distribution.

```
softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
```

Every output is between 0 and 1. All outputs sum to 1. This makes it the standard final activation for multi-class classification. The largest logit gets the highest probability, but unlike argmax, softmax is differentiable and preserves information about relative confidence.

### Comparison of Shapes

```mermaid
graph LR
    subgraph "激活函数s"
        S["Sigmoid<br/>Range: (0,1)<br/>Saturates both ends"]
        T["Tanh<br/>Range: (-1,1)<br/>Zero-centered"]
        R["ReLU<br/>Range: [0,inf)<br/>Dead neurons"]
        G["GELU<br/>Range: ~(-0.17,inf)<br/>Smooth gating"]
    end
    S -->|"Vanishing 梯度"| Problem["Deep networks<br/>don't train"]
    T -->|"Less severe but<br/>still vanishes"| Problem
    R -->|"梯度 = 1<br/>for x > 0"| Solution["Deep networks<br/>train fast"]
    G -->|"Smooth 梯度<br/>everywhere"| Solution
```

### 梯度 Flow Comparison

```mermaid
graph TD
    Input["Input Signal"] --> L1["层 1"]
    L1 --> L5["层 5"]
    L5 --> L10["层 10"]
    L10 --> Output["Output"]

    subgraph "梯度 at 层 1"
        SigGrad["Sigmoid: ~0.000001"]
        TanhGrad["Tanh: ~0.001"]
        ReluGrad["ReLU: ~1.0"]
        GeluGrad["GELU: ~0.8"]
    end
```

### Which Activation When

```mermaid
flowchart TD
    Start["What are you building?"] --> Hidden{"Hidden 层s<br/>or output?"}

    Hidden -->|"Hidden 层s"| Arch{"Architecture?"}
    Hidden -->|"Output 层"| Task{"Task type?"}

    Arch -->|"Transformer / NLP"| GELU["Use GELU"]
    Arch -->|"CNN / Vision"| ReLU["Use ReLU or Swish"]
    Arch -->|"RNN / LSTM"| Tanh["Use Tanh"]
    Arch -->|"Simple MLP"| ReLU2["Use ReLU"]

    Task -->|"Binary classification"| Sigmoid["Use Sigmoid"]
    Task -->|"Multi-class classification"| Softmax["Use Softmax"]
    Task -->|"Regression"| Linear["Use Linear (no activation)"]
```

## Build It

### Step 1: Implement All 激活函数s with Derivatives

Each function takes a single float and returns a float. Each derivative function takes the same input and returns the 梯度.

```python
import math

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return math.tanh(x)

def tanh_derivative(x):
    t = math.tanh(x)
    return 1 - t * t

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
```

### Step 2: Visualize Where 梯度s Die

Compute the 梯度 at 100 evenly-spaced points from -5 to 5. Print a text histogram showing where each activation's 梯度 is near-zero.

```python
def 梯度_scan(name, derivative_fn, start=-5, end=5, n=100):
    step = (end - start) / n
    near_zero = 0
    healthy = 0
    for i in range(n):
        x = start + i * step
        g = derivative_fn(x)
        if abs(g) < 0.01:
            near_zero += 1
        else:
            healthy += 1
    pct_dead = near_zero / n * 100
    print(f"{name:15s}: {healthy:3d} healthy, {near_zero:3d} near-zero ({pct_dead:.0f}% dead zone)")

梯度_scan("Sigmoid", sigmoid_derivative)
梯度_scan("Tanh", tanh_derivative)
梯度_scan("ReLU", relu_derivative)
梯度_scan("Leaky ReLU", leaky_relu_derivative)
梯度_scan("GELU", gelu_derivative)
梯度_scan("Swish", swish_derivative)
```

### Step 3: Vanishing 梯度 Experiment

Forward-pass a signal through N 层s using sigmoid vs ReLU. Measure how the activation magnitude changes.

```python
import random

def vanishing_梯度_experiment(activation_fn, name, n_层s=10, n_inputs=5):
    random.seed(42)
    values = [random.gauss(0, 1) for _ in range(n_inputs)]

    print(f"\n{name} through {n_层s} 层s:")
    for 层 in range(n_层s):
        权重s = [random.gauss(0, 1) for _ in range(n_inputs)]
        z = sum(w * v for w, v in zip(权重s, values))
        activated = activation_fn(z)
        magnitude = abs(activated)
        bar = "#" * int(magnitude * 20)
        print(f"  层 {层+1:2d}: magnitude = {magnitude:.6f} {bar}")
        values = [activated] * n_inputs

vanishing_梯度_experiment(sigmoid, "Sigmoid")
vanishing_梯度_experiment(relu, "ReLU")
vanishing_梯度_experiment(gelu, "GELU")
```

### Step 4: Dead Neuron Detector

Create a ReLU network, pass random inputs through it, count how many neurons never fire.

```python
def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    random.seed(0)
    权重s = [[random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)]
    偏置es = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * hidden_size

    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(权重s[neuron_idx], inputs)) + 偏置es[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx] += 1

    dead = sum(1 for c in fire_counts if c == 0)
    rarely_fire = sum(1 for c in fire_counts if 0 < c < n_samples * 0.05)
    healthy = hidden_size - dead - rarely_fire

    print(f"\nDead Neuron Report ({hidden_size} neurons, {n_samples} samples):")
    print(f"  Dead (never fired):     {dead}")
    print(f"  Barely alive (<5%):     {rarely_fire}")
    print(f"  Healthy:                {healthy}")
    print(f"  Dead neuron rate:       {dead/hidden_size*100:.1f}%")

    for i, c in enumerate(fire_counts):
        status = "DEAD" if c == 0 else "WEAK" if c < n_samples * 0.05 else "OK"
        bar = "#" * (c * 40 // n_samples)
        print(f"  Neuron {i:2d}: {c:4d}/{n_samples} fires [{status:4s}] {bar}")

dead_neuron_detector()
```

### Step 5: Training Comparison -- Sigmoid vs ReLU vs GELU

Train the same two-层 network on the circle dataset (points inside a circle = class 1, outside = class 0) with three different activations. Compare 收敛 speed.

```python
def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class ActivationNetwork:
    def __init__(self, activation_fn, activation_deriv, hidden_size=8, lr=0.1):
        random.seed(0)
        self.act = activation_fn
        self.act_d = activation_deriv
        self.lr = lr
        self.hidden_size = hidden_size

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(self.act(z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        error = self.out - target
        d_out = error * self.out * (1 - self.out)

        for i in range(self.hidden_size):
            d_h = d_out * self.w2[i] * self.act_d(self.z1[i])
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += (pred - y) ** 2
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append(avg_loss)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


data = make_circle_data()

configs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("ReLU", relu, relu_derivative),
    ("GELU", gelu, gelu_derivative),
]

results = {}
for name, act_fn, act_d_fn in configs:
    print(f"\n=== Training with {name} ===")
    net = ActivationNetwork(act_fn, act_d_fn, hidden_size=8, lr=0.1)
    losses = net.train(data, epochs=200)
    results[name] = losses

print("\n=== Final Loss Comparison ===")
for name, losses in results.items():
    print(f"  {name:10s}: start={losses[0]:.4f} -> end={losses[-1]:.4f} (improvement: {(1 - losses[-1]/losses[0])*100:.1f}%)")
```

## Use It

PyTorch provides all of these as both functional and module forms:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 10)

relu_out = F.relu(x)
gelu_out = F.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = F.silu(x)

logits = torch.randn(4, 5)
probs = F.softmax(logits, dim=1)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.GELU(),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 5),
)
```

Hidden 层s in a transformer: GELU. Hidden 层s in a CNN: ReLU. Output 层 for classification: softmax. Output 层 for regression: none (linear). Output 层 for probabilities: sigmoid. That's it. Start with these defaults. Change them only when you have evidence.

RNNs and LSTMs use tanh for hidden state and sigmoid for gates, but if you're building from scratch today, you're probably not using RNNs. If neurons are dying in your ReLU network, switch to GELU. Don't reach for Leaky ReLU unless you have a specific reason -- GELU solves the dead neuron problem and gives better 梯度 flow.

## Ship It

This lesson produces:
- `outputs/prompt-activation-selector.md` -- a reusable prompt that helps you pick the right 激活函数 for any architecture

## Exercises

1. Implement Parametric ReLU (PReLU) where the negative slope alpha is a learnable parameter. Train it on the circle dataset and compare to fixed Leaky ReLU.

2. Run the vanishing 梯度 experiment with 50 层s instead of 10. Plot the magnitude at each 层 for sigmoid, tanh, ReLU, and GELU. At which 层 does each activation's signal effectively reach zero?

3. Implement the ELU (Exponential Linear Unit): elu(x) = x if x > 0, alpha * (e^x - 1) if x <= 0. Compare its dead neuron rate to ReLU on the same network.

4. Build a "梯度 health monitor" that runs during training: at each epoch, compute the average 梯度 magnitude at each 层. Print a warning when any 层's 梯度 drops below 0.001 or exceeds 100.

5. Modify the training comparison to use the XOR dataset from Lesson 01 instead of circles. Which activation converges fastest on XOR? Why does this differ from the circle results?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Activation function | "The nonlinear part" | A function applied to each neuron's output that breaks linearity, enabling the network to learn nonlinear mappings |
| Vanishing 梯度 | "梯度s disappear in deep networks" | 梯度s shrink exponentially through 层s when the activation's derivative is less than 1, making early 层s untrainable |
| Exploding 梯度 | "梯度s blow up" | 梯度s grow exponentially through 层s when the effective multiplier exceeds 1, causing unstable training |
| Dead neuron | "A neuron that stopped learning" | A ReLU neuron whose input is permanently negative, producing zero output and zero 梯度 |
| Sigmoid | "Squishes values to 0-1" | The logistic function 1/(1+e^-x), historically important but causes vanishing 梯度s in deep networks |
| ReLU | "Clips negatives to zero" | max(0, x) -- the activation that made deep learning practical by preserving 梯度 magnitude |
| GELU | "The transformer activation" | Gaussian Error Linear Unit, a smooth activation that 权重s inputs by their probability of being positive |
| Swish/SiLU | "Self-gated ReLU" | x * sigmoid(x), discovered through automated search, used in EfficientNet |
| Softmax | "Turns scores into probabilities" | Normalizes a vector of logits into a probability distribution where all values are in (0,1) and sum to 1 |
| Leaky ReLU | "ReLU that doesn't die" | max(alpha*x, x) where alpha is small (0.01), preventing dead neurons by allowing small negative 梯度s |
| Saturation | "The flat part of sigmoid" | Regions where an activation's derivative approaches zero, blocking 梯度 flow |
| Logit | "The raw score before softmax" | The unnormalized output of the final 层 before applying softmax or sigmoid |

## Further Reading

- Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines" (2010) -- the paper that introduced ReLU and enabled training of deep networks
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016) -- introduced the 激活函数 that became the default for transformers
- Ramachandran et al., "Searching for 激活函数s" (2017) -- used automated search to discover Swish, showing that activation design can be automated
- Glorot & Bengio, "Understanding the difficulty of training deep 前向传播 神经网络s" (2010) -- the paper that diagnosed vanishing/exploding 梯度s and proposed Xavier initialization
- Goodfellow, Bengio, Courville, "深度学习" Chapter 6.3 (https://www.deeplearningbook.org/) -- rigorous treatment of hidden units and 激活函数s
