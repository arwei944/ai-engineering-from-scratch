# 感知机

> 感知机 是 atom 的 神经网络. Split it open 和 you find 权重, 偏置, 和 decision.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (线性代数 Intuition)
**Time:** ~60 minutes

## Learning Objectives

- Implement 感知机 从 scratch 在 Python, including 权重 update rule 和 step 激活函数
- Explain why single 感知机 can only solve linearly separable problems 和 demonstrate XOR failure case
- Construct multi-层 感知机 通过 composing OR, NAND, 和 AND gates 到 solve XOR
- Train two-层 network 使用 sigmoid activation 和 反向传播 到 learn XOR automatically

## Problem

你知道 向量 和 dot products. 你知道 矩阵 transforms 输入 into 输出. But how does machine *learn* which transformation 到 use?

感知机 answers 这个. It's simplest possible learning machine: take some 输入, multiply 通过 权重, add 偏置, 和 make binary decision. Then adjust. That's it. Every 神经网络 ever built 是 层 的 这个 idea stacked together.

Understanding 感知机 means understanding what "learning" actually means 在 代码: adjusting numbers until 输出 matches reality.

## Concept

### One 神经元, One Decision

感知机 takes n 输入, multiplies each 通过 权重, sums them up, adds 偏置, 和 passes result through 激活函数.

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]
```

step 函数 是 brutal: if weighted sum plus 偏置 是 >= 0, 输出 1. Otherwise, 输出 0.

```
step(z) = 1  if z >= 0
           0  if z < 0
```

这是 linear classifier. 权重 和 偏置 define line (或 hyperplane 在 higher dimensions) splits 输入 space into two regions.

### Decision Boundary

For two 输入, 感知机 draws line through 2D space:

```
  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1
```

Everything 在 one side 的 line 输出 0. Everything 在 other side 输出 1. 训练 moves 这个 line until it correctly separates classes.

### Learning Rule

感知机 learning rule 是 simple:

```
For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

If prediction 是 correct, error = 0, nothing changes. If it predicts 0 but should be 1, 权重 increase. If it predicts 1 but should be 0, 权重 decrease. 学习率 controls how big each adjustment 是.

### XOR Problem

Here's where it breaks. Look 在 这些 logic gates:

```
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```

AND 和 OR 是 linearly separable: you can draw single line 到 separate 0s 从 1s. XOR 是 not. No single line can separate [0,1] 和 [1,0] 从 [0,0] 和 [1,1].

```
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!
```

这是 fundamental limit. single 感知机 can only solve linearly separable problems. Minsky 和 Papert proved 这个 在 1969 和 it nearly killed 神经网络 research 为了 decade.

fix: stack perceptrons into 层. multi-层 感知机 can solve XOR 通过 combining two linear decisions into nonlinear one.

## Build It

### Step 1: 感知机 class

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```

### Step 2: Train 在 logic gates

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### Step 3: Watch XOR fail

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```

It will never converge. 这是 hard proof single 感知机 cannot learn XOR.

### Step 4: Solve XOR 使用 two 层

trick: XOR = (x1 OR x2) AND NOT (x1 AND x2). Combine three perceptrons:

```mermaid
graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```

All four cases correct. Stacking perceptrons into 层 creates decision boundaries no single 感知机 can produce.

### Step 5: Train Two-层 Network

Step 4 hand-wired 权重. That works 为了 XOR, but not 为了 real problems where you don't know right 权重 在 advance. fix: replace step 函数 使用 sigmoid 和 learn 权重 automatically through 反向传播.

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```

Two key differences 从 Step 4. First, sigmoid replaces step 函数 -- it's smooth, so gradients exist. Second, `train` method propagates error backward 从 输出 到 hidden 层, adjusting every 权重 proportionally 到 its contribution 到 error. That's 反向传播 在 20 lines.

这是 bridge 到 Lesson 03. math behind `d_output` 和 `hidden_deltas` 是 chain rule applied 到 network graph. We'll derive it properly there.

## Use It

Everything you just built 从 scratch exists 在 one import:

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

Five lines. Your 30-line `感知机` class does same thing. sklearn version adds 收敛 checks, multiple loss 函数, 和 sparse 输入 support -- but core loop 是 identical: weighted sum, step 函数, 权重 update 在 error.

real gap shows up 在 scale. What changes 在 production networks:

- step 函数 becomes sigmoid, ReLU, 或 other smooth activations
- Weights 是 learned automatically via 反向传播 (Lesson 03)
- Layers get deeper: 3, 10, 100+ 层
- same principle holds: each 层 creates new 特征 从 previous 层's 输出

single 感知机 can only draw straight lines. Stack them, 和 you can draw any shape.

## Ship It

This lesson produces:
- `输出/skill-感知机.md` - skill covering when single-层 vs multi-层 architectures 是 needed

## Exercises

1. Train 感知机 在 NAND gate ( universal gate - any logic circuit can be built 从 NAND). Verify its 权重 和 偏置 form valid decision boundary.
2. Modify 感知机 class 到 track decision boundary (w1*x1 + w2*x2 + b = 0) 在 each 轮次. Print how line shifts during 训练 在 AND gate.
3. Build 3-输入 感知机 输出 1 only when 在 least 2 的 3 输入 是 1 ( majority vote 函数). Is 这个 linearly separable? Why?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| 感知机 | " fake 神经元" | linear classifier: dot product 的 输入 和 权重, plus 偏置, through step 函数 |
| 权重 | "How important 输入 是" | multiplier scales each 输入's contribution 到 decision |
| 偏置 | " threshold" | constant shifts decision boundary, letting 感知机 fire even 使用 zero 输入 |
| Activation 函数 | " thing squishes values" | 函数 applied after weighted sum - step 函数 为了 perceptrons, sigmoid/ReLU 为了 modern networks |
| Linearly separable | "你可以 draw line between them" | 数据集 where single hyperplane can perfectly separate classes |
| XOR problem | " thing perceptrons can't do" | Proof single-层 networks cannot learn non-linearly-separable 函数 |
| Decision boundary | "Where classifier switches" | hyperplane w*x + b = 0 divides 输入 space into two classes |
| Multi-层 感知机 | " real 神经网络" | Perceptrons stacked 在 层, where each 层's 输出 feeds next 层's 输入 |

## Further Reading

- Frank Rosenblatt, " 感知机: Probabilistic 模型 为了 Information Storage 和 Organization 在 Brain" (1958) -- original paper started it all
- Minsky & Papert, "Perceptrons" (1969) -- book proved XOR was unsolvable 通过 single-层 networks 和 killed 感知机 research 为了 decade
- Michael Nielsen, "Neural Networks 和 Deep Learning", Chapter 1 (http://neuralnetworksanddeeplearning.com/) -- free online, best visual explanation 的 how perceptrons compose into networks
