# 多层神经网络 and Forward Pass

> One neuron draws a line. Stack them, and you can draw anything.

**类型:** 实现
**语言:** Python
**Prerequisites:** Phase 01 (Math Foundations), Lesson 03.01 (The 感知机)
**Time:** ~90 minutes

## 学习目标

- Build a multi-层 network from scratch with 层 and Network classes that perform a complete forward pass
- Trace matrix dimensions through each 层 of a network and identify shape mismatches
- Explain how stacking nonlinear activations enables a network to learn curved decision boundaries
- Solve the XOR problem using a 2-2-1 architecture with hand-tuned sigmoid 权重s

## The Problem

A single neuron is a line drawer. That's it. One straight line through your data. Every real problem in AI -- image recognition, language understanding, playing Go -- requires curves. Stacking neurons into 层s is how you get curves.

In 1969, Minsky and Papert proved this limitation was fatal: a single-层 network cannot learn XOR. Not "struggles to learn" -- mathematically cannot. The XOR truth table places [0,1] and [1,0] on one side, [0,0] and [1,1] on the other. No single line separates them.

This killed 神经网络 funding for over a decade. The fix was obvious in hindsight: stop using one 层. Stack neurons into 层s. Let the first 层 carve the input space into new features, and let the second 层 combine those features into decisions no single line could make.

That stack is the multi-层 network. It is the foundation of every deep learning model in production today. The forward pass -- data flowing from input through hidden 层s to output -- is the first thing you need to build before anything else works.

## The Concept

### 层s: Input, Hidden, Output

A multi-层 network has three types of 层s:

**Input 层** -- not really a 层. It holds your raw data. Two features means two input nodes. No computation happens here.

**Hidden 层s** -- where the work happens. Each neuron takes every output from the previous 层, applies 权重s and a 偏置, then passes the result through an 激活函数. "Hidden" because you never see these values directly in the training data.

**Output 层** -- the final answer. For binary classification, one neuron with sigmoid. For multi-class, one neuron per class.

```mermaid
graph LR
    subgraph Input["Input 层"]
        x1["x1"]
        x2["x2"]
    end
    subgraph Hidden["Hidden 层 (3 neurons)"]
        h1["h1"]
        h2["h2"]
        h3["h3"]
    end
    subgraph Output["Output 层"]
        y["y"]
    end
    x1 --> h1
    x1 --> h2
    x1 --> h3
    x2 --> h1
    x2 --> h2
    x2 --> h3
    h1 --> y
    h2 --> y
    h3 --> y
```

This is a 2-3-1 network. Two inputs, three hidden neurons, one output. Every connection carries a 权重. Every neuron (except input) carries a 偏置.

Each 层 produces a vector of numbers called a hidden state. For text, hidden states increase dimensionality -- encoding a word as 768 numbers to capture semantic meaning. For images, they reduce dimensionality -- compressing millions of pixels into a manageable representation. The hidden state is where the learning lives.

### Neurons and Activations

Each neuron does three things:

1. Multiply every input by its corresponding 权重
2. Sum all the products and add a 偏置
3. Pass the sum through an 激活函数

For now, the activation is sigmoid:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Sigmoid squashes any number into the range (0, 1). Large positive inputs push toward 1. Large negative inputs push toward 0. Zero maps to 0.5. This smooth curve is what makes learning possible -- unlike the 感知机's hard step, sigmoid has a 梯度 everywhere.

### Forward Pass: How Data Flows

The forward pass pushes input data through the network, 层 by 层, until it reaches the output. No learning happens during the forward pass. It is pure computation: multiply, add, activate, repeat.

```mermaid
graph TD
    X["Input: [x1, x2]"] --> WH["Multiply by 权重 Matrix W1 (2x3)"]
    WH --> BH["Add 偏置 Vector b1 (3,)"]
    BH --> AH["Apply sigmoid to each element"]
    AH --> H["Hidden Output: [h1, h2, h3]"]
    H --> WO["Multiply by 权重 Matrix W2 (3x1)"]
    WO --> BO["Add 偏置 Vector b2 (1,)"]
    BO --> AO["Apply sigmoid"]
    AO --> Y["Output: y"]
```

At each 层, three operations happen in sequence:

```
z = W * input + b       (linear transformation)
a = sigmoid(z)           (activation)
```

The output of one 层 becomes the input to the next. That is the entire forward pass.

### Matrix Dimensions

Tracking dimensions is the single most important debugging skill in deep learning. Here is the 2-3-1 network:

| Step | Operation | Dimensions | Result Shape |
|------|-----------|------------|-------------|
| Input | x | -- | (2,) |
| Hidden linear | W1 * x + b1 | W1: (3, 2), b1: (3,) | (3,) |
| Hidden activation | sigmoid(z1) | -- | (3,) |
| Output linear | W2 * h + b2 | W2: (1, 3), b2: (1,) | (1,) |
| Output activation | sigmoid(z2) | -- | (1,) |

The rule: 权重 matrix W at 层 k has shape (neurons_in_层_k, neurons_in_层_k_minus_1). Rows match the current 层. Columns match the previous 层. If the shapes do not line up, you have a bug.

### Universal Approximation Theorem

In 1989, George Cybenko proved something remarkable: a 神经网络 with a single hidden 层 and enough neurons can approximate any continuous function to any desired accuracy.

This does not mean one hidden 层 is always best. It means the architecture is theoretically capable. In practice, deeper networks (more 层s, fewer neurons per 层) learn the same functions with far fewer total parameters than shallow-wide networks. That is why deep learning works.

The intuition: each neuron in the hidden 层 learns one "bump" or feature. Enough bumps placed in the right locations can approximate any smooth curve. More neurons, more bumps, better approximation.

```mermaid
graph LR
    subgraph FewNeurons["4 Hidden Neurons"]
        A["Rough approximation"]
    end
    subgraph MoreNeurons["16 Hidden Neurons"]
        B["Close approximation"]
    end
    subgraph ManyNeurons["64 Hidden Neurons"]
        C["Near-perfect fit"]
    end
    FewNeurons --> MoreNeurons --> ManyNeurons
```

### Composability

Neural networks are composable. You can stack them, chain them, run them in parallel. A Whisper model uses an encoder network to process audio and a separate decoder network to generate text. Modern LLMs are decoder-only. BERT is encoder-only. T5 is encoder-decoder. The architecture choice defines what the model can do.

## Build It

Pure Python. No numpy. Every matrix operation written from scratch.

### Step 1: Sigmoid Activation

```python
import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
```

The clamp to [-500, 500] prevents overflow. `math.exp(500)` is large but finite. `math.exp(1000)` is infinity.

### Step 2: 层 Class

The most important operation in all of deep learning is matrix multiplication. Every 层, every attention head, every forward pass -- it's matmuls all the way down. A linear 层 takes an input vector, multiplies it by a 权重 matrix, and adds a 偏置 vector: y = Wx + b. That single equation is 90% of the compute in a 神经网络.

A 层 holds a 权重 matrix and a 偏置 vector. Its forward method takes an input vector and returns the activated output.

```python
class 层:
    def __init__(self, n_inputs, n_neurons, 权重s=None, 偏置es=None):
        if 权重s is not None:
            self.权重s = 权重s
        else:
            import random
            self.权重s = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if 偏置es is not None:
            self.偏置es = 偏置es
        else:
            self.偏置es = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.权重s)):
            z = sum(
                w * x for w, x in zip(self.权重s[neuron_idx], inputs)
            )
            z += self.偏置es[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output
```

The 权重 matrix has shape (n_neurons, n_inputs). Each row is one neuron's 权重s across all inputs. The forward method loops through neurons, computes the 权重ed sum plus 偏置, applies sigmoid, and collects the results.

### Step 3: Network Class

A network is a list of 层s. The forward pass chains them: output of 层 k feeds into 层 k+1.

```python
class Network:
    def __init__(self, 层s):
        self.层s = 层s

    def forward(self, inputs):
        current = inputs
        for 层 in self.层s:
            current = 层.forward(current)
        return current
```

That is the entire forward pass. Four lines of logic. Data goes in, flows through every 层, comes out the other side.

### Step 4: XOR with Hand-Tuned 权重s

In Lesson 01, we solved XOR by combining OR, NAND, and AND 感知机s. Now do the same thing with our 层 and Network classes. The 2-2-1 architecture: two inputs, two hidden neurons, one output.

```python
hidden = 层(
    n_inputs=2,
    n_neurons=2,
    权重s=[[20.0, 20.0], [-20.0, -20.0]],
    偏置es=[-10.0, 30.0],
)

output = 层(
    n_inputs=2,
    n_neurons=1,
    权重s=[[20.0, 20.0]],
    偏置es=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected})")
```

The large 权重s (20, -20) make sigmoid act like a step function. The first hidden neuron approximates OR. The second approximates NAND. The output neuron combines them into AND, which is XOR.

### Step 5: Circle Classification

A harder problem: classify 2D points as inside or outside a circle of radius 0.5 centered at the origin. This requires a curved decision boundary -- impossible for a single 感知机.

```python
import random
import math

random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

circle_net = Network([
    层(n_inputs=2, n_neurons=8),
    层(n_inputs=8, n_neurons=1),
])
```

With random 权重s, the network will not classify well. But the forward pass still runs. This is the point -- the forward pass is just computation. Learning the right 权重s is 反向传播, coming in Lesson 03.

```python
correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

print(f"Accuracy with random 权重s: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")
```

Random 权重s give poor accuracy -- often worse than guessing the majority class. After training (Lesson 03), this same architecture with 8 hidden neurons will draw a curved boundary that separates inside from outside.

## Use It

PyTorch does everything above in four lines:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
output = model(x)
print(output)
```

`nn.Linear(2, 8)` is your 层 class: 权重 matrix of shape (8, 2), 偏置 vector of shape (8,). `nn.Sigmoid()` is your sigmoid function applied element-wise. `nn.Sequential` is your Network class: chain 层s in order.

The difference is speed and scale. PyTorch runs on GPUs, handles 批次es of millions of samples, and automatically computes 梯度s for 反向传播. But the forward pass logic is identical to what you just built from scratch.

## Ship It

This lesson produces a reusable prompt for designing network architectures:

- `outputs/prompt-network-architect.md`

Use it when you need to decide how many 层s, how many neurons per 层, and which 激活函数s to use for a given problem.

## Exercises

1. Build a 2-4-2-1 network (two hidden 层s) and run the forward pass on XOR data with random 权重s. Print the intermediate hidden 层 outputs to see how the representation transforms at each 层.

2. Change the hidden 层 size in the circle classifier from 8 to 2, then to 32. Run the forward pass with random 权重s each time. Does the number of hidden neurons change the output range or distribution? Why?

3. Implement a `count_parameters` method on the Network class that returns the total number of trainable 权重s and 偏置es. Test it on a 784-256-128-10 network (the classic MNIST architecture). How many parameters does it have?

4. Build a forward pass for a 3-4-4-2 network. Feed it RGB color values (normalized to 0-1) and observe the two outputs. This is the architecture for a simple color classifier with two classes.

5. Replace sigmoid with a "leaky step" function: return 0.01 * z if z < 0, else 1.0. Run the forward pass on XOR with the same hand-tuned 权重s from Step 4. Does it still work? Why is the smooth sigmoid preferred over hard cutoffs?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Forward pass | "Running the model" | Pushing input through every 层 -- multiply by 权重s, add 偏置, activate -- to produce an output |
| Hidden 层 | "The middle part" | Any 层 between input and output whose values are not directly observed in the data |
| Multi-层 network | "A deep 神经网络" | 层s of neurons stacked sequentially, where each 层's output feeds the next 层's input |
| Activation function | "The nonlinearity" | A function applied after the linear transformation that introduces curves into the decision boundary |
| Sigmoid | "The S-curve" | sigma(z) = 1/(1+e^(-z)), squashes any real number to (0,1), smooth and differentiable everywhere |
| 权重 matrix | "The parameters" | A matrix W of shape (current_层_neurons, previous_层_neurons) containing learnable connection strengths |
| 偏置 vector | "The offset" | A vector added after the matrix multiply that lets neurons activate even when all inputs are zero |
| Universal approximation | "Neural nets can learn anything" | A single hidden 层 with enough neurons can approximate any continuous function -- but "enough" can mean billions |
| Linear transformation | "The matrix multiply step" | z = W * x + b, the computation before activation, which maps inputs to a new space |
| Decision boundary | "Where the classifier switches" | The surface in input space where the network output crosses the classification threshold |

## Further Reading

- Michael Nielsen, "神经网络s and 深度学习", Chapter 1-2 (http://neuralnetworksanddeeplearning.com/) -- the clearest free explanation of forward passes and network structure, with interactive visualizations
- Cybenko, "Approximation by Superpositions of a Sigmoidal Function" (1989) -- the original universal approximation theorem paper, surprisingly readable
- 3Blue1Brown, "But what is a 神经网络?" (https://www.youtube.com/watch?v=aircAruvnKk) -- 20-minute visual walkthrough of 层s, 权重s, and forward passes that builds the right mental model
- Goodfellow, Bengio, Courville, "深度学习", Chapter 6 (https://www.deeplearningbook.org/) -- the standard reference for multi-层 networks, free online
