# JAX入门

> PyTorch mutates 张量s. 张量Flow builds graphs. JAX compiles pure functions. That last one changes how you think about deep learning.

**类型:** 实现
**语言:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## 学习目标

- Write pure-function 神经网络 code using JAX's functional API (jax.numpy, jax.grad, jax.jit, jax.向量化映射)
- Explain the key design difference between PyTorch's eager mutation and JAX's functional compilation model
- Apply jit compilation and 向量化映射 向量化 to accelerate training loops compared to naive Python
- Train a simple network in JAX and contrast the explicit state management with PyTorch's object-oriented approach

## The Problem

You know how to build 神经网络s in PyTorch. You define an `nn.模块`, call `.backward()`, step the 优化器. It works. Millions of people use it.

But PyTorch has a constraint baked into its DNA: it traces operations eagerly, one at a time, in Python. Every `张量 + 张量` is a separate kernel launch. Every training step re-interprets the same Python code. This works fine until you need to train a 540-billion-parameter model across 2,048 TPUs. Then the overhead kills you.

Google DeepMind trains Gemini on JAX. Anthropic trained Claude on JAX. These are not small operations -- they are the largest 神经网络 training runs on Earth. They chose JAX because it treats your training loop as a compilable program, not a sequence of Python calls.

JAX is NumPy with three superpowers: 自动微分, JIT编译 compilation to XLA, and automatic 向量化. You write a function that processes one example. JAX gives you a function that processes a batch, computes gradients, compiles to machine code, and runs across multiple devices. All without changing the original function.

## The Concept

### The JAX Philosophy

JAX is a functional 框架. No classes, no mutable state, no `.backward()` method. Instead:

| PyTorch | JAX |
|---------|-----|
| `nn.模块` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT编译 compilation via XLA |
| `for x in batch:` manual loop | `jax.向量化映射(f)` auto-向量化 |
| `DataParallel` / `FSDP` | `jax.并行映射(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

This is not a style preference. It is a compiler constraint. JIT编译 compilation requires pure functions -- same inputs always produce same outputs, no side effects. That restriction is what makes 100x speedups possible.

### jax.numpy: The Familiar Surface

JAX reimplements the NumPy API on accelerators:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Same function names. Same broadcasting rules. Same slicing semantics. But the arrays live on GPU/TPU, and every operation is traceable by the compiler.

One critical difference: JAX arrays are immutable. No `a[0] = 5`. Instead: `a = a.at[0].set(5)`. This feels awkward for a week, then it clicks -- immutability is what makes transformations like `grad`, `jit`, and `向量化映射` composable.

### jax.grad: Functional Autodiff

PyTorch attaches gradients to 张量s (`.grad`). JAX attaches gradients to functions.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` takes a function and returns a new function that computes the gradient. No `.backward()` call. No computation graph stored on 张量s. The gradient is just another function you can call, compose, or JIT编译-compile.

This composes arbitrarily:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Second derivatives. Third derivatives. Jacobians. Hessians. All by composing `grad`. PyTorch can do this too (`torch.自动微分.functional.hessian`), but it is bolted on. In JAX, it is the foundation.

The constraint: `grad` only works on pure functions. No print statements inside (they run during tracing, not execution). No mutation of external state. No random number generation without explicit key management.

### jit: Compile to XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

On the first call, JAX traces the function -- it records which operations happen, without executing them. Then it hands that trace to XLA (Accelerated Linear Algebra), Google's compiler for TPUs and GPUs. XLA fuses operations, eliminates redundant memory copies, and generates optimized machine code.

Subsequent calls skip Python entirely. The compiled code runs on the accelerator at C++ speed.

When JIT编译 helps:
- Training steps (same computation repeated thousands of times)
- Inference (same model, different inputs)
- Any function called more than once with similar-shaped inputs

When JIT编译 hurts:
- Functions with Python control flow that depends on values (`if x > 0` where x is a traced array)
- One-shot computations (compilation overhead exceeds runtime)
- Debugging (tracing hides the actual execution)

The control flow restriction is real. `jax.lax.cond` replaces `if/else`. `jax.lax.scan` replaces `for` loops. These are not optional -- they are the price of compilation.

### 向量化映射: Automatic 向量化

You write a function that processes one example:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`向量化映射` lifts it to process a batch:

```python
batch_predict = jax.向量化映射(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` means: do not batch over `params` (shared), batch over axis 0 of `x`. No manual `for` loop. No reshaping. No batch dimension threading. JAX figures out the batch dimension and vectorizes the entire computation.

This is not syntactic sugar. `向量化映射` generates fused vectorized code that runs 10-100x faster than a Python loop. And it composes with `jit` and `grad`:

```python
per_example_grads = jax.向量化映射(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Per-example gradients. One line. This is nearly impossible in PyTorch without hacks.

### 并行映射: Data Parallelism Across Devices

```python
parallel_step = jax.并行映射(train_step, axis_name='devices')
```

`并行映射` replicates the function across all available devices (GPUs/TPUs) and splits the batch. Inside the function, `jax.lax.pmean` and `jax.lax.psum` synchronize gradients across devices.

Google trains Gemini across thousands of TPU v5e chips using `并行映射` (and its successor `shard_map`). The programming model: write the single-device version, wrap with `并行映射`, done.

### Pytrees: The Universal Data Structure

JAX operates on "pytrees" -- nested combinations of lists, tuples, dicts, and arrays. Your model parameters are a pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Every JAX transformation -- `grad`, `jit`, `向量化映射` -- knows how to traverse pytrees. `jax.tree.map(f, tree)` applies `f` to every leaf. This is how 优化器s update all parameters at once:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

No `.parameters()` method. No parameter registration. The tree structure is the model.

### Functional vs Object-Oriented

PyTorch stores state inside objects:

```python
class Model(nn.模块):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX uses pure functions with explicit state:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

The params are passed in. Nothing is stored. Nothing is mutated. This makes every function testable, composable, and compilable. It also means you manage the params yourself -- or use a library like Flax or Equinox.

### The JAX Ecosystem

JAX gives you primitives. Libraries give you ergonomics:

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.模块` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | 优化器s + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax is the standard 优化器 library. It separates the gradient transformation (Adam, SGD, clipping) from the parameter update, making it trivial to compose:

```python
优化器 = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### When to Use JAX vs PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, 并行映射, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | 张量Flow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

The honest answer: use PyTorch unless you have a specific reason to use JAX. Those reasons are -- TPU access, need for per-example gradients, multi-device training at massive scale, or working at Google/DeepMind/Anthropic.

### Random Numbers in JAX

JAX does not have a global random state. Every random operation requires an explicit PRNG key:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

This is annoying at first. But it guarantees reproducibility across devices and compilations -- a property that PyTorch's `torch.manual_seed` cannot guarantee in multi-GPU settings.

## Build It

### Step 1: Setup and Data

We will train a 3-layer MLP on MNIST using JAX and Optax. 784 inputs, two hidden layers of 256 and 128 neurons, 10 output classes.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.数据集s import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Step 2: Initialize Parameters

No class. Just a function that returns a pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He-initialization, done manually. Three PRNG keys split from one seed. Every weight is an immutable array in a nested dict.

### Step 3: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Pure functions. Params in, prediction out. No `self`, no stored state. `loss_fn` computes cross-entropy from scratch -- softmax, log, negative mean.

### Step 4: JIT编译-Compiled Training Step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = 优化器.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` returns both the loss value and the gradients in one pass. The `@jax.jit` decorator compiles both functions to XLA. After the first call, each training step runs without touching Python.

### Step 5: Training Loop

```python
优化器 = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = 优化器.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epochs. ~97% test accuracy. The first epoch is slow (JIT编译 compilation). Epochs 2-10 are fast.

Notice what is missing: no `.zero_grad()`, no `.backward()`, no `.step()`. The entire update is one composed function call. Gradients are computed, transformed by Adam, and applied to parameters -- all inside `train_step`.

## Use It

### Flax: The Google Standard

Flax is the most common JAX 神经网络 library. It adds `nn.模块` back, but with explicit state management:

```python
import flax.linen as nn

class MLP(nn.模块):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Same structure as PyTorch, but `params` is separate from the model. `model.init()` creates params. `model.apply(params, x)` runs the forward pass. The model object has no state.

### Equinox: The Pythonic Alternative

Equinox (by Patrick Kidger) represents models as pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

The model itself is a pytree. No `.apply()` needed. Parameters are just the model's leaves. This is closer to how JAX thinks.

### Optax: Composable 优化器s

Optax decouples the gradient transformation from the update:

```python
schedule = optax.预热_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    预热_steps=1000, decay_steps=50000
)

优化器 = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping, 学习率 预热, 权重衰减 -- all composed as a chain of transforms. Each transform sees the gradients, modifies them, and passes them to the next. No monolithic 优化器 class.

## Ship It

**Installation:**

```bash
pip install jax jaxlib optax flax
```

For GPU support:

```bash
pip install jax[cuda12]
```

For TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- First JIT编译 call is slow (compilation). Warm up before benchmarking.
- Avoid Python loops over JAX arrays inside JIT编译. Use `jax.lax.scan` or `jax.lax.fori_loop`.
- `jax.debug.print()` works inside JIT编译. Regular `print()` does not.
- Profile with `jax.profiler` or 张量Board. XLA compilation can hide bottlenecks.
- JAX pre-allocates 75% of GPU memory by default. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to disable.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-优化器.md` -- a prompt for choosing the right JAX 优化器 configuration
- `outputs/skill-jax-patterns.md` -- a skill covering functional patterns in JAX

## Exercises

1. Add dropout to the MLP. In JAX, dropout requires a PRNG key -- thread a key through the forward pass and split it for each dropout layer. Compare test accuracy with and without.

2. Use `jax.向量化映射` to compute per-example gradients for a batch of 32 MNIST images. Compute the gradient norm for each example. Which examples have the largest gradients, and why?

3. Replace the manual forward function with a generic `mlp_forward(params, x)` that works for any number of layers. Use `jax.tree.leaves` to determine the depth automatically.

4. Benchmark the training step with and without `@jax.jit`. Time 100 steps of each. How large is the speedup on your hardware? What is the compilation overhead on the first call?

5. Implement gradient clipping by composing `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Train with and without clipping. Plot the gradient norm over training to see the effect.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | "The thing that makes JAX fast" | Accelerated Linear Algebra -- a compiler that fuses operations and generates optimized GPU/TPU kernels from a computation graph |
| JIT编译 | "Just-in-time compilation" | JAX traces the function on first call, compiles to XLA, then runs the compiled version on subsequent calls |
| Pure function | "No side effects" | A function where the output depends only on inputs -- no global state, no mutation, no randomness without explicit keys |
| 向量化映射 | "Auto-batching" | Transforms a function that processes one example into one that processes a batch, without rewriting |
| 并行映射 | "Auto-parallelism" | Replicates a function across multiple devices and splits the input batch |
| Pytree | "Nested dict of arrays" | Any nested structure of lists, tuples, dicts, and arrays that JAX can traverse and transform |
| Tracing | "Recording the computation" | JAX executes the function with abstract values to build a computation graph, without computing real results |
| Functional autodiff | "grad of a function" | Computing derivatives by transforming functions, not by attaching gradient storage to 张量s |
| Optax | "JAX's 优化器 library" | A composable library of gradient transformations -- Adam, SGD, clipping, scheduling -- that chain together |
| Flax | "JAX's nn.模块" | Google's 神经网络 library for JAX, adding layer abstractions while keeping state explicit |

## Further Reading

- JAX documentation: https://jax.readthedocs.io/ -- the official docs, with excellent tutorials on grad, jit, and 向量化映射
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- the original paper explaining the design philosophy
- Flax documentation: https://flax.readthedocs.io/ -- Google's 神经网络 library for JAX
- Patrick Kidger, "Equinox: 神经网络s in JAX via callable PyTrees and filtered transformations" (2021) -- the Pythonic alternative to Flax
- DeepMind, "Optax: composable gradient transformation and optimisation" -- the standard 优化器 library
- "You Don't Know JAX" (Colin Raffel, 2020) -- a practical guide to JAX gotchas and patterns, from one of the T5 authors
