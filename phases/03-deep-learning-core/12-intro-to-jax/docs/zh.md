# Introduction 到 JAX

> PyTorch mutates 张量. TensorFlow builds graphs. JAX compiles pure 函数. That last one changes how you think about deep learning.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Learning Objectives

- Write pure-函数 神经网络 代码 using JAX's functional API (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Explain key design difference between PyTorch's eager mutation 和 JAX's functional compilation 模型
- Apply jit compilation 和 vmap vectorization 到 accelerate 训练 loops compared 到 naive Python
- Train simple network 在 JAX 和 contrast explicit state management 使用 PyTorch's object-oriented approach

## Problem

你知道 how 到 build 神经网络 在 PyTorch. You define `nn.Module`, call `.backward()`, step 优化器. It works. Millions 的 people use it.

But PyTorch has constraint baked into its DNA: it traces operations eagerly, one 在 time, 在 Python. Every `张量 + 张量` 是 separate kernel launch. Every 训练 step re-interprets same Python 代码. This works fine until you need 到 train 540-billion-参数 模型 across 2,048 TPUs. Then overhead kills you.

Google DeepMind trains Gemini 在 JAX. Anthropic trained Claude 在 JAX. These 是 not small operations -- they 是 largest 神经网络 训练 runs 在 Earth. They chose JAX because it treats your 训练 loop 作为 compilable program, not sequence 的 Python calls.

JAX 是 NumPy 使用 three superpowers: automatic differentiation, JIT compilation 到 XLA, 和 automatic vectorization. You write 函数 processes one example. JAX gives you 函数 processes 批次, computes gradients, compiles 到 machine 代码, 和 runs across multiple devices. All without changing original 函数.

## Concept

### JAX Philosophy

JAX 是 functional framework. No classes, no mutable state, no `.backward()` method. Instead:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class 使用 state | Pure 函数: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `为了 x 在 批次:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `模型.参数()` | Immutable pytree 的 arrays |

这是 not style preference. 它是 compiler constraint. JIT compilation requires pure 函数 -- same 输入 always produce same 输出, no side effects. That restriction 是 what makes 100x speedups possible.

### jax.numpy: Familiar Surface

JAX reimplements NumPy API 在 accelerators:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Same 函数 names. Same broadcasting rules. Same slicing semantics. But arrays live 在 GPU/TPU, 和 every operation 是 traceable 通过 compiler.

One critical difference: JAX arrays 是 immutable. No `[0] = 5`. Instead: ` = .在[0].set(5)`. This feels awkward 为了 week, then it clicks -- immutability 是 what makes transformations like `grad`, `jit`, 和 `vmap` composable.

### jax.grad: Functional Autodiff

PyTorch attaches gradients 到 张量 (`.grad`). JAX attaches gradients 到 函数.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` takes 函数 和 returns new 函数 computes gradient. No `.backward()` call. No computation graph stored 在 张量. gradient 是 just another 函数 you can call, compose, 或 JIT-compile.

This composes arbitrarily:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Second derivatives. Third derivatives. Jacobians. Hessians. All 通过 composing `grad`. PyTorch can do 这个 too (`torch.autograd.functional.hessian`), but it 是 bolted 在. In JAX, it 是 foundation.

constraint: `grad` only works 在 pure 函数. No print statements inside (they run during tracing, not execution). No mutation 的 external state. No random number generation without explicit key management.

### jit: Compile 到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

On first call, JAX traces 函数 -- it records which operations happen, without executing them. Then it hands trace 到 XLA (Accelerated 线性代数), Google's compiler 为了 TPUs 和 GPUs. XLA fuses operations, eliminates redundant memory copies, 和 generates optimized machine 代码.

Subsequent calls skip Python entirely. compiled 代码 runs 在 accelerator 在 C++ speed.

When JIT helps:
- 训练 steps (same computation repeated thousands 的 times)
- Inference (same 模型, different 输入)
- Any 函数 called more than once 使用 similar-shaped 输入

When JIT hurts:
- Functions 使用 Python control flow depends 在 values (`if x > 0` where x 是 traced array)
- One-shot computations (compilation overhead exceeds runtime)
- Debugging (tracing hides actual execution)

control flow restriction 是 real. `jax.lax.cond` replaces `if/else`. `jax.lax.scan` replaces `为了` loops. These 是 not optional -- they 是 price 的 compilation.

### vmap: Automatic Vectorization

You write 函数 processes one example:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` lifts it 到 process 批次:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` means: do not 批次 over `params` (shared), 批次 over axis 0 的 `x`. No manual `为了` loop. No reshaping. No 批次 dimension threading. JAX figures out 批次 dimension 和 vectorizes entire computation.

这是 not syntactic sugar. `vmap` generates fused vectorized 代码 runs 10-100x faster than Python loop. And it composes 使用 `jit` 和 `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Per-example gradients. One line. 这是 nearly impossible 在 PyTorch without hacks.

### pmap: 数据 Parallelism Across Devices

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` replicates 函数 across all available devices (GPUs/TPUs) 和 splits 批次. Inside 函数, `jax.lax.pmean` 和 `jax.lax.psum` synchronize gradients across devices.

Google trains Gemini across thousands 的 TPU v5e chips using `pmap` (和 its successor `shard_map`). programming 模型: write single-device version, wrap 使用 `pmap`, done.

### Pytrees: Universal 数据 Structure

JAX operates 在 "pytrees" -- nested combinations 的 lists, tuples, dicts, 和 arrays. Your 模型 参数 是 pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Every JAX transformation -- `grad`, `jit`, `vmap` -- knows how 到 traverse pytrees. `jax.tree.map(f, tree)` applies `f` 到 every leaf. 这是 how optimizers update all 参数 在 once:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

No `.参数()` method. No 参数 registration. tree structure 是 模型.

### Functional vs Object-Oriented

PyTorch stores state inside objects:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX uses pure 函数 使用 explicit state:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params 是 passed 在. Nothing 是 stored. Nothing 是 mutated. This makes every 函数 testable, composable, 和 compilable. It also means you manage params yourself -- 或 use library like Flax 或 Equinox.

### JAX Ecosystem

JAX gives you primitives. Libraries give you ergonomics:

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network 层 | `nn.Module` 使用 explicit state |
| **Equinox** (Patrick Kidger) | Neural network 层 | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | 训练 loop utilities |

Optax 是 standard 优化器 library. It separates gradient transformation (Adam, SGD, clipping) 从 参数 update, making it trivial 到 compose:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### When 到 Use JAX vs PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-在-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-通过-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale 训练 | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate 和 go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

honest answer: use PyTorch unless you have specific reason 到 use JAX. Those reasons 是 -- TPU access, need 为了 per-example gradients, multi-device 训练 在 massive scale, 或 working 在 Google/DeepMind/Anthropic.

### Random Numbers 在 JAX

JAX does not have global random state. Every random operation requires explicit PRNG key:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

这是 annoying 在 first. But it guarantees reproducibility across devices 和 compilations -- property PyTorch's `torch.manual_seed` cannot guarantee 在 multi-GPU settings.

## Build It

### Step 1: Setup 和 数据

我们将 train 3-层 MLP 在 MNIST using JAX 和 Optax. 784 输入, two hidden 层 的 256 和 128 神经元, 10 输出 classes.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Step 2: Initialize Parameters

No class. Just 函数 returns pytree:

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

He-initialization, done manually. Three PRNG keys split 从 one seed. Every 权重 是 immutable array 在 nested dict.

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

Pure 函数. Params 在, prediction out. No `self`, no stored state. `loss_fn` computes cross-entropy 从 scratch -- softmax, log, negative mean.

### Step 4: JIT-Compiled 训练 Step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` returns both loss value 和 gradients 在 one pass. `@jax.jit` decorator compiles both 函数 到 XLA. After first call, each 训练 step runs without touching Python.

### Step 5: 训练 Loop

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

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

10 轮次. ~97% test 准确率. first 轮次 是 slow (JIT compilation). Epochs 2-10 是 fast.

Notice what 是 missing: no `.zero_grad()`, no `.backward()`, no `.step()`. entire update 是 one composed 函数 call. Gradients 是 computed, transformed 通过 Adam, 和 applied 到 参数 -- all inside `train_step`.

## Use It

### Flax: Google Standard

Flax 是 most common JAX 神经网络 library. It adds `nn.Module` back, but 使用 explicit state management:

```python
import flax.linen as nn

class MLP(nn.Module):
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

Same structure 作为 PyTorch, but `params` 是 separate 从 模型. `模型.init()` creates params. `模型.apply(params, x)` runs forward pass. 模型 object has no state.

### Equinox: Pythonic Alternative

Equinox (通过 Patrick Kidger) represents 模型 作为 pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型 itself 是 pytree. No `.apply()` needed. Parameters 是 just 模型's leaves. 这是 closer 到 how JAX thinks.

### Optax: Composable Optimizers

Optax decouples gradient transformation 从 update:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping, 学习率 warmup, 权重 decay -- all composed 作为 chain 的 transforms. Each transform sees gradients, modifies them, 和 passes them 到 next. No monolithic 优化器 class.

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

- First JIT call 是 slow (compilation). Warm up before benchmarking.
- Avoid Python loops over JAX arrays inside JIT. Use `jax.lax.scan` 或 `jax.lax.fori_loop`.
- `jax.debug.print()` works inside JIT. Regular `print()` does not.
- Profile 使用 `jax.profiler` 或 TensorBoard. XLA compilation can hide bottlenecks.
- JAX pre-allocates 75% 的 GPU memory 通过 default. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` 到 disable.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `输出/prompt-jax-优化器.md` -- prompt 为了 choosing right JAX 优化器 configuration
- `输出/skill-jax-patterns.md` -- skill covering functional patterns 在 JAX

## Exercises

1. Add dropout 到 MLP. In JAX, dropout requires PRNG key -- thread key through forward pass 和 split it 为了 each dropout 层. Compare test 准确率 使用 和 without.

2. Use `jax.vmap` 到 compute per-example gradients 为了 批次 的 32 MNIST images. Compute gradient norm 为了 each example. Which examples have largest gradients, 和 why?

3. Replace manual forward 函数 使用 generic `mlp_forward(params, x)` works 为了 any number 的 层. Use `jax.tree.leaves` 到 determine depth automatically.

4. Benchmark 训练 step 使用 和 without `@jax.jit`. Time 100 steps 的 each. How large 是 speedup 在 your hardware? What 是 compilation overhead 在 first call?

5. Implement gradient clipping 通过 composing `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Train 使用 和 without clipping. Plot gradient norm over 训练 到 see effect.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | " thing makes JAX fast" | Accelerated 线性代数 -- compiler fuses operations 和 generates optimized GPU/TPU kernels 从 computation graph |
| JIT | "Just-在-time compilation" | JAX traces 函数 在 first call, compiles 到 XLA, then runs compiled version 在 subsequent calls |
| Pure 函数 | "No side effects" | 函数 where 输出 depends only 在 输入 -- no global state, no mutation, no randomness without explicit keys |
| vmap | "Auto-batching" | Transforms 函数 processes one example into one processes 批次, without rewriting |
| pmap | "Auto-parallelism" | Replicates 函数 across multiple devices 和 splits 输入 批次 |
| Pytree | "Nested dict 的 arrays" | Any nested structure 的 lists, tuples, dicts, 和 arrays JAX can traverse 和 transform |
| Tracing | "Recording computation" | JAX executes 函数 使用 abstract values 到 build computation graph, without computing real results |
| Functional autodiff | "grad 的 函数" | Computing derivatives 通过 transforming 函数, not 通过 attaching gradient storage 到 张量 |
| Optax | "JAX's 优化器 library" | composable library 的 gradient transformations -- Adam, SGD, clipping, scheduling -- chain together |
| Flax | "JAX's nn.Module" | Google's 神经网络 library 为了 JAX, adding 层 abstractions while keeping state explicit |

## Further Reading

- JAX documentation: https://jax.readthedocs.io/ -- official docs, 使用 excellent tutorials 在 grad, jit, 和 vmap
- "JAX: composable transformations 的 Python+NumPy programs" (Bradbury et al., 2018) -- original paper explaining design philosophy
- Flax documentation: https://flax.readthedocs.io/ -- Google's 神经网络 library 为了 JAX
- Patrick Kidger, "Equinox: 神经网络 在 JAX via callable PyTrees 和 filtered transformations" (2021) -- Pythonic alternative 到 Flax
- DeepMind, "Optax: composable gradient transformation 和 optimisation" -- standard 优化器 library
- "You Don't Know JAX" (Colin Raffel, 2020) -- practical guide 到 JAX gotchas 和 patterns, 从 one 的 T5 authors
