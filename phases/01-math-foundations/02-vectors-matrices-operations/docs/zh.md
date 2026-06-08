# Vectors, Matrices & Operations

> Every 神经网络 是 just 矩阵 multiplication 使用 extra steps.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (线性代数 Intuition)
**Time:** ~60 minutes

## Learning Objectives

- Build 矩阵 class 使用 element-wise operations, 矩阵 multiplication, transpose, determinant, 和 inverse
- Distinguish element-wise multiplication 从 矩阵 multiplication 和 explain when each applies
- Implement single dense 神经网络 层 (`relu(W @ x + b)`) using only 从-scratch 矩阵 class
- Explain broadcasting rules 和 how 偏置 addition works 在 神经网络 frameworks

## Problem

You want 到 build 神经网络. You read 代码 和 see 这个:

```
output = activation(weights @ input + bias)
```

That `@` 是 矩阵 multiplication. `权重` 是 矩阵. `输入` 是 向量. If you do not know what 那些 operations do, 这个 line 是 magic. If you do know, it 是 entire forward pass 的 层 在 three operations.

Every image your 模型 processes 是 矩阵 的 pixel values. Every word embedding 是 向量. Every 层 的 every 神经网络 是 矩阵 transformation. You cannot build AI systems without being fluent 在 矩阵 operations same way you cannot write 代码 without understanding variables.

This lesson builds fluency 从 scratch.

## Concept

### Vectors: ordered lists 的 numbers

向量 是 list 的 numbers 使用 direction 和 magnitude. In AI, 向量 represent 数据 points, 特征, 或 参数.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D 向量 `[3, 4]` points 到 coordinates (3, 4) 在 plane. Its length (magnitude) 是 5 ( 3-4-5 triangle).

### Matrices: grids 的 numbers

矩阵 是 2D grid. Rows 和 columns. m x n 矩阵 has m rows 和 n columns.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

In 神经网络, 权重 矩阵 transform 输入 向量 into 输出 向量. 层 使用 784 输入 和 128 输出 uses 128x784 权重 矩阵.

### Why shapes matter

矩阵 multiplication has strict rule: `(m x n) @ (n x p) = (m x p)`. inner dimensions must match.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

If you get shape mismatch error 在 PyTorch, 这个 是 why.

### operations map

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding 偏置 到 输出 |
| Scalar multiply | Scale every element | Learning rate * gradients |
| 矩阵 multiply | Transform 向量 | 层 forward pass |
| Transpose | Flip rows 和 columns | 反向传播 |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo transformation | Solving linear systems |
| Identity | Do-nothing 矩阵 | Initialization, residual connections |

### Element-wise vs 矩阵 multiplication

This distinction trips up beginners constantly.

Element-wise: multiply matching positions. Both 矩阵 must be same shape.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵 multiplication: dot products 的 rows 和 columns. Inner dimensions must match.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Different operations, different results, different rules.

### Broadcasting

When you add 偏置 向量 到 矩阵 的 输出, shapes do not match. Broadcasting stretches smaller array 到 fit.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Every modern framework does 这个 automatically. Understanding it prevents confusion when shapes seem wrong but 代码 runs.

## Build It

### Step 1: 向量 class

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Step 2: 矩阵 class 使用 core operations

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Step 3: See it work

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4: Connect 到 神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这是 single dense 层: `输出 = relu(W @ x + b)`. Every dense 层 在 every 神经网络 does exactly 这个.

## Use It

NumPy does everything above 在 fewer lines 和 orders 的 magnitude faster.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

`@` operator 在 Python calls `__matmul__`. NumPy implements it 使用 optimized BLAS routines written 在 C 和 Fortran. Same math, 100x faster.

Broadcasting 在 NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy automatically broadcasts 1D 偏置 across both rows. 这是 how 偏置 addition works 在 every 神经网络 framework.

## Ship It

This lesson produces prompt 为了 teaching 矩阵 operations through geometric intuition. See `输出/prompt-矩阵-operations.md`.

矩阵 class built here 是 foundation 为了 mini 神经网络 framework we build 在 Phase 3, Lesson 10.

## Exercises

1. **Verify inverse.** Multiply ` @ .inverse_2x2()` 和 confirm you get identity 矩阵. Try it 使用 three different 2x2 矩阵. What happens when determinant 是 zero?

2. **Implement 3x3 inverse.** Extend 矩阵 class 到 compute inverses 为了 3x3 矩阵 using adjugate method. Test it against NumPy's `np.linalg.inv`.

3. **Build two-层 network.** Using only your 矩阵 class (no NumPy), create two-层 神经网络: 输入 (3) -> hidden (4) -> 输出 (2). Initialize random 权重, run forward pass, 和 verify all shapes 是 correct.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| 向量 | " arrow" | ordered list 的 numbers. In AI: point 在 high-dimensional space. |
| 矩阵 | " table 的 numbers" | linear transformation. It maps 向量 从 one space 到 another. |
| 矩阵 multiply | "Just multiply numbers" | Dot products between every row 的 first 矩阵 和 every column 的 second. Order matters. |
| Transpose | "Flip it" | Swap rows 和 columns. Turns m x n 矩阵 into n x m. Critical 在 反向传播. |
| Determinant | "Some number 从 矩阵" | Measures how much 矩阵 scales area (2D) 或 volume (3D). Zero means transformation crushes dimension. |
| Inverse | "Undo 矩阵" | 矩阵 reverses transformation. Only exists when determinant 是 not zero. |
| Identity 矩阵 | " boring 矩阵" | 矩阵 equivalent 的 multiplying 通过 1. Used 在 residual connections (ResNets). |
| Broadcasting | "Magic shape fixing" | Stretching smaller array 到 match larger one 通过 repeating along missing dimensions. |
| Element-wise | "Regular multiplication" | Multiply matching positions. Both arrays must have same shape (或 be broadcastable). |

## Further Reading

- [3Blue1Brown: Essence 的 线性代数](https://www.3blue1brown.com/topics/linear-algebra) - visual intuition 为了 every operation covered here
- [NumPy documentation 在 broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - exact rules NumPy follows
- [Stanford CS229 线性代数 Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - concise reference 为了 ML-specific 线性代数
