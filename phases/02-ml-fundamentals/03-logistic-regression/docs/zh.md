# Logistic 回归

> Logistic 回归 bends straight line into S-curve 到 answer yes-或-no questions 使用 probabilities.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 Lesson 1-2 (What Is ML, Linear 回归)
**Time:** ~90 minutes

## Learning Objectives

- Implement logistic 回归 从 scratch using sigmoid 函数 和 binary cross-entropy loss
- Compute 和 interpret 精确率, 召回率, F1 score, 和 confusion 矩阵 为了 binary 分类
- Explain why MSE fails 为了 分类 和 why binary cross-entropy produces convex cost surface
- Build softmax 回归 模型 为了 multi-class 分类 和 evaluate threshold tuning tradeoffs

## Problem

You want 到 predict whether tumor 是 malignant 或 benign given its size. You try linear 回归. It 输出 numbers like 0.3 或 1.7 或 -0.5. What do 那些 mean? Is 1.7 "very malignant"? Is -0.5 "very benign"? Linear 回归 输出 unbounded numbers. 分类 needs bounded probabilities between 0 和 1, 和 clear decision: yes 或 no.

Logistic 回归 solves 这个. It takes same linear combination (wx + b) 和 passes it through sigmoid 函数, which squashes any number into range (0, 1). 输出 是 概率. You set threshold (usually 0.5) 和 make decision.

这是 one 的 most widely used 算法 在 practice. Despite its name, logistic 回归 是 分类 算法, not 回归 算法. name comes 从 logistic (sigmoid) 函数 it uses.

## Concept

### Why Linear 回归 Fails 为了 分类

Imagine predicting pass/fail (1/0) based 在 study hours. Linear 回归 fits line through 数据:

```
hours:  1   2   3   4   5   6   7   8   9   10
actual: 0   0   0   0   1   1   1   1   1   1
```

linear fit might produce predictions like -0.2 在 hour 1 和 1.3 在 hour 10. These values 是 not probabilities. They go below 0 和 above 1. Worse, single outlier (someone who studied 50 hours) would drag entire line, changing predictions 为了 everyone.

分类 needs 函数 :
- Outputs values between 0 和 1 (probabilities)
- Creates sharp transition ( decision boundary)
- Is not distorted 通过 outliers far 从 boundary

### Sigmoid 函数

sigmoid 函数 does exactly 这个:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Properties:
- When z 是 large 和 positive, sigmoid(z) approaches 1
- When z 是 large 和 negative, sigmoid(z) approaches 0
- When z = 0, sigmoid(z) = 0.5
- 输出 是 always between 0 和 1
- 函数 是 smooth 和 differentiable everywhere

derivative has convenient form: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)). This makes gradient computation efficient.

### Logistic 回归 = Linear 模型 + Sigmoid

模型 computes z = wx + b (same 作为 linear 回归), then applies sigmoid:

```mermaid
flowchart LR
    X[Input features x] --> L["Linear: z = wx + b"]
    L --> S["Sigmoid: p = 1/(1+e^-z)"]
    S --> D{"p >= 0.5?"}
    D -->|Yes| P[Predict 1]
    D -->|No| N[Predict 0]
```

输出 p 是 interpreted 作为 P(y=1 | x), 概率 输入 belongs 到 class 1. decision boundary 是 where wx + b = 0, which makes sigmoid 输出 exactly 0.5.

### Binary Cross-Entropy Loss

You cannot use MSE 为了 logistic 回归. MSE 使用 sigmoid creates non-convex cost surface 使用 many local minima. Instead, use binary cross-entropy (log loss):

```
Loss = -(1/n) * sum(y * log(p) + (1-y) * log(1-p))
```

Why 这个 works:
- When y=1 和 p 是 close 到 1: log(1) = 0, so loss 是 near 0 (correct, low cost)
- When y=1 和 p 是 close 到 0: log(0) approaches negative infinity, so loss 是 huge (wrong, high cost)
- When y=0 和 p 是 close 到 0: log(1) = 0, so loss 是 near 0 (correct, low cost)
- When y=0 和 p 是 close 到 1: log(0) approaches negative infinity, so loss 是 huge (wrong, high cost)

This 损失函数 是 convex 为了 logistic 回归, guaranteeing single global minimum.

### 梯度下降 为了 Logistic 回归

gradients 为了 binary cross-entropy 使用 sigmoid have clean form:

```
dL/dw = (1/n) * sum((p - y) * x)
dL/db = (1/n) * sum(p - y)
```

These look identical 到 linear 回归 gradients. difference 是 p = sigmoid(wx + b) instead 的 p = wx + b. sigmoid introduces nonlinearity, but gradient update rule stays same.

```mermaid
flowchart TD
    A[Initialize w=0, b=0] --> B[Forward pass: z = wx+b, p = sigmoid z]
    B --> C[Compute loss: binary cross-entropy]
    C --> D["Compute gradients: dw = (1/n) * sum((p-y)*x)"]
    D --> E[Update: w = w - lr*dw, b = b - lr*db]
    E --> F{Converged?}
    F -->|No| B
    F -->|Yes| G[Model trained]
```

### Decision Boundary

For 2D 输入 (two 特征), decision boundary 是 line where:

```
w1*x1 + w2*x2 + b = 0
```

Points 在 one side get classified 作为 1, points 在 other side 作为 0. Logistic 回归 always produces linear decision boundary. If you need curved boundary, you either add polynomial 特征 或 use nonlinear 模型.

### Multi-Class 分类 使用 Softmax

Binary logistic 回归 handles two classes. For k classes, use softmax 函数:

```
softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)
```

Each class has its own 权重 向量. 模型 computes score z_i 为了 each class, then softmax converts scores 到 probabilities sum 到 1. predicted class 是 one 使用 highest 概率.

损失函数 becomes categorical cross-entropy:

```
Loss = -(1/n) * sum(sum(y_k * log(p_k)))
```

where y_k 是 1 为了 true class 和 0 为了 all others (one-hot encoding).

### Evaluation Metrics

准确率 alone 是 not enough. For 数据集 使用 95% negative 和 5% positive, 模型 always predicts negative gets 95% 准确率 but 是 useless.

**Confusion 矩阵**:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actually Positive | True Positive (TP) | False Negative (FN) |
| Actually Negative | False Positive (FP) | True Negative (TN) |

**精确率**: Of all predicted positives, how many 是 actually positive?
```
Precision = TP / (TP + FP)
```

**召回率** (Sensitivity): Of all actual positives, how many did we catch?
```
Recall = TP / (TP + FN)
```

**F1 Score**: Harmonic mean 的 精确率 和 召回率. Balances both metrics.
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

When 到 prioritize:
- **精确率**: when false positives 是 costly (spam filter, you do not want 到 block legitimate email)
- **召回率**: when false negatives 是 costly (cancer screening, you do not want 到 miss tumor)
- **F1**: when you need single balanced metric

## Build It

### Step 1: Sigmoid 函数 和 数据 generation

```python
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")
```

### Step 2: Logistic 回归 从 scratch

```python
class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("\n=== Training Logistic Regression ===")
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
print(f"Bias: {model.bias:.4f}")
```

### Step 3: Confusion 矩阵 和 metrics 从 scratch

```python
class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


y_pred_test = [model.predict(x) for x in X_test]
print("\n=== Classification Report (Test Set) ===")
metrics = ClassificationMetrics(y_test, y_pred_test)
metrics.print_report()
```

### Step 4: Decision boundary analysis

```python
print("\n=== Decision Boundary ===")
w1, w2 = model.weights
b = model.bias
print(f"Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Solved for x2:     x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

print("\nSample predictions near the boundary:")
test_points = [
    [3.0, 3.0],
    [3.5, 3.5],
    [4.0, 4.0],
    [2.5, 2.5],
    [5.0, 5.0],
]
for point in test_points:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f"  [{point[0]}, {point[1]}] -> prob={prob:.4f}, class={pred}")
```

### Step 5: Multi-class 使用 softmax

```python
class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0
            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


random.seed(42)
X_3class = []
y_3class = []

centers = [(1, 1), (5, 1), (3, 5)]
for label, (cx, cy) in enumerate(centers):
    for _ in range(50):
        X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
        y_3class.append(label)

combined = list(zip(X_3class, y_3class))
random.shuffle(combined)
X_3class, y_3class = zip(*combined)
X_3class = list(X_3class)
y_3class = list(y_3class)

split_3 = int(0.8 * len(X_3class))
X_train_3 = X_3class[:split_3]
y_train_3 = y_3class[:split_3]
X_test_3 = X_3class[split_3:]
y_test_3 = y_3class[split_3:]

print("\n=== Multi-class Softmax Regression (3 classes) ===")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

print("\nSample predictions:")
for i in range(5):
    probs = softmax_model.predict_proba(X_test_3[i])
    pred = softmax_model.predict(X_test_3[i])
    print(f"  True: {y_test_3[i]}, Predicted: {pred}, Probs: [{', '.join(f'{p:.3f}' for p in probs)}]")
```

### Step 6: Threshold tuning

```python
print("\n=== Threshold Tuning ===")
print("Default threshold: 0.5. Adjusting the threshold trades precision for recall.\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in thresholds:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")
```

## Use It

Now same thing 使用 scikit-learn.

```python
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_0 = np.random.randn(100, 2) + [2, 2]
X_1 = np.random.randn(100, 2) + [5, 5]
X_sk = np.vstack([X_0, X_1])
y_sk = np.array([0] * 100 + [1] * 100)

X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = SklearnLR()
lr.fit(X_tr_sc, y_tr)
y_pred = lr.predict(X_te_sc)

print("=== Scikit-learn Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_te, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")
```

Your 从-scratch implementation produces same decision boundary 和 metrics. Scikit-learn adds solver options (liblinear, lbfgs, saga), automatic 正则化, multi-class strategies (one-vs-rest, multinomial), 和 numerical stability optimizations.

## Ship It

This lesson produces:
- `代码/logistic_regression.py` - logistic 回归 从 scratch 使用 metrics

## Exercises

1. Generate 数据集 是 NOT linearly separable (e.g., two concentric circles). Train logistic 回归 和 observe its failure. Then add polynomial 特征 (x1^2, x2^2, x1*x2) 和 train again. Show 准确率 improves.
2. Implement multi-class confusion 矩阵 为了 3-class softmax 模型. Compute per-class 精确率 和 召回率. Which class 是 hardest 到 classify?
3. Build ROC curve 从 scratch. For 100 threshold values 从 0 到 1, compute true positive rate 和 false positive rate. Calculate AUC (area under curve) using trapezoidal rule.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Logistic 回归 | "回归 为了 分类" | linear 模型 followed 通过 sigmoid 函数 输出 class probabilities |
| Sigmoid 函数 | " S-curve" | 函数 1/(1+e^(-z)) maps any real number 到 range (0, 1) |
| Binary cross-entropy | "Log loss" | 损失函数 -[y*log(p) + (1-y)*log(1-p)] penalizes confident wrong predictions severely |
| Decision boundary | " dividing line" | surface where 模型's 输出 概率 equals 0.5, separating predicted classes |
| Softmax | "Multi-class sigmoid" | 函数 converts 向量 的 scores into probabilities sum 到 1 |
| 精确率 | "How many selected 是 relevant" | TP / (TP + FP), fraction 的 positive predictions 是 actually positive |
| 召回率 | "How many relevant 是 selected" | TP / (TP + FN), fraction 的 actual positives 模型 correctly identifies |
| F1 score | "Balanced 准确率" | harmonic mean 的 精确率 和 召回率: 2*P*R / (P+R) |
| Confusion 矩阵 | " error breakdown" | table showing TP, TN, FP, FN counts 为了 each class pair |
| Threshold | " cutoff" | 概率 value above which 模型 predicts class 1 (default 0.5, tunable) |
| One-hot encoding | "Binary columns 为了 categories" | Representing class k 作为 向量 的 zeros 使用 1 在 position k |
| Categorical cross-entropy | "Multi-class log loss" | extension 的 binary cross-entropy 到 k classes using one-hot encoded labels |
