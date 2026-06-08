# Norms 和 Distances

> Your distance 函数 defines what "similar" means. Choose wrong 和 everything downstream breaks.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (线性代数 Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## Learning Objectives

- Implement L1, L2, cosine, Mahalanobis, Jaccard, 和 edit distance 函数 从 scratch
- Select appropriate distance metric 为了 given ML task 和 explain why alternatives fail
- Connect L1 和 L2 norms 到 LASSO 和 Ridge 正则化 和 their geometric constraint regions
- Demonstrate how same 数据集 produces different nearest neighbors under different metrics

## Problem

You have two 向量. Maybe they 是 word embeddings. Maybe they 是 user profiles. Maybe they 是 pixel arrays. 你需要 到 know: how close 是 they?

answer depends entirely 在 which distance 函数 you pick. Two 数据 points can be nearest neighbors under one metric 和 far apart under another. Your KNN classifier, your recommendation engine, your 向量 database, your 聚类 算法, your 损失函数 -- they all depend 在 这个 choice. Get it wrong 和 your 模型 optimizes 为了 wrong thing.

有 no universal best distance. L2 works 为了 spatial 数据. Cosine similarity dominates NLP. Jaccard handles sets. Edit distance handles strings. Mahalanobis accounts 为了 correlations. Wasserstein moves 概率 mass. Each one encodes different assumption about what "similar" means.

This lesson builds every major distance 函数 从 scratch, shows you when each one 是 right tool, 和 demonstrates how same 数据 produces completely different nearest neighbors depending 在 which metric you use.

## Concept

### Norms: measuring 向量 magnitude

norm measures "size" 的 向量. Every distance 函数 between two 向量 can be written 作为 norm 的 their difference: d(, b) = || - b||. So understanding norms 是 understanding distances.

### L1 Norm (Manhattan distance)

L1 norm sums absolute values 的 all components.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它是 called Manhattan distance because it measures how far you walk 在 city grid where you can only move along axes. No diagonals.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

When 到 use L1:
- High-dimensional sparse 数据 (text 特征, one-hot encodings)
- When you want robustness 到 outliers ( single huge difference does not dominate)
- 特征 selection problems (L1 正则化 promotes sparsity)

Connection 到 L1 正则化 (Lasso): adding ||w||_1 到 your 损失函数 penalizes sum 的 absolute 权重 values. This pushes small 权重 到 exactly zero, performing automatic 特征 selection. L1 penalty creates diamond-shaped constraint regions 在 权重 space, 和 corners 的 diamonds lie 在 axes where some 权重 是 zero.

Connection 到 loss 函数: Mean Absolute Error (MAE) 是 average L1 distance between predictions 和 targets. It penalizes all errors linearly, making it robust 到 outliers compared 到 MSE.

### L2 Norm (Euclidean distance)

L2 norm 是 straight-line distance. Square root 的 sum 的 squared components.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这是 distance you learned 在 geometry class. Pythagoras 在 n dimensions.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

When 到 use L2:
- Low-到-medium dimensional continuous 数据
- When 特征 scales 是 comparable
- Physical distances (spatial 数据, sensor readings)
- Image similarity 在 pixel level

Connection 到 L2 正则化 (Ridge): adding ||w||_2^2 到 your 损失函数 penalizes large 权重. Unlike L1, it does not push 权重 到 zero. It shrinks all 权重 toward zero proportionally. L2 penalty creates circular constraint regions, so there 是 no corners 在 axes. Weights get small but rarely exactly zero.

Connection 到 loss 函数: Mean Squared Error (MSE) 是 average 的 L2 distances squared. Squaring penalizes large errors more heavily than small ones.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp Norms: general family

L1 和 L2 是 special cases 的 Lp norm:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Different values 的 p produce different shaped "unit balls" ( set 的 all points 在 distance 1 从 origin):

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity Norm (Chebyshev distance)

As p approaches infinity, Lp norm converges 到 maximum absolute component.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

distance between two points 是 determined 通过 single dimension where they differ most. All other dimensions 是 ignored.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

When 到 use L-infinity:
- When worst-case deviation 在 any single dimension matters
- Game boards ( king 在 chess moves 在 L-infinity: one step 在 any direction costs 1)
- Manufacturing tolerances (every dimension must be within spec)

### Cosine Similarity 和 Cosine Distance

Cosine similarity measures angle between two 向量, ignoring their magnitudes.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

It ranges 从 -1 (opposite directions) 到 +1 (same direction). Perpendicular 向量 have cosine similarity 0.

Cosine distance converts it 到 distance: cosine_distance = 1 - cosine_similarity. This ranges 从 0 (identical direction) 到 2 (opposite direction).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Why cosine dominates NLP 和 embeddings: 在 text, document length should not affect similarity. document about cats 是 twice 作为 long 作为 another document about cats should still be "similar." Cosine similarity ignores magnitude (length) 和 only cares about direction. Two documents 使用 same word distribution but different lengths point 在 same direction 和 get cosine similarity 1.0.

When 到 use cosine similarity:
- Text similarity (TF-IDF 向量, word embeddings, sentence embeddings)
- Any domain where magnitude 是 noise 和 direction 是 signal
- Recommendation systems (user preference 向量)
- Embedding search (向量 databases almost always use cosine 或 dot product)

### Dot Product Similarity vs Cosine Similarity

dot product 的 two 向量 是:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Cosine similarity 是 dot product normalized 通过 both magnitudes. When both 向量 是 already unit-normalized (magnitude = 1), dot product 和 cosine similarity 是 identical.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

When they differ: dot product includes magnitude information. 向量 使用 larger magnitude gets higher dot product score. This matters 在 some retrieval systems where you want "popular" items 到 rank higher. magnitude acts 作为 implicit quality 或 importance signal.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

In practice:
- Use cosine similarity when you want pure directional similarity
- Use dot product when magnitudes carry meaningful information
- Many 向量 databases (Pinecone, Weaviate, Qdrant) let you choose between them
- If your embeddings 是 L2-normalized, choice does not matter

### Mahalanobis Distance

Euclidean distance treats all dimensions equally. But if your 特征 是 correlated 或 have different scales, L2 gives misleading results.

Mahalanobis distance accounts 为了 covariance structure 的 数据.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

where S 是 covariance 矩阵 的 数据.

Intuitively: Mahalanobis distance first decorrelates 和 normalizes 数据 (whitening), then computes L2 distance 在 transformed space. If S 是 identity 矩阵 (uncorrelated, unit variance 特征), Mahalanobis distance reduces 到 Euclidean distance.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

When 到 use Mahalanobis distance:
- Outlier detection (points 使用 large Mahalanobis distance 从 mean 是 outliers)
- 分类 when 特征 have different scales 和 correlations
- When you have enough 数据 到 estimate reliable covariance 矩阵
- Quality control 在 manufacturing (multivariate process monitoring)

### Jaccard Similarity (为了 sets)

Jaccard similarity measures overlap between two sets.

```
J(A, B) = |A intersect B| / |A union B|
```

It ranges 从 0 (no overlap) 到 1 (identical sets). Jaccard distance = 1 - Jaccard similarity.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

When 到 use Jaccard:
- Comparing sets 的 tags, categories, 或 特征
- Document similarity based 在 word presence (not frequency)
- Near-duplicate detection (MinHash approximation 的 Jaccard)
- Comparing binary 特征 向量 (presence/absence 数据)
- Evaluating segmentation 模型 (Intersection over Union = Jaccard)

### Edit Distance (Levenshtein Distance)

Edit distance counts minimum number 的 single-character operations needed 到 transform one string into another. operations 是: insert, delete, 或 substitute.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Computed using dynamic programming. Fill 矩阵 where entry (i, j) 是 edit distance between first i characters 的 string 和 first j characters 的 string B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

When 到 use edit distance:
- Spell checking 和 correction
- DNA sequence alignment (使用 weighted operations)
- Fuzzy string matching
- Deduplication 的 messy text 数据

### KL Divergence (not distance, but used like one)

KL divergence measures how one 概率 distribution differs 从 another. Covered 在 Lesson 09, but it belongs 在 这个 discussion because people use it 作为 "distance" despite it not being one.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Critical property: KL divergence 是 NOT symmetric.

```
D_KL(P || Q) != D_KL(Q || P)
```

This means it fails basic requirement 的 distance metric. It also does not satisfy triangle inequality. 它是 divergence, not distance.

Forward KL (D_KL(P || Q)) 是 "mean-seeking": Q tries 到 cover all modes 的 P.
Reverse KL (D_KL(Q || P)) 是 "mode-seeking": Q focuses 在 single mode 的 P.

When you see KL divergence:
- VAEs ( KL term 在 ELBO pushes latent distribution toward prior)
- Knowledge distillation (student tries 到 match teacher's distribution)
- RLHF ( KL penalty keeps fine-tuned 模型 close 到 base 模型)
- Policy gradient methods (constraining policy updates)

### Wasserstein Distance (Earth Mover's Distance)

Wasserstein distance measures minimum "work" needed 到 transform one 概率 distribution into another. Think 的 it 作为: if one distribution 是 pile 的 dirt 和 other 是 hole, how much dirt do you have 到 move 和 how far?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

For 1D distributions, it simplifies 到 integral 的 absolute difference 的 cumulative distribution 函数:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Why Wasserstein matters:
- 它是 true metric (symmetric, satisfies triangle inequality)
- It provides gradients even when distributions do not overlap (KL divergence goes 到 infinity)
- This property made it central 到 Wasserstein GANs (WGANs), which solved 训练 instability 的 original GANs

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

When 到 use Wasserstein:
- GAN 训练 (WGAN, WGAN-GP)
- Comparing distributions may not overlap
- Optimal transport problems
- Image retrieval (comparing color histograms)

### Why Different Tasks Need Different Distances

| Task | Best distance | Why |
|------|--------------|-----|
| Text similarity | Cosine | Magnitude 是 noise, direction 是 meaning |
| Image pixel comparison | L2 | Spatial relationships matter, 特征 是 comparable scale |
| Sparse high-dim 特征 | L1 | Robust, does not amplify rare large differences |
| Set overlap (tags, categories) | Jaccard | 数据 是 naturally set-valued, not vectorial |
| String matching | Edit distance | Operations map 到 human editing intuition |
| Outlier detection | Mahalanobis | Accounts 为了 特征 correlations 和 scales |
| Comparing distributions | KL divergence | Measures information lost 通过 using Q instead 的 P |
| GAN 训练 | Wasserstein | Provides gradients even when distributions do not overlap |
| Embeddings (向量 DB) | Cosine 或 dot product | Embeddings 是 trained 到 encode meaning 在 direction |
| Recommendation | Dot product | Magnitude can encode popularity 或 confidence |
| DNA sequences | Weighted edit distance | Substitution costs vary 通过 nucleotide pair |
| Manufacturing QC | L-infinity | Worst-case deviation 在 any dimension matters |

### Connection 到 Loss Functions

Loss 函数 是 distance 函数 applied 到 predictions vs targets.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Connection 到 正则化

正则化 adds norm penalty 在 权重 到 损失函数.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

Why L1 produces sparsity but L2 does not: picture constraint region 在 2D 权重 space. L1 是 diamond, L2 是 circle. 损失函数's contours (ellipses) 是 most likely 到 touch diamond 在 corner, where one 权重 是 zero. They touch circle 在 smooth point, where both 权重 是 nonzero.

### Nearest Neighbor Search

Every distance 函数 implies nearest neighbor search problem: given query point, find closest points 在 数据集.

Exact nearest neighbor search 是 O(n * d) per query 在 数据集 的 n points 使用 d dimensions. For large 数据集, 这个 是 too slow.

Approximate Nearest Neighbor (ANN) 算法 trade small amount 的 准确率 为了 massive speed gains:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) 是 dominant 算法 在 modern 向量 databases. It builds multi-层 graph where each 节点 connects 到 its approximate nearest neighbors. Search starts 在 top 层 (sparse, long jumps) 和 descends 到 bottom 层 (dense, short jumps).

## Build It

### Step 1: All norm 和 distance 函数

See `代码/distances.py` 为了 complete implementation. Every 函数 是 built 从 scratch using only basic Python math.

### Step 2: Same 数据, different distances, different neighbors

demo 在 `distances.py` creates 数据集, picks query point, 和 shows how nearest neighbor changes depending 在 distance metric. point 是 "closest" under L1 may not be closest under L2 或 cosine.

### Step 3: Embedding similarity search

代码 includes mock embedding similarity search finds most similar "documents" 到 query using cosine similarity vs L2 distance, showing rankings can differ.

## Use It

most common practical use: finding similar items 在 向量 database.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

When you call `模型.encode(text)` 和 then search 向量 database, 这个 是 what happens under hood. embedding 模型 maps text 到 向量. 向量 database computes cosine similarity (或 dot product) between your query 向量 和 every stored 向量, using ANN 算法 到 avoid checking all 的 them.

## Exercises

1. Compute L1, L2, 和 L-infinity distances between (1, 2, 3) 和 (4, 0, 6). Verify L-inf <= L2 <= L1 always holds 为了 any pair 的 points. Prove why 这个 ordering 是 guaranteed.

2. Create two 向量 where cosine similarity 是 high (> 0.9) but L2 distance 是 large (> 10). Explain geometrically what 是 happening. Then create two 向量 where cosine similarity 是 low (< 0.3) but L2 distance 是 small (< 0.5).

3. Implement 函数 takes 数据集 和 query point 和 returns nearest neighbor under L1, L2, cosine, 和 Mahalanobis distance. Find 数据集 where all four disagree 在 which point 是 nearest.

4. Compute Wasserstein distance between [0.5, 0.5, 0, 0] 和 [0, 0, 0.5, 0.5] 通过 hand using CDF method. Then compute it between [0.25, 0.25, 0.25, 0.25] 和 [0, 0, 0.5, 0.5]. Which 是 larger 和 why?

5. Implement MinHash 为了 approximate Jaccard similarity. Generate 100 random sets, compute exact Jaccard 为了 all pairs, 和 compare 使用 MinHash approximation using 50, 100, 和 200 hash 函数. Plot approximation error.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Norm | "Size 的 向量" | 函数 maps 向量 到 non-negative scalar, satisfying triangle inequality, absolute homogeneity, 和 zero only 为了 zero 向量 |
| L1 norm | "Manhattan distance" | Sum 的 absolute component values. Produces sparsity 在 optimization. Robust 到 outliers |
| L2 norm | "Euclidean distance" | Square root 的 sum 的 squared components. straight-line distance 在 Euclidean space |
| Lp norm | "Generalized norm" | p-th root 的 sum 的 p-th powers 的 absolute components. L1 和 L2 是 special cases |
| L-infinity norm | "Max norm" 或 "Chebyshev distance" | maximum absolute component value. limit 的 Lp 作为 p approaches infinity |
| Cosine similarity | "Angle between 向量" | Dot product normalized 通过 both magnitudes. Ranges 从 -1 到 +1. Ignores 向量 length |
| Cosine distance | "1 minus cosine similarity" | Converts cosine similarity 到 distance. Ranges 从 0 到 2 |
| Dot product | "Unnormalized cosine" | Sum 的 component-wise products. Equals cosine similarity times both magnitudes |
| Mahalanobis distance | "Correlation-aware distance" | L2 distance 在 space has been whitened (decorrelated 和 normalized) using 数据 covariance 矩阵 |
| Jaccard similarity | "Set overlap" | Size 的 intersection divided 通过 size 的 union. For sets, not 向量 |
| Edit distance | "Levenshtein distance" | Minimum insertions, deletions, 和 substitutions 到 transform one string into another |
| KL divergence | "Distance between distributions" | Not true distance (not symmetric). Measures extra bits 从 using Q 到 encode P |
| Wasserstein distance | "Earth mover's distance" | Minimum work 到 transport mass 从 one distribution 到 another. true metric |
| Approximate nearest neighbor | "ANN search" | Algorithms (HNSW, LSH, IVF) find approximately closest points much faster than exact search |
| HNSW | " 向量 DB 算法" | Hierarchical Navigable Small World graph. Multi-层 graph 为了 fast approximate nearest neighbor search |
| L1 正则化 | "Lasso" | Adding L1 norm 的 权重 到 loss. Drives 权重 到 zero (sparsity) |
| L2 正则化 | "Ridge" 或 "权重 decay" | Adding squared L2 norm 的 权重 到 loss. Shrinks 权重 toward zero without sparsity |
| Elastic Net | "L1 + L2" | Combines L1 和 L2 正则化. Handles correlated 特征 groups better than either alone |

## Further Reading

- [FAISS: Library 为了 Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta's library 为了 billion-scale ANN search
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - paper introduced Earth Mover's distance 到 GANs
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - foundational ANN 算法
- [Efficient Estimation 的 Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, where cosine similarity became default 为了 embeddings
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - practical guide 到 distance metrics 和 neighbor 算法 在 scikit-learn
