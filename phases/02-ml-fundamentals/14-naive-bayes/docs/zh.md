# Naive Bayes

> "naive" assumption 是 wrong, 和 it works anyway. That's beauty 的 it.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (分类, Bayes' theorem)
**Time:** ~75 minutes

## Learning Objectives

- Implement Multinomial Naive Bayes 从 scratch 使用 Laplace smoothing 为了 text 分类
- Explain why naive independence assumption 是 mathematically wrong but produces correct class rankings 在 practice
- Compare Multinomial, Bernoulli, 和 Gaussian Naive Bayes variants 和 select right one 为了 given 特征 type
- Evaluate Naive Bayes against logistic 回归 在 high-dimensional sparse 数据 和 explain 偏置-variance tradeoff 在 work

## Problem

你需要 到 classify text. Emails into spam 或 not-spam. Customer reviews into positive 或 negative. Support tickets into categories. You have thousands 的 特征 (one per word) 和 limited 训练 数据.

Most classifiers choke here. Logistic 回归 needs enough samples 到 estimate thousands 的 权重 reliably. Decision trees split 在 one word 在 time 和 overfit wildly. KNN 在 10,000 dimensions 是 meaningless because every point 是 equally far 从 every other point.

Naive Bayes handles 这个. It makes mathematically wrong assumption ( every 特征 是 independent 的 every other 特征 given class), 和 it still outperforms "smarter" 模型 在 text 分类, especially 使用 small 训练 sets. It trains 在 single pass through 数据. It scales 到 millions 的 特征. It produces 概率 estimates (though often poorly calibrated due 到 independence assumption).

Understanding why wrong assumption leads 到 good predictions teaches you something fundamental about machine learning: best 模型 是 not most correct one, it 是 one 使用 best 偏置-variance tradeoff 为了 your 数据.

## Concept

### Bayes' Theorem (Quick Review)

Bayes' theorem flips conditional probabilities:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

We want `P(class | 特征)` -- 概率 document belongs 到 class given words 在 it. 我们可以 compute 这个 从:
- `P(特征 | class)` -- likelihood 的 seeing 这些 words 在 documents 的 这个 class
- `P(class)` -- prior 概率 的 class (how common 是 spam 在 general?)
- `P(特征)` -- evidence, same 为了 all classes, so we can ignore it when comparing

class 使用 highest `P(class | 特征)` wins.

### Naive Independence Assumption

Computing `P(特征 | class)` exactly requires estimating joint 概率 的 all 特征 together. With vocabulary 的 10,000 words, you would need 到 estimate distribution over 2^10,000 possible combinations. Impossible.

naive assumption: every 特征 是 conditionally independent given class.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Instead 的 one impossible joint distribution, you estimate n simple per-特征 distributions. Each one needs only count.

This assumption 是 obviously wrong. words "machine" 和 "learning" 是 not independent 在 any document. But classifier does not need correct 概率 estimates. It needs correct rankings -- which class has highest 概率. independence assumption introduces systematic errors, but 那些 errors affect all classes similarly, so ranking stays correct.

### Why It Still Works

Three reasons:

1. **Ranking over calibration.** 分类 only needs top-ranked class 到 be correct. Even if P(spam) = 0.99999 when true 概率 是 0.7, classifier still picks spam correctly. We do not need correct probabilities. We need correct winner.

2. **High 偏置, low variance.** independence assumption 是 strong prior. It constrains 模型 heavily, which prevents 过拟合. With limited 训练 数据, 模型 是 slightly wrong but stable beats 模型 是 theoretically right but wildly unstable. 这是 偏置-variance tradeoff 在 action.

3. **特征 redundancy cancels out.** Correlated 特征 provide redundant evidence. classifier double-counts 这个 evidence, but it double-counts it 为了 correct class too. If "machine" 和 "learning" always appear together, both provide evidence 为了 "tech" class. NB counts them twice, but it counts them twice 为了 right class.

fourth, practical reason: Naive Bayes 是 extremely fast. 训练 是 single pass through 数据 counting frequencies. Prediction 是 矩阵 multiplication. 你可以 train 在 million documents 在 seconds. This speed means you can iterate faster, try more 特征 sets, 和 run more experiments than 使用 slower 模型.

### Math Step 通过 Step

让我们 trace through concrete example. Suppose we have two classes: spam 和 not-spam. Our vocabulary has three words: "free", "money", "meeting".

训练 数据:
- Spam emails mention "free" 80 times, "money" 60 times, "meeting" 10 times (150 total words)
- Not-spam emails mention "free" 5 times, "money" 10 times, "meeting" 100 times (115 total words)
- 40% 的 emails 是 spam, 60% 是 not-spam

With Laplace smoothing (alpha=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

New email contains: "free" (2 times), "money" (1 time), "meeting" (0 times).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam wins 通过 large margin. word "free" appearing twice 是 strong evidence 为了 spam. 注意 "meeting" not appearing contributes zero 到 both log sums (0 * log(P)) -- 在 Multinomial NB, absent words have no effect. 它是 Bernoulli NB explicitly 模型 word absence.

### Three Variants

Naive Bayes comes 在 three flavors. Each 模型 `P(特征 | class)` differently.

#### Multinomial Naive Bayes

Models each 特征 作为 count. Best 为了 text 数据 where 特征 是 word frequencies 或 TF-IDF values.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` 是 Laplace smoothing (explained below). This variant 是 workhorse 为了 text 分类.

#### Gaussian Naive Bayes

Models each 特征 作为 normal distribution. Best 为了 continuous 特征.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

Each class gets its own mean 和 variance per 特征. This works well when 特征 genuinely follow bell curve within each class.

#### Bernoulli Naive Bayes

Models each 特征 作为 binary (present 或 absent). Best 为了 short text 或 binary 特征 向量.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

Unlike Multinomial, Bernoulli explicitly penalizes absence 的 word. If "free" typically appears 在 spam but 是 absent 从 这个 email, Bernoulli counts 作为 evidence against spam.

### When 到 Use Each Variant

| Variant | 特征 Type | Best For | Example |
|---------|-------------|----------|---------|
| Multinomial | Counts 或 frequencies | Text 分类, bag-的-words | Email spam, topic 分类 |
| Gaussian | Continuous values | Tabular 数据 使用 normal-ish 特征 | Iris 分类, sensor 数据 |
| Bernoulli | Binary (0/1) | Short text, binary 特征 向量 | SMS spam, presence/absence 特征 |

### Laplace Smoothing

What happens when word appears 在 test 数据 but never appeared 在 训练 数据 为了 particular class?

Without smoothing: `P(word | class) = 0/N = 0`. One zero multiplied through entire product makes `P(class | 特征) = 0`, regardless 的 all other evidence. single unseen word destroys entire prediction, no matter how much other evidence supports it.

Laplace smoothing adds small count `alpha` (usually 1) 到 every 特征 count:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

With alpha=1, every word gets 在 least tiny 概率. word "discombobulate" appearing 在 test email no longer kills spam 概率. smoothing has Bayesian interpretation: it 是 equivalent 到 placing uniform Dirichlet prior 在 word distributions.

Higher alpha means stronger smoothing (more uniform distributions). Lower alpha means 模型 trusts 数据 more. Alpha 是 超参数 you tune.

effect 的 alpha:

| Alpha | Effect | When 到 use |
|-------|--------|-------------|
| 0.001 | Almost no smoothing, trust 数据 | Very large 训练 set, no unseen 特征 expected |
| 0.1 | Light smoothing | Large 训练 set |
| 1.0 | Standard Laplace smoothing | Default starting point |
| 10.0 | Heavy smoothing, flattens distributions | Very small 训练 set, many unseen 特征 expected |

### Log-Space Computation

Multiplying hundreds 的 probabilities (each less than 1) causes floating-point underflow. product becomes zero 在 floating point even though true value 是 very small positive number.

solution: work 在 log space. Instead 的 multiplying probabilities, add their logarithms:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

This turns prediction into dot product:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵 multiplication. 那是 why Naive Bayes prediction 是 so fast -- it 是 same operation 作为 single-层 linear 模型.

### Naive Bayes vs Logistic 回归

Both 是 linear classifiers 为了 text. difference 是 在 what they 模型.

| Aspect | Naive Bayes | Logistic 回归 |
|--------|------------|-------------------|
| Type | Generative (模型 P(X\|Y)) | Discriminative (模型 P(Y\|X)) |
| 训练 | Count frequencies | Optimize 损失函数 |
| Small 数据 | Better (strong prior helps) | Worse (not enough 到 estimate 权重) |
| Large 数据 | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, very fast | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

Rule 的 thumb: start 使用 Naive Bayes. If you have enough 数据 和 NB plateaus, switch 到 logistic 回归.

### 分类 Pipeline

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

In practice, we work 在 log space 到 avoid floating-point underflow. Instead 的 multiplying many small probabilities, we add their logarithms:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## Build It

代码 在 `代码/naive_bayes.py` implements both MultinomialNB 和 GaussianNB 从 scratch.

### MultinomialNB

从-scratch implementation:

1. **fit(X, y)**: For each class, count frequency 的 each 特征. Add Laplace smoothing. Compute log probabilities. Store class priors (log 的 class frequencies).

2. **predict_log_proba(X)**: For each sample, compute log P(class) + sum 的 log P(feature_i | class) 为了 all classes. 这是 矩阵 multiplication: X @ log_probs.T + log_priors.

3. **predict(X)**: Return class 使用 highest log 概率.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

key insight: after fitting, prediction 是 just 矩阵 multiplication plus 偏置. 这是 why Naive Bayes 是 so fast.

### GaussianNB

For continuous 特征, we estimate mean 和 variance per class per 特征:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

Prediction uses Gaussian PDF per 特征, multiplied across 特征 (added 在 log space).

### Demo: Text 分类

代码 generates synthetic bag-的-words 数据 simulating two classes (tech articles vs sports articles). Each class has different word frequency distribution. MultinomialNB classifies them using word counts.

synthetic 数据 works like 这个: we create 200 "words" (特征 columns). Words 0-39 have high frequency 在 tech articles 和 low 在 sports. Words 80-119 have high frequency 在 sports 和 low 在 tech. Words 40-79 是 medium frequency 在 both. This creates realistic scenario where some words 是 strong class indicators 和 others 是 noise.

### Demo: Continuous Features

代码 generates Iris-like 数据 (3 classes, 4 特征, Gaussian clusters). GaussianNB classifies using per-class mean 和 variance. Each class has different center (mean 向量) 和 different spread (variance), mimicking real-world 数据 where measurements differ systematically between categories.

代码 also demonstrates:
- **Smoothing comparison:** 训练 MultinomialNB 使用 different alpha values 到 show effect 的 smoothing strength 在 准确率.
- **训练 size experiment:** How NB 准确率 improves 作为 训练 数据 grows 从 20 到 1600 samples. NB reaches decent 准确率 even 使用 very few samples -- 这个 是 its main advantage.
- **Confusion 矩阵:** Per-class 精确率, 召回率, 和 F1 score 到 show where NB makes mistakes.

### Prediction Speed

Naive Bayes prediction 是 矩阵 multiplication. For n samples 使用 d 特征 和 k classes:
- MultinomialNB: one 矩阵 multiply (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k Gaussian PDF evaluations, each over d 特征 = O(n * d * k)

Both 是 linear 在 every dimension. Compare 这个 到 KNN (which requires distance computation 到 all 训练 points) 或 SVM 使用 RBF kernel (which requires kernel evaluation against all support 向量). NB 是 faster 通过 orders 的 magnitude 在 prediction time.

## Use It

With sklearn, both variants 是 one-liners:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

For text 分类 使用 sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

代码 在 `naive_bayes.py` compares 从-scratch implementations against sklearn 在 same 数据 到 verify correctness.

### TF-IDF 使用 Naive Bayes

Raw word counts give every word equal 权重 per occurrence. But common words like "" 和 "是" appear frequently 在 every class -- they carry no information. TF-IDF (Term Frequency - Inverse Document Frequency) downweights common words 和 upweights rare, discriminative words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF values 是 non-negative, so they work 使用 MultinomialNB. combination 的 TF-IDF + MultinomialNB 是 one 的 strongest baselines 为了 text 分类. It frequently beats more complex 模型 在 数据集 使用 fewer than 10,000 训练 samples.

### BernoulliNB 为了 Short Text

For short text (tweets, SMS, chat messages), BernoulliNB can outperform MultinomialNB. Short texts have low word counts, so frequency information MultinomialNB relies 在 是 noisy. BernoulliNB only cares about presence 或 absence, which 是 more reliable 使用 short text.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

`binary=True` flag 在 CountVectorizer converts all counts 到 0/1. Without it, BernoulliNB still works but 是 seeing counts it was not designed 为了.

### Calibrating NB Probabilities

NB probabilities 是 poorly calibrated. When NB says P(spam) = 0.95, true 概率 might be 0.7. If you need reliable 概率 estimates (为了 example, 到 set threshold 或 到 combine 使用 other 模型), use sklearn's CalibratedClassifierCV:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

This fits logistic 回归 在 top 的 NB's raw scores using cross-验证. resulting probabilities 是 much closer 到 true class frequencies.

### Common Gotchas

1. **Negative 特征 values.** MultinomialNB requires non-negative 特征. If you have negative values (like TF-IDF 使用 certain settings 或 standardized 特征), use GaussianNB instead, 或 shift 特征 到 be positive.

2. **Zero variance 特征.** GaussianNB divides 通过 variance. If 特征 has zero variance 为了 class (all values identical), 概率 computation breaks. 代码 adds small smoothing term (1e-9) 到 all variances 到 prevent 这个.

3. **Class imbalance.** If 99% 的 emails 是 not-spam, prior P(not-spam) = 0.99 是 so strong it overwhelms likelihood evidence. 你可以 set class priors manually 或 use class_prior 参数 在 sklearn.

4. **特征 scaling.** MultinomialNB does not need scaling (it works 在 counts). GaussianNB does not need scaling either (it estimates per-特征 统计学). 这是 advantage over logistic 回归 和 SVM, which 是 sensitive 到 特征 scales.

## Ship It

This lesson produces:
- `输出/skill-naive-bayes-chooser.md` -- decision skill 为了 picking right NB variant
- `代码/naive_bayes.py` -- MultinomialNB 和 GaussianNB 从 scratch, 使用 sklearn comparison

### When Naive Bayes Fails

NB fails when independence assumption causes incorrect rankings (not just incorrect probabilities). This happens when:

1. **Strong 特征 interactions.** If class depends 在 combination 的 two 特征 but not either alone (XOR-like patterns), NB will miss it entirely. Each 特征 alone provides no evidence, 和 NB cannot combine them nonlinearly.

2. **Highly correlated 特征 使用 opposing evidence.** If 特征 says "spam" 和 特征 B says "not-spam", but 和 B 是 perfectly correlated (they always agree 在 reality), NB will see conflicting evidence where there 是 none.

3. **Very large 训练 sets.** With enough 数据, discriminative 模型 like logistic 回归 learn true decision boundary 和 outperform NB. independence assumption helped 使用 small 数据 now holds 模型 back.

In practice, 这些 failure modes 是 rare 为了 text 分类. Text 特征 是 numerous, individually weak, 和 independence assumption's errors tend 到 cancel out. For tabular 数据 使用 few strongly correlated 特征, consider logistic 回归 或 tree-based 模型 first.

## Exercises

1. **Smoothing experiment.** Train MultinomialNB 在 text 数据 使用 alpha values 的 0.01, 0.1, 1.0, 10.0, 和 100.0. Plot 准确率 vs alpha. Where does performance peak? Why does very high alpha hurt?

2. **特征 independence test.** Take real text 数据集. Pick two words 是 obviously correlated ("machine" 和 "learning"). Compute P(word1 | class) * P(word2 | class) 和 compare 到 P(word1 AND word2 | class). How wrong 是 independence assumption? Does it affect 分类 准确率?

3. **Bernoulli implementation.** Extend 代码 使用 BernoulliNB class. Convert bag-的-words 到 binary (present/absent) 和 compare 准确率 against MultinomialNB 在 text 数据. When does Bernoulli win?

4. **NB vs Logistic 回归.** Train both 在 text 数据. Start 使用 100 训练 samples 和 increase 到 10,000. Plot 准确率 vs 训练 set size 为了 both. At what point does Logistic 回归 overtake Naive Bayes?

5. **Spam filter.** Build complete spam classifier: tokenize raw email text, build vocabulary, create bag-的-words 特征, train MultinomialNB, evaluate 使用 精确率 和 召回率 (not just 准确率 -- why?).

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Naive Bayes | "Simple probabilistic classifier" | classifier applies Bayes' theorem 使用 assumption 特征 是 conditionally independent given class |
| Conditional independence | "Features don't affect each other" | P(, B \| C) = P( \| C) * P(B \| C) -- knowing B tells you nothing new about once you know C |
| Laplace smoothing | "Add-one smoothing" | Adding small count 到 every 特征 到 prevent zero probabilities 从 dominating prediction |
| Prior | "What you believed before seeing 数据" | P(class) -- 概率 的 each class before observing any 特征 |
| Likelihood | "How well 数据 fits" | P(特征 \| class) -- 概率 的 observing 这些 特征 if class 是 known |
| Posterior | "What you believe after seeing 数据" | P(class \| 特征) -- updated 概率 的 class after observing 特征 |
| Generative 模型 | "Models how 数据 是 generated" | 模型 learns P(X \| Y) 和 P(Y), then uses Bayes' theorem 到 get P(Y \| X) |
| Discriminative 模型 | "Models decision boundary" | 模型 directly learns P(Y \| X) without modeling how X 是 generated |
| Log 概率 | "Avoid underflow" | Working 使用 log P instead 的 P 到 prevent product 的 many small numbers 从 becoming zero 在 floating point |

## Further Reading

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) -- all three variants 使用 mathematical details
- [McCallum 和 Nigam, Comparison 的 Event Models 为了 Naive Bayes Text 分类 (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- classic comparison 的 Multinomial vs Bernoulli 为了 text
- [Rennie et al., Tackling Poor Assumptions 的 Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- improvements 到 NB 为了 text
- [Ng 和 Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- proves NB converges faster than LR 使用 less 数据
