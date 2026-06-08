# 统计学 为了 Machine Learning

> 统计学 是 how you know if your 模型 actually works 或 just got lucky.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (概率 和 Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- Compute descriptive 统计学, Pearson/Spearman correlation, 和 covariance 矩阵 从 scratch
- Perform hypothesis tests (t-test, chi-squared) 和 interpret p-values 和 confidence intervals correctly
- Use bootstrap resampling 到 construct confidence intervals 为了 any metric without distributional assumptions
- Distinguish statistical significance 从 practical significance using effect size measures

## Problem

You trained two 模型. 模型 scores 0.87 在 your test set. 模型 B scores 0.89. You deploy 模型 B. Three weeks later, production metrics 是 worse than before. What happened?

模型 B did not actually outperform 模型 . 0.02 difference was noise. Your test set was too small, 或 variance too high, 或 both. You shipped randomness dressed up 作为 improvement.

This happens constantly. Kaggle leaderboard shakeups. Papers fail 到 reproduce. /B tests declare winners based 在 few hundred samples. root cause 是 always same: someone skipped 统计学.

统计学 gives you tools 到 distinguish signal 从 noise. It tells you when difference 是 real, how confident you should be, 和 how much 数据 you need before you can trust result. Every ML pipeline, every 模型 comparison, every experiment needs 统计学. Without it, you 是 guessing.

## Concept

### Descriptive 统计学: Summarizing Your 数据

Before you 模型 anything, you need 到 know what your 数据 looks like. Descriptive 统计学 compress 数据集 into few numbers capture its shape.

**Measures 的 central tendency** answer "where 是 middle?"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

mean 是 balance point. median 是 halfway mark. When they diverge, your distribution 是 skewed. Income distributions have mean >> median (right skew 从 billionaires). Loss distributions during 训练 often have mean << median (left skew 从 easy samples).

**Measures 的 spread** answer "how dispersed 是 数据?"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Percentiles** divide sorted 数据 into 100 equal parts. 25th percentile (Q1) means 25% 的 values fall below 这个 point. 50th percentile 是 median. 75th percentile 是 Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

In ML, you care about percentiles 为了 inference latency, prediction confidence distributions, 和 understanding error distributions. 模型 使用 low average error but terrible P99 error might be useless 为了 safety-critical applications.

**Sample vs population 统计学.** When computing variance 从 sample, divide 通过 (n-1) instead 的 n. 这是 Bessel's correction. It compensates 为了 fact your sample mean 是 not true population mean. With n 在 denominator, you systematically underestimate true variance. With (n-1), estimate 是 unbiased.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

In practice: if n 是 large (thousands 的 samples), difference 是 negligible. If n 是 small (dozens 的 samples), it matters.

### Correlation: How Variables Move Together

Correlation measures strength 和 direction 的 linear relationship between two variables.

**Pearson correlation coefficient** measures linear association:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson assumes relationship 是 linear 和 both variables 是 roughly normally distributed. 它是 sensitive 到 outliers. single extreme point can drag r 从 0.1 到 0.9.

**Spearman rank correlation** measures monotonic association:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**When 到 use each:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

** golden rule:** correlation does not imply causation. Ice cream sales 和 drowning deaths 是 correlated because both increase 在 summer. Your 模型's 准确率 和 number 的 参数 是 correlated, but adding 参数 does not automatically improve 准确率 (see: 过拟合).

### Covariance 矩阵

covariance between two variables measures how they vary together:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

For d 特征, covariance 矩阵 C 是 d x d 矩阵 where C[i][j] = Cov(feature_i, feature_j). diagonal entries C[i][i] 是 variances 的 each 特征.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Connection 到 PCA.** PCA eigendecomposes covariance 矩阵. eigenvectors 是 principal components (directions 的 maximum variance). eigenvalues tell you how much variance each component captures. 这是 exactly what Lesson 10 covered, but now you see why covariance 矩阵 是 right thing 到 decompose: it encodes all pairwise linear relationships 在 your 数据.

**Connection 到 correlation.** correlation 矩阵 是 covariance 矩阵 的 standardized variables (each divided 通过 its standard deviation). Correlation normalizes covariance so all values fall 在 [-1, 1].

### Hypothesis 测试

Hypothesis 测试 是 framework 为了 making decisions under uncertainty. You start 使用 claim, collect 数据, 和 determine if 数据 是 consistent 使用 claim.

** setup:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

** p-value** 是 概率 的 seeing 数据 作为 extreme 作为 what you observed, assuming H0 是 true. 它是 NOT 概率 H0 是 true. 这是 single most common misunderstanding 在 统计学.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Confidence intervals** give range 的 plausible values 为了 参数:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

width 的 confidence interval tells you about 精确率. Wide intervals mean high uncertainty. Narrow intervals mean your estimate 是 precise (but not necessarily accurate, if your 数据 是 biased).

### t-test

t-test compares means. 有 several flavors.

**One-sample t-test:** 是 population mean different 从 hypothesized value?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Two-sample t-test (independent):** 是 two group means different?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Paired t-test:** when measurements come 在 pairs (same 模型 evaluated 在 same 数据 splits):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

In ML, paired t-test 是 common: you run both 模型 在 same 10 cross-验证 folds 和 compare their scores pairwise.

### Chi-squared Test

chi-squared test checks if observed frequencies match expected frequencies. Useful 为了 categorical 数据.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### /B 测试 为了 ML Models

/B 测试 在 ML 是 not same 作为 web /B 测试. 模型 comparison has specific challenges:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

** procedure:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Statistical Significance vs Practical Significance

result can be statistically significant but practically meaningless. With enough 数据, even trivial difference becomes statistically significant.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Effect size** quantifies how big difference 是, independent 的 sample size:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Always report both p-value 和 effect size. p-value tells you if difference 是 real. effect size tells you if it matters.

### Multiple Comparison Problem

When you test many hypotheses, some will be "significant" 通过 chance. If you test 20 things 在 alpha = 0.05, you expect 1 false positive even when nothing 是 real.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni correction:** divide alpha 通过 number 的 tests.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

In ML, 这个 matters when you compare 模型 across multiple metrics, test many 超参数 configurations, 或 evaluate 在 multiple 数据集.

### Bootstrap Methods

Bootstrapping estimates sampling distribution 的 statistic 通过 resampling your 数据 使用 replacement. No assumptions about underlying distribution required.

** 算法:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap confidence interval (percentile method):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Why bootstrap matters 为了 ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap 为了 模型 comparison:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这是 more robust than paired t-test because it makes no distributional assumptions.

### Parametric vs Non-parametric Tests

**Parametric tests** assume specific distribution (usually normal):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Non-parametric tests** make no distributional assumptions:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**When 到 use non-parametric:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**When 到 use parametric:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

In ML experiments, you typically have small n (5 或 10 cross-验证 folds), so non-parametric tests like Wilcoxon signed-rank 是 often more appropriate than t-tests.

### Central Limit Theorem: Practical Implications

CLT says distribution 的 sample means approaches normal distribution 作为 n grows, regardless 的 underlying population distribution.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Why 这个 matters 为了 ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**What CLT does NOT do:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Common Statistical Mistakes 在 ML Papers

1. **测试 在 训练 set.** Guarantees 过拟合. Always hold out 数据 模型 never sees during 训练.

2. **No confidence intervals.** Reporting single 准确率 number without uncertainty makes results unreproducible 和 unverifiable.

3. **Ignoring multiple comparisons.** 测试 50 configurations 和 reporting best one without correction inflates false positive rates.

4. **Confusing statistical 和 practical significance.** p-value 的 0.001 在 0.01% 准确率 improvement 是 not meaningful.

5. **Using 准确率 在 imbalanced 数据.** 99% 准确率 在 数据集 使用 99% negative class means 模型 learned nothing. Use 精确率, 召回率, F1, 或 AUC.

6. **Cherry-picking metrics.** Reporting only metric where your 模型 wins. Honest evaluation reports all relevant metrics.

7. **Leaking information across train/test splits.** Normalizing before splitting, 或 using future 数据 到 predict past.

8. **Small test sets 使用 no variance estimates.** Evaluating 在 100 samples 和 claiming 2% improvement 是 noise, not signal.

9. **Assuming independence when 数据 是 not independent.** Medical images 从 same patient, multiple sentences 从 same document. Observations within group 是 correlated.

10. **P-hacking.** Trying different tests, subsets, 或 exclusion criteria until you get p < 0.05. result 是 artifact 的 search.

## Building It

你将实现:

1. **Descriptive 统计学 从 scratch** (mean, median, mode, standard deviation, percentiles, IQR)
2. **Correlation 函数** (Pearson 和 Spearman, 使用 covariance 矩阵)
3. **Hypothesis tests** (one-sample t-test, two-sample t-test, chi-squared test)
4. **Bootstrap confidence intervals** (为了 any statistic, no assumptions needed)
5. **/B test simulator** (generate 数据, test, check 为了 Type I 和 Type II errors)
6. **Statistical vs practical significance demo** (showing large n makes everything "significant")

All 从 scratch, using only `math` 和 `random`. No numpy, no scipy.

## Key Terms

| Term | Definition |
|---|---|
| Mean | Sum 的 values divided 通过 count. Sensitive 到 outliers. |
| Median | Middle value 的 sorted 数据. Robust 到 outliers. |
| Standard deviation | Square root 的 variance. Measures spread 在 original units. |
| Percentile | Value below which given percentage 的 数据 falls. |
| IQR | Interquartile range. Q3 minus Q1. spread 的 middle 50%. |
| Pearson correlation | Measures linear association between two variables. Range [-1, 1]. |
| Spearman correlation | Measures monotonic association using ranks. |
| Covariance 矩阵 | 矩阵 的 pairwise covariances between all 特征. |
| Null hypothesis | Default assumption 的 no effect 或 no difference. |
| p-value | 概率 的 数据 这个 extreme given null hypothesis 是 true. |
| Confidence interval | Range 的 plausible values 为了 参数 在 given confidence level. |
| t-test | Tests whether means differ significantly. Uses t-distribution. |
| Chi-squared test | Tests whether observed frequencies differ 从 expected frequencies. |
| Effect size | Magnitude 的 difference, independent 的 sample size. Cohen's d 是 common. |
| Bonferroni correction | Divides significance threshold 通过 number 的 tests 到 control false positives. |
| Bootstrap | Resampling 使用 replacement 到 estimate sampling distributions. |
| Type I error | False positive. Rejecting H0 when it 是 true. |
| Type II error | False negative. Failing 到 reject H0 when it 是 false. |
| Statistical power | 概率 的 correctly rejecting false H0. Power = 1 minus Type II error rate. |
| Central limit theorem | Sample means converge 到 normal distribution 作为 sample size grows. |
| Parametric test | Assumes specific distribution 为了 数据 (usually normal). |
| Non-parametric test | Makes no distributional assumptions. Works 在 ranks 或 signs. |
