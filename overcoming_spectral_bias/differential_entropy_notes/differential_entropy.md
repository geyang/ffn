
# Differential Entropy

The goal: compare the differential entropy computed directly from an analytical distribution,
with those computed from a collection of samples. There are a few sample-based algorithms to
choose from [^1] in `scipy`.

[^1]: [scipy.org ... /scipy.stats.differential_entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.differential_entropy.html)

## An Analytical Example

Consider a uniform distribution between $[0, 1/2]$
$$
p(x) = \begin{cases}
    2&\text{ if } x \in [0, 1/2]\\
    0&\text{ otherwise. }
\end{cases}
$$

The differential entropy is 
$$
\begin{aligned}
H(u) &= - \int_0^{1/2} 2 * \log(2)  \mathrm d x \\
     &= - \log 2 \\
     &= - 0.693
\end{aligned}
$$


There are two ways to compute this numerically. The first is to use analytical
calculation from the distribution. We do so by discretizing the probability distribution
function into bins. We do so by normalizing the sequence of probabilities. This gives us
$H(x) = -\log 2 = -0.69$:

```python
def H_d(ps):
    ps_norm = ps / ps.sum()
    return - np.sum(np.log(ps) * ps_norm)
```
```python
xs = np.linspace(0, 1 / 2, 1001)
ps = np.ones(1001) * 2
h_d = H_d(ps)
doc.print(f"analytical: {h_d:0.2f}")
```

```
analytical: -0.69
```

This result agrees with the `scipy.stats.entropy` function, which takes in a sequence of 
probability values. the scipy entropy does not assume that the probability is normalized,
so it normalizes the `ps` internally first. This means that in order to convert this to the 
differential entropy, we need to scale the result by the log sum.

```python
h_d_analytic = stats.entropy(ps) - np.log(ps.sum())
doc.print(f"analytic entropy w/ scipy: {h_d_analytic:0.3f}")
```

```
analytic entropy w/ scipy: -0.693
```

The second way is to sample from the distribution

```python
from scipy import stats

samples = np.random.uniform(0, 1/2, 10001)
doc.print(f"verify the distribution: min: {samples.min():0.2f}, max: {samples.max():0.2f}")
```

```
verify the distribution: min: 0.00, max: 0.50
```
```python
h_d_sample = stats.differential_entropy(samples)
doc.print(f"sample-based: {h_d_sample:0.3f}")
```

```
sample-based: -0.702
```

## Using The Wrong Measure

What happens when you use the Shannon entropy on the samples, and 
treat the samples as the probabilities? 

```python
samples = np.random.uniform(0, 1/2, 10001)
doc.print(f"verify the distribution: min: {samples.min():0.2f}, max: {samples.max():0.2f}")
```

```
verify the distribution: min: 0.00, max: 0.50
```
```python
h_d_wrong = H_d(samples)
doc.print(f"Shannon Entropy is (incorrectly): {h_d_wrong:0.3f}")
```

```
Shannon Entropy is (incorrectly): 1.195
```
