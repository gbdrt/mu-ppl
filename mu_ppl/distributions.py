from typing import TypeVar, Generic, List, Tuple, ParamSpec
from abc import ABC, abstractmethod

import numpy as np
import numpy.random as rand
from scipy.special import logsumexp  # type: ignore
import scipy.stats as stats  # type: ignore

import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt

# sns.set_theme()


T = TypeVar("T")


class Distribution(ABC, Generic[T]):
    @abstractmethod
    def sample(self) -> T:
        """
        Draw a sample from the distribution.

        Returns
        -------
        T:
            A sample from the Distribution[T]
        """
        pass

    @abstractmethod
    def log_prob(self, x: T) -> float:
        """
        Compute the log probability of the argument (logpdf for continuous distribution, logpmf for discrete distribution).

        Parameters
        ----------
        x: T
            Value in the definition domain of the distribution

        Returns
        -------
        float:
            Log probability of the value `x`
        """
        pass

    @abstractmethod
    def stats(self) -> Tuple[float, float]:
        """
        Compute basic stats of the distribution

        Returns
        -------
        Tuple[float, float]:
            mean and stddev
        """
        pass


class Categorical(Distribution[T]):
    """
    Categorical distribution, i.e., finite support distribution where values can be of arbitrary type.
    """

    def __init__(self, pairs: List[Tuple[T, float]]):
        """
        Parameters
        ----------
        pairs: List[Tuple[T, float]]
            List of pairs (value, score), where the score is in log scale.
        """
        self.values, self.logits = zip(*pairs)
        lse = logsumexp(self.logits)
        self.probs = np.exp(self.logits - lse)

    def shrink(self):
        """
        Remove duplicate values.
        """
        res = {}
        for v, w in zip(self.values, self.probs):
            if v in res:
                res[v] += w
            else:
                res[v] = w
        self.values = list(res.keys())
        self.probs = list(res.values())
        self.logits = np.log(self.probs)

    def support(self) -> List[Tuple[T, float]]:
        """
        Returns the support of the distribution.

        Returns
        -------
        List[Tuple[T, float]]
            A list of pairs (value, proba)
        """
        self.shrink
        return list(zip(self.values, self.probs))

    def sample(self) -> T:
        u = rand.rand()
        i = np.searchsorted(np.cumsum(self.probs), u)
        return self.values[i]

    def log_prob(self, x: T) -> float:
        i = self.values.index(x)
        return np.log(self.probs[i])

    def stats(self) -> Tuple[float, float]:
        values = np.array(self.values)
        mean = np.average(values, weights=self.probs).item()
        std = np.sqrt(np.cov(values, aweights=self.probs)).item()
        return (mean, std)

    def sort(self):
        sorted_indices = np.argsort(self.logits)[::-1]
        self.values = [self.values[i] for i in sorted_indices]
        self.logits = np.array(self.logits)[sorted_indices]
        self.probs = np.array(self.probs)[sorted_indices]


class Empirical(Categorical[T]):
    """
    Empirical distribution, i.e., a simple list of samples.
    """

    def __init__(self, samples: List[T]):
        """
        Parameters
        ----------
        samples: List[T]
            List of samples
        """
        self.samples = samples

    def sample(self) -> T:
        i = rand.randint(len(self.samples))
        return self.samples[i]

    def log_prob(self, x: T) -> float:
        return 1 / len(self.samples) if x in self.samples else 0

    def stats(self) -> Tuple[float, float]:
        samples = np.array(self.samples)
        return (np.mean(samples), np.std(samples))

    def support(self) -> List[Tuple[T, float]]:
        n = len(self.samples)
        return list(zip(self.samples, [1 / n for _ in self.samples]))


class Dirac(Categorical[T]):
    """
    Dirac distribution. Only defined on one value.
    """

    def __init__(self, v: T):
        """
        Parameters
        ----------
        v: T
            Parameter of the Dirac distribution
        """
        self.v = v

    def support(self) -> List[Tuple[T, float]]:
        return [(self.v, 1.0)]

    def sample(self) -> T:
        return self.v

    def log_prob(self, x) -> float:
        return 0.0 if x == self.v else -10e-10

    def stats(self) -> Tuple[float, float]:
        assert isinstance(self.v, float)
        return (self.v, 0.0)


class Bernoulli(Categorical[int]):
    """
    Bernoulli distribution.
    """

    def __init__(self, p: float):
        """
        Parameters
        ----------
        p: float
            Success probability (0 <= p <= 1)
        """
        assert 0 <= p <= 1
        self.p = p

    def support(self) -> List[Tuple[int, float]]:
        return [(0, (1 - self.p)), (1, self.p)]

    def sample(self) -> int:
        return rand.binomial(1, self.p)

    def log_prob(self, x) -> float:
        return stats.bernoulli.logpmf(x, self.p)

    def stats(self) -> Tuple[float, float]:
        return stats.bernoulli.stats(self.p)


class Binomial(Distribution[int]):
    """
    Binomial distribution.
    """

    def __init__(self, n: int, p: float):
        """
        Parameters
        ----------
        n: int
            Number of trials (0 < n)
        p: float
            Success probability (0 <= p <= 1)
        """
        assert n > 0
        assert 0 <= p <= 1
        self.n = n
        self.p = p

    def sample(self) -> int:
        return rand.binomial(self.n, self.p)

    def log_prob(self, x) -> float:
        return stats.binom.logpmf(x, self.n, self.p)

    def stats(self) -> Tuple[float, float]:
        return stats.binom.stats(self.n, self.p)


class RandInt(Categorical[int]):
    """
    Uniform distribution over a range of integers.
    """

    def __init__(self, a, b):
        """
        Parameters
        ----------
        a: int
            Interval lower bound
        b: int
            Interval upper bound (a <= b)
        """
        assert a <= b
        self.a = a
        self.b = b

    def sample(self) -> int:
        return rand.randint(self.a, self.b)

    def log_prob(self, v: int) -> float:
        if self.a <= v <= self.b:
            return -np.log(self.b - self.a)
        else:
            return -np.inf

    def support(self) -> List[Tuple[int, float]]:
        n = self.b + 1 - self.a
        return [(v, 1 / n) for v in range(self.a, self.b + 1)]

    def stats(self) -> Tuple[float, float]:
        return stats.randint.stats(self.a, self.b)


class Uniform(Distribution[float]):
    """
    Uniform (continuous) distribution.
    """

    def __init__(self, a: float, b: float):
        """
        Parameters
        ----------
        a: float
            Interval lower bound
        b: float
            Interval upper bound (a <= b)
        """
        assert a <= b
        self.a = a
        self.b = b

    def sample(self) -> float:
        return rand.uniform(self.a, self.b)

    def log_prob(self, v: float) -> float:
        if self.a <= v <= self.b:
            return -np.log(self.b - self.a)
        else:
            return -np.inf

    def stats(self) -> Tuple[float, float]:
        return stats.uniform.stats(self.a, self.b)


class Gaussian(Distribution[float]):
    """
    Gaussian distribution.
    """

    def __init__(self, mu: float, sigma: float):
        """
        Parameters
        ----------
        mu: float
            mean
        sigma: float
            scale (sigma > 0)
        """
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> float:
        return rand.normal(self.mu, self.sigma)

    def log_prob(self, x: float) -> float:
        return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)

    def stats(self) -> Tuple[float, float]:
        return stats.norm.stats(loc=self.mu, scale=self.sigma)


def split(dist: Distribution[List[T]]) -> List[Distribution[T]]:
    """
    Split a list of distribution over list into the list of marginal distributions.

    Parameters
    ----------
    dist: Distribution[List[T]]
        A distribution over lists of values

    Returns
    -------
    List[Distribution[T]]
        A list of distributions of values
    """
    match dist:
        case Empirical():
            return [Empirical(list(samples)) for samples in zip(*dist.samples)]
        case Categorical():
            return [
                Categorical(list(zip(list(values), dist.logits)))
                for values in zip(*dist.values)
            ]
        case _:
            raise RuntimeError("We can only split discrete or empirical distributions")


def viz(dist: Distribution[float], **kwargs):
    """
    Visualize a distribution over real numbers
    """
    match dist:
        case Empirical():
            sns.histplot(dist.samples, kde=True, stat="probability", **kwargs)
        case Categorical():
            dist.shrink()
            if len(dist.values) < 100:
                sns.barplot(x=dist.values, y=dist.probs, errorbar=None, **kwargs)
            else:
                sns.histplot(
                    x=dist.values,
                    weights=dist.probs,
                    bins=50,
                    kde=True,
                    stat="probability",
                    **kwargs,
                )
        case _:
            assert False, f"No viz available for {dist}"
