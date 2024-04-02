from typing import TypeVar, Generic, List, Tuple
from abc import ABC, abstractmethod

import numpy as np
import numpy.random as rand
from scipy.special import logsumexp  # type: ignore
import scipy.stats as stats  # type: ignore

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()


T = TypeVar("T")


class Distribution(ABC, Generic[T]):
    @abstractmethod
    def sample(self) -> T:
        pass

    @abstractmethod
    def log_prob(self, x: T) -> float:
        pass

    @abstractmethod
    def stats(self) -> Tuple[float, float]:
        pass


class Categorical(Distribution[T]):
    def __init__(self, values: List[T], logits: List[float]):
        assert len(values) == len(logits)
        self.values = values
        self.logits = logits
        lse = logsumexp(logits)
        self.probs = np.exp(logits - lse)

    def shrink(self):
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
        return list(zip(self.values, self.probs))

    def sample(self) -> T:
        u = rand.rand()
        i = np.searchsorted(np.cumsum(self.probs), u)
        return self.values[i]

    def log_prob(self, v: T) -> float:
        i = self.values.index(v)
        return np.log(self.probs[i])

    def stats(self) -> Tuple[float, float]:
        values = np.array(self.values)
        mean = np.average(values, weights=self.probs).item()
        std = np.sqrt(np.cov(values, aweights=self.probs)).item()
        return (mean, std)

    def plot(self, **kwargs):
        plt.plot(self.values, self.probs, marker=".", linestyle="", **kwargs)

    def hist(self, **kwargs):
        self.shrink()
        sns.barplot(x=self.values, y=self.probs, errorbar=None)


class Empirical(Distribution[T]):
    def __init__(self, samples: List[T]):
        self.samples = samples

    def sample(self) -> T:
        i = rand.randint(len(self.samples))
        return self.samples[i]

    def log_prob(self, v: T) -> float:
        return 1 / len(self.samples) if v in self.samples else 0

    def stats(self) -> Tuple[float, float]:
        samples = np.array(self.samples)
        return (np.mean(samples), np.std(samples))

    def hist(self, **kwargs):
        sns.histplot(self.samples, kde=True, stat="probability", **kwargs)



class Dirac(Categorical[T]):
    def __init__(self, v: T):
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


class Bernoulli(Categorical[float]):
    def __init__(self, p: float):
        assert 0 <= p <= 1
        self.p = p

    def support(self) -> List[Tuple[T, float]]:
        return [(0, (1 - self.p)), (1, self.p)]

    def sample(self) -> float:
        return rand.binomial(1, self.p)

    def log_prob(self, x) -> float:
        return stats.bernoulli.logpmf(x, self.p)

    def stats(self) -> Tuple[float, float]:
        return stats.bernoulli.stats(self.p)


class Binomial(Distribution[float]):
    def __init__(self, n: int, p: float):
        assert n > 0
        assert 0 <= p <= 1
        self.n = n
        self.p = p

    def sample(self) -> float:
        return rand.binomial(self.n, self.p)

    def log_prob(self, x) -> float:
        return stats.binom.logpmf(x, self.n, self.p)

    def stats(self) -> Tuple[float, float]:
        return stats.binom.stats(self.n, self.p)


class Uniform(Distribution[float]):
    def __init__(self, a: float, b: float):
        assert a <= b
        self.a = a
        self.b = b

    def sample(self):
        return rand.uniform(self.a, self.b)

    def log_prob(self, v: float) -> float:
        return 1.0

    def stats(self):
        return stats.uniform.stats(self.a, self.b)


class Gaussian(Distribution[float]):
    def __init__(self, mu: float, sigma: float):
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
    match dist:
        case Categorical():
            return [Categorical(list(values), dist.logits) for values in zip(*dist.values)]
        case Empirical():
            return [Empirical(list(samples)) for samples in zip(*dist.samples)]
        case _:
            raise RuntimeError("We can only split discrete or empirical distributions")
