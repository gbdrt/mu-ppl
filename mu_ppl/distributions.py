from typing import TypeVar, Generic, List, Tuple
from abc import ABC, abstractmethod

import numpy as np
import numpy.random as rand
from scipy.special import logsumexp  # type: ignore
import scipy.stats as stats  # type: ignore


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


class Dirac(Distribution[T]):
    def __init__(self, v: T):
        self.v = v

    def sample(self) -> T:
        return self.v

    def log_prob(self, x) -> float:
        return 0.0 if x == self.v else -10e-10

    def stats(self) -> Tuple[float, float]:
        assert isinstance(self.v, float)
        return (self.v, 0.0)


class Bernoulli(Distribution[float]):
    def __init__(self, p: float):
        assert 0 <= p <= 1
        self.p = p

    def sample(self) -> float:
        return rand.binomial(1, self.p)

    def log_prob(self, x) -> float:
        return stats.bernoulli.logpmf(x, self.p)

    def stats(self) -> Tuple[float, float]:
        return stats.bernoulli.stats(self.p)


class Uniform(Distribution[float]):
    def __init__(self, a, b):
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


class Discrete(Distribution[T]):
    def __init__(self, values: List[T], logits: List[float]):
        assert len(values) == len(logits)
        self.values = values
        self.logits = logits
        lse = logsumexp(logits)
        self.probs = np.exp(logits - lse)

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
