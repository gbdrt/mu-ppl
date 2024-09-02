from typing import (
    TypeVar,
    Callable,
    ParamSpec,
    List,
    Generic,
    Iterator,
    Any,
    Dict,
    Optional,
    Tuple,
)
from abc import ABC, abstractmethod
import numpy as np
from .distributions import Distribution, Dirac, Categorical, Empirical
from copy import deepcopy
from tqdm import tqdm  # type: ignore

T = TypeVar("T")
P = ParamSpec("P")


class Handler:
    """
    Handler based inference à la Pyro
    https://probprog.cc/2020/assets/posters/thu/49.pdf

    A handler interprets probabilistic constructs inside a context manager.
    We can try several inference algorithms on the same model.
    The implementation of each constructs depends on the inference algorithm.
    The default handler simply draw a sample from the model and ignore conditioning operators (`assume`, `observe`).
    """

    def __enter__(self):
        global _HANDLER
        self.old_handler = _HANDLER
        _HANDLER = self
        return self

    def __exit__(self, type, value, traceback):
        global _HANDLER
        _HANDLER = self.old_handler

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        return dist.sample()  # draw sample

    def assume(self, p: bool):
        pass  # ignore

    def factor(self, weight: float):
        pass  # ignore

    def observe(self, dist: Distribution[T], value: T):
        pass  # ignore

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Distribution[T]:
        return Dirac(model(*args, **kwargs))


_HANDLER = Handler()


def sample(dist: Distribution[T], name: Optional[str] = None) -> T:
    """
    Sample a distribution

    Parameters
    ----------
    dist: Distribution[T]
        Prior distribution
    name: Optional[str]
        Unique name for the sample site (required by some inference)

    Returns
    -------
    T
        A sample of the distribution
    """
    return _HANDLER.sample(dist, name=name)


def assume(p: bool):
    """
    Reject execution which do not satisfy a boolean predicate

    Parameters
    ----------
    p: bool
        Boolean condition
    """
    return _HANDLER.assume(p)


def factor(weight: float):
    """
    Add a factor to the log probability of the model

    Parameters
    ----------
    weight: float
        Value to be added
    """
    return _HANDLER.factor(weight)


def observe(dist: Distribution[T], value: T):
    """
    Assume that a value `v` was sampled form a distribution `dist`.

    Parameters
    ----------
    dist: Distribution[T]
        Assumed distribution
    value: T
        Observed value
    """
    return _HANDLER.observe(dist, value)


def infer(model: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Distribution[T]:
    """
    Compute the posterior distribution the possible outputs of a model given its arguments.

    Parameters
    ----------
    model: Callable[P, T]
        The model
    *args: P.args
        Model arguments
    **kwargs: P.kwargs
        Model keywords arguments

    Returns
    -------
    Distribution[T]
        The posterior distribution
    """
    return _HANDLER.infer(model, *args, **kwargs)


class Reject(Exception):
    pass


class Enumeration(Handler):
    """
    Enumeration.
    Performs a depth-first search on all possible values for each sample sites.
    Only works with finite support distributions for `sample`.

    The DFS algorithm requires all sample site to have a unique name.
    """

    def __init__(self) -> None:
        self.stack: List[Dict[str, Any]] = []
        self.trace: Dict[str, Any] = {}
        self.score: float = 0

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        assert name, "Enumeration inference requires naming sample sites"
        assert isinstance(
            dist, Categorical
        ), "Enumeration only works with Categorical (finite support) distributions"
        if not self.stack:
            # empty stack: Add all possible values to the stack
            self.stack = [{name: (v, w)} for v, w in dist.support()]
        if not name in self.stack[0]:
            # new sample site
            self.stack = [  # add all possible traces starting with self.trace
                {**d, name: (v, w)}
                for d in self.stack
                for (v, w) in dist.support()
                if self.trace == d
            ] + [  # keep all other traces
                d for d in self.stack if self.trace != d
            ]

        v, w = self.stack[0][name]  # pick the first trace
        self.trace[name] = v, w  # record the current choice in self.trace
        self.score += np.log(w)  # update the score
        return v

    def assume(self, p: bool):
        if not p:
            raise Reject

    def factor(self, weight: float):
        self.score += weight  # update the score

    def observe(self, dist: Distribution[T], value: T):
        self.score += dist.log_prob(value)  # update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Categorical[T]:
        samples: List[Tuple[T, float]] = []

        while True:
            self.score = 0  # reset the score
            self.trace = {}  # reset the trace
            try:
                samples.append(
                    (model(*args, **kwargs), self.score)
                )  # try the first trace
                self.stack.pop(0)  # remove trace from the stack
            except Reject:
                self.stack.pop(0)  # drop impossible trace
            if not self.stack:
                break  # no more trace to explore

        return Categorical(samples)


class ImportanceSampling(Handler):
    """
    Importance sampling.
    Launches a set of `num_particles` independent executions (the particles).
    Each particles returns a sample for the output and a score.
    All results are gathered into a Categorical distribution.
    """

    def __init__(self, num_particles: int = 1000) -> None:
        """
        Parameters
        ----------
        num_particles: int
            number of particles (default 1000)
        """
        self.num_particles = num_particles
        self.score: float = 0

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        return dist.sample()  # draw sample

    def assume(self, p: bool):
        if not p:
            self.score += -np.inf

    def factor(self, weight: float):
        self.score += weight  # update the score

    def observe(self, dist: Distribution[T], value: T):
        self.score += dist.log_prob(value)  # update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Categorical[T]:
        samples: List[Tuple[T, float]] = []
        for _ in tqdm(range(self.num_particles)):  # run num_particles executions
            self.score = 0  # reset the score
            samples.append((model(*args, **kwargs), self.score))
        return Categorical(samples)


class RejectionSampling(ImportanceSampling):
    """
    Rejection sampling.
    Tries to generate `num_samples` valid samples.
    If an `assume` raises the `Reject` exception, the inference loops to generate a new sample.
    """

    def __init__(self, num_samples: int = 1000, max_score: float = 0) -> None:
        """
        Parameters
        ----------
        num_samples: int
            samples size (default 1000)
        max_score: float
            Maximum possible score (default 0)
        """
        self.num_samples = num_samples
        self.max_score: float = max_score
        self.score: float = 0

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Empirical[T]:
        samples: List[T] = []

        def gen():  # generate one sample
            while True:
                self.score = 0  # reset the score
                value = model(*args, **kwargs)
                alpha = np.exp(min(0, self.score - self.max_score))
                u = np.random.random()
                if u <= alpha:
                    return value  # accept

        samples = [gen() for _ in tqdm(range(self.num_samples))]
        return Empirical(samples)


class SimpleMetropolis(ImportanceSampling):
    """
    Multi-Sites Markov Chain Monte Carlo (MCMC).
    Try to generate `num_samples` valid samples.
    At each step:
    - Run the model to generate a sample
    - Compare the scores of the new and the previous execution
    - Accept or reject with Metropolis-Hasting
    """

    def __init__(
        self, num_samples: int = 1000, warmups: int = 0, thinning: int = 1
    ) -> None:
        """
        Parameters
        ----------
        num_samples: int
            samples size (default 1000)
        warmups: int
            initial iterations before collecting samples, wait convergence (default 0)
        thinning: int
            fraction of samples to keep, avoid autocorrelation (default 1)
        """
        self.num_samples = num_samples
        self.warmups = warmups
        self.thinning = thinning
        self.score: float = 0  # current score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Empirical[T]:
        samples: List[T] = []
        new_value = model(*args, **kwargs)  # generate first sample

        for _ in tqdm(range(self.warmups + self.num_samples * self.thinning)):
            old_score = self.score  # store state
            old_value = new_value  # store current value
            self.score = 0  # reset the score
            new_value = model(*args, **kwargs)  # generate a candidate
            alpha = np.exp(min(0, self.score - old_score))
            u = np.random.random()
            if not (u < alpha):
                self.score = old_score  # rollback
                new_value = old_value
            samples.append(new_value)

        return Empirical(samples[self.warmups :: self.thinning])


class MetropolisHastings(ImportanceSampling):
    """
    Single-Site Markov Chain Monte Carlo (MCMC).
    Try to generate `num_samples` valid samples.
    At each step:
    - Pick a sample site `regen` at random
    - For each sample site, reuse previous values (!= regen)
    - Log the score of all observation
    - Compare the scores of the new and the previous trace
    - Accept or reject with Metropolis-Hasting
    """

    def __init__(
        self, num_samples: int = 1000, warmups: int = 0, thinning: int = 1
    ) -> None:
        """
        Parameters
        ----------
        num_samples: int
            samples size (default 1000)
        warmups: int
            initial iterations before collecting samples, wait convergence (default 0)
        thinning: int
            fraction of samples to keep, avoid autocorrelation (default 1)
        """
        self.num_samples = num_samples
        self.warmups = warmups
        self.thinning = thinning

        self.score: float = 0
        self.x_samples: Dict[str, Any] = {}  # samples store
        self.x_scores: Dict[str, float] = {}  # X scores
        self.cache: Dict[str, Any] = {}  # sample cache to be reused

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        assert name, "MCMC inference requires naming sample sites"
        try:  # reuse if possible
            v = self.cache[name]
        except KeyError:
            v = dist.sample()  # otherwise draw a sample
        self.x_samples[name] = v  # store the sample
        self.x_scores[name] = dist.log_prob(v)
        return v

    def mh(self, p_state) -> float:
        if np.isinf(self.score):
            return 0.0
        p_score, _, p_x_scores = p_state
        l_alpha = np.log(len(p_x_scores)) - np.log(len(self.x_scores))
        l_alpha += self.score - p_score
        for x in self.cache.keys() & self.x_scores.keys():
            l_alpha += self.x_scores[x]
            l_alpha -= p_x_scores[x]
        return np.exp(min(0.0, l_alpha))

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Empirical[T]:
        samples: List[T] = []
        new_value = model(*args, **kwargs)  # generate first trace

        for _ in tqdm(range(self.warmups + self.num_samples * self.thinning)):
            p_state = self.score, self.x_samples, self.x_scores  # store state
            p_value = new_value  # store current value
            regen = np.random.choice([n for n in self.x_samples])
            self.cache = deepcopy(self.x_samples)  # use samples as next cache
            del self.cache[regen]  # force regen to be resampled
            self.score, self.x_samples, self.x_scores = 0, {}, {}  # reset the state
            new_value = model(*args, **kwargs)  # regen a new trace from regen_from
            alpha = self.mh(p_state)
            u = np.random.random()
            if not (u < alpha):
                self.score, self.x_samples, self.scores = p_state  # rollback
                new_value = p_value
            samples.append(new_value)

        return Empirical(samples[self.warmups :: self.thinning])
