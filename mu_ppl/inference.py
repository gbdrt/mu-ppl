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
    The default handler simply draw a sample from the model and ignore conditionning operators (`assume`, `observe`).
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
        return dist.sample()  # Draw sample

    def assume(self, cond: bool, name: Optional[str] = None):
        assert False, "Not implemented"  # Ignore

    def factor(self, weight: float, name: Optional[str] = None):
        assert False, "Not implemented"  # Ignore

    def observe(self, dist: Distribution[T], value: T, name: Optional[str] = None):
        assert False, "Not implemented"  # Ignore

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


def assume(cond: bool, name: Optional[str] = None):
    """
    Reject execution which do not satisfy a boolean predicate

    Parameters
    ----------
    cond: bool
        Boolean condition
    name: Optional[str]
        Unique name (required by some inference)
    """
    return _HANDLER.assume(cond, name=name)


def factor(weight: float, name: Optional[str] = None):
    """
    Add a factor to the log probability of the model

    Parameters
    ----------
    weight: float
        Value to be added
    name: Optional[str]
        Unique name (required by some inference)
    """
    return _HANDLER.factor(weight, name=name)


def observe(dist: Distribution[T], value: T, name: Optional[str] = None):
    """
    Assume that a value `v` was sampled form a distribution `dist`.

    Parameters
    ----------
    dist: Distribution[T]
        Assumed distribution
    value: T
        Observed value
    name: Optional[str]
        Unique name for the observation (required by some inference)
    """
    return _HANDLER.observe(dist, value, name=name)


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
    return _HANDLER.infer(model, *args)


class Reject(Exception):
    pass


class Enumeration(Handler):
    """
    Enumeration.
    Performs a depth-first search on all possible values for each sample sites.
    Only works with finite support distributions for `sample`.

    The DFS algorithm requires all sample site to have a unique name.
    """

    def __init__(self):
        self.stack: List[Dict[str, Any]] = []
        self.trace: Dict[str, Any] = {}
        self.score: float = 0

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        assert name, "Enumeration inference requires naming sample sites"
        assert isinstance(
            dist, Categorical
        ), "Enumeration only works with Categorical (finite support) distributions"
        if not self.stack:
            # Empty stack: Add all possible values to the stack
            self.stack = [{name: (v, w)} for v, w in dist.support()]
        if not name in self.stack[0]:
            # New sample site
            self.stack = [  # Add all possible traces starting with self.trace
                {**d, name: (v, w)}
                for d in self.stack
                for (v, w) in dist.support()
                if self.trace == d
            ] + [  # Keep all other traces
                d for d in self.stack if self.trace != d
            ]

        v, w = self.stack[0][name]  # Pick the first trace
        self.trace[name] = v, w  # Record the current choice in self.trace
        self.score += np.log(w)  # Update the score
        return v

    def assume(self, cond: bool, name: Optional[str] = None):
        if not cond:
            raise Reject

    def factor(self, weight: float, name: Optional[str] = None):
        self.score += weight  # Update the score

    def observe(self, dist: Distribution[T], v: T, name: Optional[str] = None):
        self.score += dist.log_prob(v)  # Update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Categorical[T]:
        values: List[T] = []
        scores: List[float] = []

        while True:
            self.score = 0  # Reset the score
            self.trace = {}  # Reset the trace
            try:
                values.append(model(*args, **kwargs))  # Run the first trace
                scores.append(self.score)  # Log the score
                self.stack.pop(0)  # Remove trace from the stack
            except Reject:
                self.stack.pop(0)  # Drop impossible trace
            if not self.stack:
                break  # No more trace to explore

        return Categorical(values, scores)


class ImportanceSampling(Handler):
    """
    Importance sampling.
    Launches a set of `num_particles` independent executions (the particles).
    Each particles returns a sample for the output and a score.
    All results are gathered into a Categorical distribution.
    """

    def __init__(self, num_particles: int = 1000):
        """
        Parameters
        ----------
        num_particles: int
            number of particles (default 1000)
        """
        self.num_particles = num_particles
        self.score: float = 0

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        return dist.sample()  # Draw sample

    def assume(self, cond: bool, name: Optional[str] = None):
        if not cond:
            self.score += -np.inf

    def factor(self, weight: float, name: Optional[str] = None):
        self.score += weight  # Update the score

    def observe(self, dist: Distribution[T], v: T, name: Optional[str] = None):
        self.score += dist.log_prob(v)  # Update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Categorical[T]:
        values: List[T] = []
        scores: List[float] = []
        for i in tqdm(range(self.num_particles)):  # Run num_particles executions
            self.score = 0  # Reset the score
            values.append(model(*args, **kwargs))
            scores.append(self.score)
        return Categorical(values, scores)


class RejectionSampling(Handler):
    """
    Rejection sampling.
    Tries to generate `num_samples` valid samples.
    If an `assume` raises the `Reject` exception, the inference loops to generate a new sample.
    """

    def __init__(self, num_samples: int = 1000, max_score: float = 0):
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

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        return dist.sample()  # Draw sample

    def assume(self, cond: bool, name: Optional[str] = None):
        if not cond:
            self.score += -np.inf

    def factor(self, weight: float, name: Optional[str] = None):
        self.score += weight

    def observe(self, dist: Distribution[T], v: T, name: Optional[str] = None):
        self.score += dist.log_prob(v)

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Empirical[T]:
        samples: List[T] = []

        def gen():  # Generate one sample
            while True:
                self.score = 0  # Reset the score
                value = model(*args, **kwargs)
                u = np.random.random()
                if self.score > self.max_score + np.log(u):
                    return value  # accept

        samples = [gen() for _ in tqdm(range(self.num_samples))]
        return Empirical(samples)


class SimpleMetropolis(Handler):
    """
    Multi-Sites Markov Chain Monte Carlo (MCMC).
    Try to generate `num_samples` valid samples.
    At each step:
    - Run the model to generate a sample
    - Compare the scores of the new and the previous execution
    - Accept or reject with Metropolis-Hasting
    """

    def __init__(self, num_samples: int = 1000, warmups: int = 0, thinning: int = 1):
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

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        return dist.sample()  # Draw sample

    def assume(self, cond: bool, name: Optional[str] = None):
        if not cond:
            self.score += -np.inf

    def factor(self, weight: float, name: Optional[str] = None):
        self.score += weight  # Update the score

    def observe(self, dist: Distribution[T], v: T, name: Optional[str] = None):
        self.score += dist.log_prob(v)  # Update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Empirical[T]:
        samples: List[T] = []
        new_value = model(*args, **kwargs)  # Generate first sample

        for _ in tqdm(range(self.warmups + self.num_samples * self.thinning)):
            old_score = self.score  # Store current state
            old_value = new_value  # Store current value
            self.score = 0  # Reset the score
            new_value = model(*args, **kwargs)  # Generate a candidate
            alpha = np.exp(self.score - old_score)
            u = np.random.random()
            if not (u < alpha):
                new_value = old_value  # Roll back to the previous value
                self.score = old_score  # Restore previous state
            samples.append(new_value)  # Keep the new trace and the new value

        return Empirical(samples[self.warmups :: self.thinning])


class MetropolisHastings(Handler):
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

    def __init__(self, num_samples: int = 1000, warmups: int = 0, thinning: int = 1):
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

        self.samples: Dict[str, Any] = {}  # samples store
        self.scores: Dict[str, float] = {}  # score store
        self.cache: Dict[str, Any] = {}  # sample cache to be reused

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        assert name, "MCMC inference requires naming sample sites"
        try:  # Reuse if possible
            v = self.cache[name]
        except KeyError:
            v = dist.sample()  # Otherwise draw a sample
        self.samples[name] = v  # Store the sample
        self.scores[name] = dist.log_prob(v)
        return v

    def assume(self, cond: bool, name: Optional[str] = None):
        assert name, "MCMC inference requires naming assume sites"
        if not cond:
            self.score[name] = -np.inf

    def factor(self, weight: float, name: Optional[str] = None):
        assert name, "MCMC inference requires naming score sites"
        self.scores[name] = weight  # Update the score

    def observe(self, dist: Distribution[T], v: T, name: Optional[str] = None):
        assert name, "MCMC inference requires naming observe sites"
        self.scores[name] = dist.log_prob(v)  # Update the score

    def mh(
        self, regen: str, old_samples: Dict[str, Any], old_scores: Dict[str, float]
    ) -> float:
        # MH acceptance
        x_new = {regen} | (self.samples.keys() - old_samples.keys())
        x_old = {regen} | (old_samples.keys() - self.samples.keys())
        alpha = np.log(len(old_samples)) - np.log(len(self.samples))
        for v in self.scores.keys() - x_new:
            alpha += self.scores[v]
        for v in old_scores.keys() - x_old:
            alpha -= old_scores[v]
        return np.exp(alpha)

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Empirical[T]:
        samples: List[T] = []
        new_value = model(*args, **kwargs)  # Generate first trace

        for _ in tqdm(range(self.warmups + self.num_samples * self.thinning)):
            p_samples, p_scores = self.samples, self.scores  # Store current state
            p_value = new_value  # Store current value
            regen = np.random.choice([n for n in self.samples])
            self.cache = deepcopy(self.samples)  # Use samples as next cache
            del self.cache[regen]  # force regen to be resampled
            self.samples, self.scores = {}, {}  # Reset the state
            new_value = model(*args, **kwargs)  # Regen a new trace from regen_from
            alpha = self.mh(regen, p_samples, p_scores)
            u = np.random.random()
            if not (u < alpha):
                new_value = p_value  # Roll back to the previous value
                self.samples, self.scores = (
                    p_samples,
                    p_scores,
                )  # Restore previous state
            samples.append(new_value)

        return Empirical(samples[self.warmups :: self.thinning])


class SSM(ABC, Generic[P, T]):
    """
    Abstract class for State-Space Models.
    These models are state machine characterized by a transition function `step`.

    The SMC inference only work on SSM instances.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def step(self, *args: P.args, **kwargs: P.kwargs) -> T:
        pass


class SMC(ImportanceSampling):
    """
    Sequential Monte-Carlo.

    Model must be expressed as a state machine (SSM).
    Similar to Importance sampling, but particles are resampled after each step.
    """

    def resample(
        self, particles: List[SSM[P, T]], scores: List[float]
    ) -> List[SSM[P, T]]:
        d = Categorical(particles, scores)
        return [
            deepcopy(d.sample()) for _ in range(self.num_particles)
        ]  # Resample a new set of particles

    def infer_stream(
        self, ssm: type[SSM[P, T]], *args: Iterator[P.args]
    ) -> Iterator[Categorical[T]]:
        particles: List[SSM[P, T]] = [
            ssm() for _ in range(self.num_particles)
        ]  # Initialise the particles
        for y in zip(*args):  # At each step
            values: List[T] = []
            scores: List[float] = []
            for i in range(self.num_particles):
                self.score = 0  # Reset the score
                values.append(particles[i].step(*y))  # Execute all the particles
                scores.append(self.score)
            yield Categorical(values, scores)  # Return current distribution
            particles = self.resample(particles, scores)  # Resample the particles
