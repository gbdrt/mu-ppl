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

    def assume(self, pred: bool):
        pass  # Ignore

    def observe(self, dist: Distribution[T], value: T, name: Optional[str] = None):
        pass  # Ignore

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


def assume(pred: bool):
    """
    Reject execution which do not satisfy a boolean predicate

    Parameters
    ----------
    pred: bool
        Boolean condition
    """
    return _HANDLER.assume(pred)


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


class RejectionSampling(Handler):
    """
    Rejection sampling.
    Tries to generate `num_samples` valid samples.
    If an `assume` raises the `Reject` exception, the inference loops to generate a new sample.
    """

    def __init__(self, num_samples: int = 1000):
        """
        Parameters
        ----------
        num_samples: int
            samples size (default 1000)
        """
        self.num_samples = num_samples

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        return dist.sample()  # Draw sample

    def assume(self, pred: bool):
        if not pred:
            raise Reject

    def observe(self, dist: Distribution[T], v: T, name: Optional[str] = None):
        x = dist.sample()  # Draw a sample
        if x != v:
            raise Reject  # Reject if it's not the observation

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Empirical[T]:
        def gen():  # Generate one sample
            while True:
                try:
                    return model(*args, **kwargs)
                except Reject:
                    pass  # Retry

        samples = [gen() for _ in tqdm(range(self.num_samples))]
        return Empirical(samples)


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

    def assume(self, pred: bool):
        if not pred:
            raise Reject

    def observe(self, dist: Distribution[T], v: T, name: Optional[str] = None):
        self.score += dist.log_prob(v)  # Update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Categorical[T]:
        def gen():  # Generate one value
            while True:
                self.score = 0  # Reset the score
                self.trace = {}  # Reset the trace
                try:
                    v, w = model(*args, **kwargs), self.score  # Run first trace
                    self.stack.pop(0)  # Remove trace from the stack
                    return v, w
                except Reject:
                    self.stack.pop(0)  # Drop impossible trace

        res = [gen()]  # First run to initialize the stack
        while self.stack:  # Explore remaining traces
            res.append(gen())

        values, scores = zip(*res)
        return Categorical(list(values), list(scores))


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
        self.id = 0
        self.scores = [0.0 for _ in range(num_particles)]

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        return dist.sample()  # Draw sample

    def observe(self, dist: Distribution[T], v: T, name: Optional[str] = None):
        self.scores[self.id] += dist.log_prob(v)  # Update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Categorical[T]:
        values: List[T] = []
        for i in tqdm(range(self.num_particles)):  # Run num_particles executions
            self.id = i
            values.append(model(*args, **kwargs))
        return Categorical(values, self.scores)


class MCMC(Handler):
    """
    Single-Site Markov Chain Monte Carlo (MCMC).
    Tries to generate `num_samples` valid samples.
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
        self.trace: List[Any] = []  # log all samples
        self.samples: Dict[str, Any] = {}  # samples store
        self.cache: Dict[str, Any] = {}  # sample cache to be reused
        self.scores: Dict[str, Any] = {}  # score store

    def sample(self, dist: Distribution[T], name: Optional[str] = None) -> T:
        assert name, "MCMC inference requires naming sample sites"
        try:  # Reuse if possible
            v = self.cache[name]
        except KeyError:
            v = dist.sample()  # Otherwise draw a sample
        self.samples[name] = v  # Store the sample
        self.scores[name] = dist.log_prob(v)
        return v

    def observe(self, dist: Distribution[T], v: T, name: Optional[str] = None):
        assert name, "MCMC inference requires naming observation sites"
        self.scores[name] = dist.log_prob(v)  # Update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Empirical[T]:
        def mh(
            regen, old_samples, old_scores, new_samples, new_scores
        ):  # MH acceptance prob
            x_new = {regen} | (new_samples.keys() - old_samples.keys())
            x_old = {regen} | (old_samples.keys() - new_samples.keys())
            alpha = np.log(len(old_samples)) - np.log(len(new_samples))
            for v in new_scores.keys() - x_new:
                alpha += new_scores[v]
            for v in old_scores.keys() - x_old:
                alpha -= old_scores[v]
            return np.exp(alpha)

        samples: List[T] = []
        new_value = model(*args, **kwargs)  # Generate first trace

        for _ in tqdm(range(self.warmups + self.num_samples - 1)):
            samples.append(new_value)  # Store current sample
            old_samples, old_scores = self.samples, self.scores  # Store current state
            old_value = new_value  # Store current value

            regen = list(self.samples.keys())[np.random.randint(len(self.samples))]
            self.cache = deepcopy(self.samples)  # Use samples as next cache
            del self.cache[regen]  # force regen to be resampled
            self.samples = {}  # Reset the samples
            self.scores = {}  # Reset the scores

            new_value = model(*args, **kwargs)  # Regen a new trace from regen_from

            if np.random.random() < mh(
                regen, old_samples, old_scores, self.samples, self.scores
            ):
                samples.append(new_value)  # Keep the new trace and the new value
            else:
                new_value = old_value  # Roll back to the previous value
                self.samples, self.scores = (
                    old_samples,
                    old_scores,
                )  # Restore previous state

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

    def resample(self, particles: List[SSM[P, T]]) -> List[SSM[P, T]]:
        d = Categorical(particles, self.scores)
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
            for i in range(self.num_particles):
                self.id = i
                values.append(particles[i].step(*y))  # Execute all the particles
            yield Categorical(values, self.scores)  # Return current distribution
            particles = self.resample(particles)  # Resample the particles
            self.scores = [0.0 for _ in range(self.num_particles)]  # Reset the score
