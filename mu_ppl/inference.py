from typing import TypeVar, Callable, ParamSpec, List, Generic, Iterator, Optional, Any, Protocol
from abc import ABC, abstractmethod

import numpy as np
from .distributions import Distribution, Dirac, Discrete, Empirical

from copy import deepcopy
from tqdm import tqdm  # type: ignore

"""
Handler based inference à la Pyro
https://probprog.cc/2020/assets/posters/thu/49.pdf
"""

T = TypeVar("T")
P = ParamSpec("P")


class Handler:
    def __enter__(self):
        global _HANDLER
        self.old_handler = _HANDLER
        _HANDLER = self
        return self

    def __exit__(self, type, value, traceback):
        global _HANDLER
        _HANDLER = self.old_handler

    def sample(self, name: str, dist: Distribution[T]) -> T:
        return dist.sample()  # Draw sample

    def observe(self, name: str, dist: Distribution[T], value: T):
        pass  # Ignore

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Distribution[T]:
        return Dirac(model(*args, **kwargs))


_HANDLER = Handler()


def sample(name: str, dist: Distribution[T]) -> T:
    return _HANDLER.sample(name, dist)


def observe(name: str, dist: Distribution[T], value: T):
    return _HANDLER.observe(name, dist, value)


def infer(model: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Distribution[T]:
    return _HANDLER.infer(model, *args)


class Reject(Exception):
    pass


class RejectionSampling(Handler):
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples

    def sample(self, name:str, dist: Distribution[T]) -> T:
        return dist.sample()  # Draw sample

    def observe(self, name:str, dist: Distribution[T], v: T):
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
                    pass

        samples = [gen() for _ in tqdm(range(self.num_samples))]
        return Empirical(samples)


class ImportanceSampling(Handler):
    def __init__(self, num_particles: int = 1000):
        self.num_particles = num_particles
        self.id = 0
        self.scores = [0. for _ in range(num_particles)]

    def sample(self, name:str, dist: Distribution[T]) -> T:
        return dist.sample()  # Draw sample

    def observe(self, name:str, dist: Distribution[T], v: T):
        self.scores[self.id] += dist.log_prob(v)  # Update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Discrete[T]:
        values: List[T] = []
        for i in tqdm(range(self.num_particles)):  # Run num_particles executions
            self.id = i
            values.append(model(*args, **kwargs))
        return Discrete(values, self.scores)


class MCMC(Handler):
    def __init__(self, num_samples: int = 1000, warmups: int = 0):
        self.num_samples = num_samples
        self.warmups = warmups
        self.score = 0.0
        self.trace: List[Any] = []  # log all samples
        self.tape: List[Any] = []  # reuse first samples

    def sample(self, name:str, dist: Distribution[T]) -> T:
        if self.tape:  # Reuse if possible otherwise draw a sample
            v = self.tape.pop(0)
        else:
            v = dist.sample()
        self.trace.append(v)  # Add to the trace
        return v

    def observe(self, name:str, dist: Distribution[T], v: T):
        self.score += dist.log_prob(v)  # Update the score

    def infer(
        self, model: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Empirical[T]:
        def _mh(old_trace, old_score, new_trace, new_score):  # MH acceptance prob
            fw = -np.log(len(old_trace))
            bw = -np.log(len(new_trace))
            return min(1.0, np.exp(np.float128(new_score - old_score + bw - fw)))

        samples: List[T] = []
        new_value = model(*args, **kwargs)  # Generate first trace

        for _ in tqdm(range(self.warmups + self.num_samples)):
            samples.append(new_value)  # Store current sample
            old_score, old_trace = self.score, self.trace  # Store current state
            old_value = new_value  # Store current value

            regen_from = np.random.randint(len(self.trace))
            self.score = 0
            self.trace = []
            self.tape = self.trace[:regen_from]
            new_value = model(*args, **kwargs)  # Regen a new trace from regen_from

            if np.random.random() < _mh(old_trace, old_score, self.trace, self.score):
                samples.append(new_value)  # Keep the new trace and the new value
            else:
                new_value = old_value  # Roll back to the previous value
                self.score, self.trace = old_score, old_trace  # Restore previous state

        return Empirical(samples[self.warmups :])


class SSM(ABC, Generic[P, T]):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def step(self, *args: P.args, **kwargs: P.kwargs) -> T:
        pass


class SMC(ImportanceSampling):
    # Model must be expressed as a state machine (SSM).
    # Resample at each step
    def resample(self):
        d = Discrete(self.particles, self.scores)
        self.particles = [  # Resample a new set of particles
            deepcopy(d.sample()) for _ in range(self.num_particles)
        ]
        self.scores = np.zeros(self.num_particles)  # Reset the score

    def infer_stream(
        self, ssm:SSM[P, T], *args: Iterator[P.args]
    ) -> Iterator[Discrete[T]]:
        self.particles = [
            ssm() for _ in range(self.num_particles) # type: ignore
        ]  # Initialise the particles
        for y in zip(*args):  # At each step
            values: List[T] = []
            for i in range(self.num_particles):
                self.id = i
                values.append(self.particles[i].step(*y))  # Execute all the particles
            yield Discrete(values, self.scores)  # Return current distribution
            self.resample()  # Resample the particles
