from typing import List, Tuple
from mu_ppl import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import arviz as az


def gauss(obs: List[Tuple[float, float]]) -> Tuple[float, float]:
    mu_x = sample(Gaussian(0, 10), name="mu_x")
    mu_y = sample(Gaussian(0, 10), name="mu_y")
    for i, (x, y) in enumerate(obs):
        observe(Gaussian(mu_x, 1), x, name=f"x_{i}")
        observe(Gaussian(mu_y, 1), y, name=f"y_{i}")
    return mu_x, mu_y


def plot_target(mu, cov):
    for sd in range(1, 5):
        eigvalues, eigvectors = np.linalg.eig(cov)
        # Scaling eigenvalues by standard deviation
        eigvalues = np.sqrt(eigvalues) * sd
        angle = np.degrees(np.arccos(eigvectors[0, 0]))
        ax = plt.gca()
        for j in range(1, 2):
            ellipse = Ellipse(
                xy=mu,
                width=eigvalues[0] * j * 0.5,
                height=eigvalues[1] * j * 0.5,
                angle=angle,
                edgecolor="b",
                lw=1.5,
                facecolor="none",
            )
            ax.add_patch(ellipse)


mean = [3, 3]
cov = [[1, 0.5], [0.5, 1]]
obs = np.random.multivariate_normal(mean, cov, 10)
np.random.seed(23)
# np.random.seed(23)

plt.box(False)
ax = plt.gca()
ax.set_axisbelow(True)
ax.grid(True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(left=False, bottom=False)
ax.axhline(0, color="black", linewidth=1.5)
ax.axvline(0, color="black", linewidth=1.5)
ax.axes.set_aspect("equal")  # type: ignore


# np.random.seed(23)
# with SimpleMetropolis(num_samples=7000):
#     dist: Empirical[Tuple[float, float]] = infer(gauss, obs)  # type: ignore
#     x, y = list(zip(*dist.samples))
#     az_data = az.convert_to_inference_data(np.array([x, y]))
#     print(az.summary(az_data))
#     plt.scatter(x, y, color='black', s=7, zorder=2)
#     plt.plot(x, y, color='gray', linewidth=1, zorder=1)
#     plot_target(mean, cov)
#     plt.savefig('2d_gaussian_simple_mh.pdf', format='pdf')
#     plt.show()

np.random.seed(2)
ax.set_xlim((-1.5, 7.1))
ax.set_ylim((-1.1, 5.9))
ax.set_xticks(np.arange(-1, 7))
ax.axes.set_aspect("equal")  # type: ignore

with MetropolisHastings(num_samples=1000):
    dist: Empirical[Tuple[float, float]] = infer(gauss, obs)  # type: ignore
    x, y = list(zip(*dist.samples))
    az_data = az.convert_to_inference_data(np.array([x, y]))
    print(az.summary(az_data))
    plt.scatter(x, y, color="black", s=7, zorder=2)
    plt.plot(x, y, color="gray", linewidth=1, zorder=1)
    plot_target(mean, cov)
    plt.savefig("2d_gaussian_mh.pdf", format="pdf")
    plt.show()
