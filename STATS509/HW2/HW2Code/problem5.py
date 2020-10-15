import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
np.random.seed(0)


def simulate(nSamples, randVar):
    """Given n number of samples m, performs n simulations drawing m samples
    each time calculating mean and the variance.

    Parameters
    ----------
    nSamples : `array`
        number of samples to draw for each simulation, where the number of sims
        is the length of the array.
    randVar : `scipy.stats.rv_frozen`
        Random variable from which draws will be made.

    Returns
    ------
    expectations : `array`
        calculated empirical expectation values for each simulation
    variances : `array`
        calculated empirical variance values for each simulation
    """
    expectation, variance = [], []

    for nSample in nSamples:
        draws = randVar.rvs(nSample)
        expectation.append(np.mean(draws))
        variance.append(np.var(draws))

    return expectation, variance


def plot(nSamples, expectations, variances, exactE, exactVar, title=""):
    """Plots the given empirical and exact expectation and variance values.

    Parameters
    ----------
    nSamples : `array`
        number of samples to draw each simulation.
    expectations : `array`
        calculated empirical expectation values
    variances : `array`
        calculated empirical variance values
    exactE : `float`
        exact expectation value
    exactVar : `float`
        exact variance value
    title : `str`, optional
        Title of the plot
    """
    fig, axes = plt.subplots(2, 1)

    axes[0].semilogx(nSamples, expectations, color="black", label="Empirical E(X)")
    axes[0].axhline(exactE, color="darkgray", label="Exact E(X)")

    axes[1].semilogx(nSamples, variances, color="black", label="Empirical Var(X)")
    axes[1].axhline(exactVar, color="darkgray", label="Exact Var(X)")

    for ax in axes:
        ax.set_xlabel("N draws in the simulation")
        ax.legend()
    axes[0].set_ylabel("E(X)")
    axes[0].set_title(title)
    axes[1].set_ylabel("Var(X)")
    plt.show()


def plot5a(nSamples):
    """Runs n simulations with m draws each and plots the calculated and exact
    values for Poisson distribution.

    Parameters
    ----------
    nSamples : `array`
        number of samples to draw each simulation.
    """
    rv = stats.poisson(2.0)
    expectations, variances = simulate(nSamples, rv)
    plot(nSamples, expectations, variances, 2.0, 2.0, "Poisson distribution")


def plot5b(nSamples):
    """Runs n simulations with m draws each and plots the calculated and exact
    values for Binomial distribution.

    Parameters
    ----------
    nSamples : `array`
        number of samples to draw each simulation.
    """
    rv = stats.binom(400, 0.3)
    expectations, variances = simulate(nSamples, rv)
    plot(nSamples, expectations, variances, 120, 84, "Binomial distribution")


if __name__ == "__main__":
    nSamples = [10, 100, 1000, 10000, 1000000]
    plot5a(nSamples)
    plot5b(nSamples)
