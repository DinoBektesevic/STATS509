import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
np.random.seed(0)


def problem2(n=500, lambd=2):
    """Instantiates an scaled exponential random variable, draws n samples from
    it and calculates an empirical CDF at a 1000 evenly spaced points in the
    range of the drawn samples.

    Parameters
    ----------
    n : `int`
        number of samples drawn from the distribution
    lambd : `float`
        The rate parameter ``lambda*exp(-lambda*x)`` from which scale will be
        calculated from as ``/lambda``

    Returns
    -------
    x : `np.array`
        Points at which empirical CDF was evaluated at
    empiricalCDF : `np.array`
        Empirical CDF evaluated for points in ``x``

    """
    scale = 1/lambd
    rv = stats.expon(scale=scale)
    expSamples = rv.rvs(size=n)

    x = np.linspace(0, max(expSamples), 1000)
    empiricalCDF = []
    for i in x:
        empiricalCDF.append(np.sum(expSamples <= i))

    empiricalCDF = np.array(empiricalCDF)/n
    exactCDF = rv.cdf(x)

    return x, empiricalCDF, exactCDF


def plot2():
    """Plots problem 2."""
    fig, ax = plt.subplots()

    x, empiricalCDF, exactCDF = problem2()

    ax.plot(x, empiricalCDF, 'black', label="Empirical CDF")
    ax.plot(x, exactCDF, 'darkgray',  label="Exact CDF")

    ax.set_xlabel("x")
    ax.set_ylabel("Proportion of observations $\leq x$")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plot2()
