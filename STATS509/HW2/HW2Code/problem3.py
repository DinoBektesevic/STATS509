import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def calcEmpiricalCDF(x, data):
    """Given a set of points and samples calculates the empirical CDF. CDF is
    calculated by counting the proportion of the given data that is smaller or
    equal to each point in the given array. 

    Parameters
    ----------
    x : `np.array`
        Array of points at which CDF will be evaluated.
    data : `np.array`
        Samples from which the CDF will be calculated.

    Returns
    -------
    empiricalCDF : `np.array`
        Empirical CDF evaluated at the given points.
    """
    empiricalCDF = []
    for i in x:
        empiricalCDF.append(np.sum(data <= i))
    empiricalCDF = np.array(empiricalCDF)/len(data)
    return empiricalCDF


def exactUniformCDF(x):
    """Evaluates the exact Uniform distribution CDF on [0,1] at the given points.

    Parameters
    ----------
    x : `np.array`
        Array of points at which CDF will be evaluated.

    Returns
    -------
    exactCDF : `np.array`
        Exact CDF evaluated at the given points.
    """
    rv = stats.uniform()
    return rv.cdf(x)


def plot3c(x, data):
    """Plots problem 3 using the given points and data.

    Parameters
    ----------
    x : `np.array`
        Array of points at which CDF will be evaluated.
    data : `np.array`
        Samples from which the CDF will be calculated.
    """
    data = np.array([0.03, 0.11, 0.42, 0.44, 0.47, 0.66, 0.75, 0.88, 0.89, 0.90])
    x = np.linspace(0, 1, 1000)

    empiricalCDF = calcEmpiricalCDF(x, data)
    exactCDF = exactUniformCDF(x)

    fig, ax = plt.subplots()

    ax.scatter(x, empiricalCDF, color="black", label="Empirical CDF")
    ax.plot(x, exactCDF, "darkgray", label="Exact CDF", linewidth=3)

    ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("$F(x)")
    plt.show()


def kolmogorovSmirnov(x, data):
    """Performs Kolmogorov-Smirnov test of a Uniform distribution on [0,1] as
    instructed by problem 3.

    Parameters
    ----------
    x : `np.array`
        Array of points at which CDF will be evaluated.
    data : `np.array`
        Samples from which the CDF will be calculated.

    """
    empiricalCDF = calcEmpiricalCDF(x, data)
    exactCDF = exactUniformCDF(x)

    cdfDiff = np.abs(empiricalCDF - exactCDF)

    idx = np.argmax(cdfDiff)
    ksTest = cdfDiff[idx]
    xMax = x[idx]

    print(f"KS statistic K={ksTest} at point t={xMax}")


if __name__ == "__main__":
    # problem 3 setup
    data = np.array([0.03, 0.11, 0.42, 0.44, 0.47, 0.66, 0.75, 0.88, 0.89, 0.90])
    x = np.linspace(0, 1, 1000)

    plot3c(x, data)
    kolmogorovSmirnov(x, data)
