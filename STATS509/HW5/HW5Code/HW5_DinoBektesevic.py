import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS as colors

import statsmodels.api as sm


def readData(fpath="../HW5Problems/pres-election-2016-pop-density.csv"):
    """Reads the election data and returns the log10 of population density
    and ln of party vote ratio per county.

    Parameters
    ----------
    fpath : `str`
        Path to file containing data

    Returns
    -------
    logPopDens : `numpy.array`
        Log 10 of population density per county
    lnPartyRatio : `numpy.array`
        Ln of vote ratios per county
    """
    data = np.genfromtxt(fpath, dtype=float, delimiter=',', names=True)
    logPopDens = data['log10popdens']
    lnPartyRatio = data['logpartyratio']
    return logPopDens, lnPartyRatio


def problem5a(x, y):
    """Fits a linear predictor to the data.

    Parameters
    ----------
    x : `np.array`
        x coordinates of data
    y : `np.array`
        y coordinates of data

    Returns
    -------
    predictor : `callable`
        Function that given x predicts y, a linear predictor.
    """
    fitCoeffs = np.polyfit(x, y, 1)
    predictor = np.poly1d(fitCoeffs)
    return predictor


def problem5b(x, y):
    """Computes binned mean statistics in range -2.5 to 4.5 with bin-widths of
    0.5.

    Parameters
    ----------
    x : `np.array`
        x coordinates of data
    y : `np.array`
        y coordinates of data

    Returns
    -------
    historgram : `scipy.BinnedStatisticsResult`
        A ordered dict containing the computed statistic, bin edges and bin
        numbers for the given data.
    """
    bins = np.arange(-2.5, 4.55, step=0.5)
    histogram = stats.binned_statistic(x, y, bins=bins, statistic="mean")
    return histogram


def problem5c(x, y):
    """Computes a nonparametric LOESS smoothed predictor over conditioned data
    y given x.

    Parameters
    ----------
    x : `np.array`
        x coordinates of data
    y : `np.array`
        y coordinates of data

    Returns
    -------
    lowess : `np.array`
        A 2D array where first column are sorted x coordinates of the data and
        second column are the returned fitted values.
    """
    uniqueX = np.unique(x)
    conditioned = np.zeros(uniqueX.shape)
    for i, xi in enumerate(uniqueX):
        conditioned[i] = np.mean(y[np.where(x == xi)])

    return sm.nonparametric.lowess(conditioned, uniqueX)


def plotAll():
    """Plots problem 5 a-c on the same plot. """
    x, y = readData()

    uniqueX = np.unique(x)
    sampleX = np.linspace(min(x), max(x), 1000)

    linPredictor = problem5a(x, y)
    means, binEdges, _ = problem5b(x, y)
    lowess = problem5c(x, y)

    plt.scatter(x, y, color='gray', label="Data", alpha=0.2)
    plt.plot(sampleX, linPredictor(sampleX), color=colors["tab:blue"], label="Best Linear Predictor")
    plt.plot(binEdges[1:]-0.25, means, color=colors["tab:orange"], label="Binned mean")
    #plt.bar(binEdges[:-1], means, width=0.5, align='edge', color=colors["tab:orange"], alpha=0.2, label="Binned means")
    plt.plot(uniqueX, lowess[:, 1], color=colors["tab:green"], label="LOESS smoothed fit")

    plt.ylabel("ln(PartyRatio)")
    plt.xlabel("log(PopDensity)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plotAll()

