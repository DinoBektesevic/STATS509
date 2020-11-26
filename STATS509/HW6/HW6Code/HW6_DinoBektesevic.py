import numpy as np
from scipy import stats
from scipy.stats import norm, gamma, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

np.random.seed(0)


def problem2a():
    """Plots the PDF

    f(t) = 1/sqrt(1/4-t)

    from problem 2a.
    """
    x = np.linspace(0, 0.25, 10000)
    y = 1 / ( np.sqrt(0.25 - x) )
    plt.scatter(x, y, s=3)
    plt.show()


def readNewcombData(filepath="../HW6Problems/newcomb-light-data.csv"):\
    """Reads supplied Newcombs data and returns read columns.
    """
    a = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    return a[:,0], a[:,1]


def calcPosterior(data, mu0, tau20, nu0=1, phi20=1, muGridRange=None,
                  phi2GridRange=None, muGrid=None, phi2Grid=None, nSamples=101):
    """Calculates posterior given in problem 5a by evaluating it over a grid.

    Parameters
    ----------
    data : `array`
        Data (Newcombs observed times)
    mu0 : `float`
        Mean of Newcombs normal prior for the mean
    tau20 : `float`
        STD of Newcombs normal prior for the mean
    nu0 : `float`, optional
        Nu of Newcombs gamma prior on sigma. Defaults to 1.
    phi20: `float`, optional
        Phi^2 of Newcoms gamma prior on sigma. Defaults to 1.
    muGridRange: `tuple` or None, optional
        If given places ``nSamples`` number of samples on a grid defined by
        min(muGridRange), max(muGridRange). Note either the range or the grid
        itself need to be provided.
    phi2GridRange: `tuple`, or None, optional
        If given places ``nSamples`` number of samples on a grid defined by
        min(phi2GridRange), max(phi2GridRange). Note either the range or the
        grid itself need to be provided.
    muGrid : `array` or `None`, optional
        Sampling grid.
    phi2Grid : `array` or `None`, optional
        Sampling grid.
    nSamples : `int`, optional
        Number of samples in case only grid ranges are provided. Defaults to 101

    Returns
    --------
    posterior : `array`
        Grid sampled posterior
    """
    G, H = nSamples, nSamples

    if muGrid is not None:
        muGrid = muGrid
    elif muGridRange is not None:
        muGrid = np.linspace(*muGridRange, num=G)
    else:
        raise ValueError("Supply mu grid or grid range.")

    if phi2Grid is not None:
        phi2Grid = phi2Grid
    elif phi2GridRange is not None:
        phi2Grid = np.linspace(*phi2GridRange, num=H)
    else:
        raise ValueError("Supply phi^2 grid or grid range.")

    tau0 = np.sqrt(tau20)
    phi0 = np.sqrt(phi20)
    phiGrid = np.sqrt(phi2Grid)

    posterior = np.zeros((G, H))
    for g in range(1, G):
        for h in range(1, H):
            a = norm.pdf(muGrid[g], loc=mu0, scale=1/tau0)
            b = gamma.pdf(phi2Grid[h], a=nu0/2, scale=(2*phi20)/nu0) #or chi2.pdf(phi2Grid[h], df=1)
            c = np.prod(norm.pdf(data, loc=muGrid[g], scale=1/phiGrid[h]))
            posterior[g,h] = a * b * c

    posterior = posterior/posterior.sum()

    return posterior


def plotPosterior(ax, posterior, xRange, yRange, xlabel="mu", ylabel="phi^2"):
    """Given an axis, posterior and physical ranges of the posterior plots the
    posterior streched by its physical axes.
    """
    ax.imshow(posterior.T, origin="lower", extent=(*xRange, *yRange))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xRange)
    ax.set_ylim(*yRange)
    ax.set_aspect(1./ax.get_data_ratio())
    return ax


def plotMarginal(ax, posterior, xRange, axis=0, label=''):
    """Given posterior and a physical range calculates the marginal
    of the posterior for a given axis and plots it.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Plotting axis, (unfortunate naming convention)
    posterior : `array`
        Posterior discretized on a grid.
    xRange : `array`
        Range over which maginalization will happen
    axis : `int`, optional
        Axis over which to marginalize the posterior, 0 - row-vise, 1-column-vise
    label : `str`, optional
        Plot label.
    """
    marginal = posterior.sum(axis=axis)
    x = np.linspace(*xRange, num=len(marginal))
    ax.plot(x, marginal, label=label)
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.legend()
    return ax


def problem5a():
    """Calculates the discretized posterior from problem 5. Plots the posterior
    and the two marginals.
    """
    day, time = readNewcombData()

    muGridRange = [24.75, 24.9]
    phi2GridRange = [20, 120]
    posterior = calcPosterior(time, 30, 0.01, muGridRange=muGridRange,
                              phi2GridRange=phi2GridRange)
    
    fig, ax = plt.subplots()
    ax = plotPosterior(ax, posterior, muGridRange, phi2GridRange)
    plt.show()

    mus = posterior.sum(axis=1)
    fig, ax = plt.subplots()
    ax = plotMarginal(ax, posterior, muGridRange, axis=1, label="mu")
    plt.show()

    fig, ax = plt.subplots()
    ax = plotMarginal(ax, posterior, phi2GridRange, axis=0, label="phi^2")
    plt.show()


def gibsSampler(data, nSamples=10000, mu0=30, tau20=0.01, nu0=1, phi20=1):
    """Performs Gibbs sampling as per problem 5.

    Parameters
    ----------
    data : `array`
        Data (Newcombs observed times)
    nSamples : `int`, optional
        Number of samples. Defaults to 10000
    mu0 : `float`
        Mean of Newcombs normal prior for the mean
    tau20 : `float`
        STD of Newcombs normal prior for the mean
    nu0 : `float`, optional
        Nu of Newcombs gamma prior on sigma. Defaults to 1.
    phi20: `float`, optional
        Phi^2 of Newcoms gamma prior on sigma. Defaults to 1.

    Returns
    -------
    samples : `array`
        Samples.        
    """
    mean = np.mean(data)
    var = np.var(data)
    n = len(data)

    samples = np.zeros((nSamples,2))
    samples[0,:] = np.array([mean, 1/var])
    for s in range(1, nSamples):
        prev_phi2 = samples[s-1,1]
        mustar = (mu0 * tau20 + n * mean * prev_phi2) / (tau20 + n * prev_phi2)
        phi2star = tau20 + n * prev_phi2
        mu_new = np.random.normal(mustar,1/np.sqrt(phi2star))

        astar = (nu0 + n)/2
        bstar = (nu0/phi20 + (n-1)*var + n*(mean - mu_new)**2)/2

        phi2new = np.random.gamma(astar, 1/bstar)
        samples[s,:] = np.array([mu_new, phi2new])

    return samples


def problem5b():
    """Produces discretized posterior, gibs sampled posterior plots and
    statistics as instructed in problem 5b.
    """
    day, time = readNewcombData()

    mu0 = 30
    tau20 = 0.01

    G, H = 101, 101
    nu0 = 1
    phi20 = 1

    muGridRange = [24.75, 24.9]
    phi2GridRange = [20, 120]
    muGrid = np.linspace(*muGridRange, num=G)
    phi2Grid = np.linspace(*phi2GridRange, num=G)
    posterior = calcPosterior(time, 30, 0.01, muGrid=muGrid, phi2Grid=phi2Grid)

    samples = gibsSampler(time)

    fig, ax = plt.subplots()
    ax = plotPosterior(ax, posterior, muGridRange, phi2GridRange)
    ax.scatter(samples[:,0], samples[:,1], s=1, color="white", marker='+')
    plt.show()

    print(f"mu 25th 50th and 97.5th quantiles: {np.percentile(samples[:,0], [2.5, 50, 97.5])}")
    print(f"phi^2 25th 50th and 97.5th quantiles: {np.percentile(samples[:,1], [2.5, 50, 97.5])}")
    stdev = 1/np.sqrt(samples[:,1])
    print(f"sigma_phi^2 25th 50th and 97.5th quantiles: {np.percentile(stdev, [2.5, 50, 97.5])}")

    d = 7.44373
    speedOfLight = d / (samples[:,0]*1e-6)
    speedQ25, speedQ50, speedQ975 = np.percentile(speedOfLight, [2.5, 50, 97.5])
    print("c (km/s) quantiles: %.6e, %.6e, %.6e"%(speedQ25, speedQ50, speedQ975))
    print(f"From sample mean c(km/s): {d / time.mean()*1e-6}")
    print(f"With sample std c(km/s): {d / time.std()*1e-6}")


    print(f"Measured mean time: {time.mean()}")
    print(f"Measured mean std: {time.std()}")

    plt.figure()
    plt.hist(time, bins=30)
    plt.xlabel("Time (ms)")
    plt.ylabel("Count")


    plt.show()


if __name__ == "__main__":
    problem2a()
    problem5a()
    problem5b()
