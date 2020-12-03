import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def problem1():
    n, p = 25, 0.89
    nTrials = 10000
    means = []
    for i in range(nTrials):
        x = stats.bernoulli.rvs(p, size=n)
        means.append(x.mean())

    means = np.array(means)
    print("Problem 1:")
    print(f"    Mean of means: {means.mean()}")
    print(f"    Variance of means: {means.std()**2}")
    print(f"    Mean squared error: {np.mean((means-p)**2)}")

    plt.hist(means)
    plt.xlabel(r"$\bar X$")
    plt.show()


def problem2():
    n, lambd = 25, 4
    nTrials = 10000
    means = []
    for i in range(nTrials):
        x = stats.poisson.rvs(lambd, size=n)
        means.append(x.mean())

    means = np.array(means)
    print("Problem 2")
    print(f"    Mean of means: {means.mean()}")
    print(f"    Variance of means: {means.std()**2}")
    print(f"    Mean squared error: {np.mean((means-lambd)**2)}")

    plt.hist(means)
    plt.xlabel(r"$\bar X$")
    plt.show()


def problem4c():
    mu, sigma = 0.3, 4
    n = 36

    xi = lambda ni: sigma * 1.6499/np.sqrt(ni)
    power = lambda n_i: 1 - stats.norm.cdf(xi(n_i), loc=mu, scale=sigma/np.sqrt(n_i))

    while power(n) < 0.9:
        n += 1

    print(f"n = {n}, Power = {power(n)}")#, end='\r')


if __name__ == "__main__":
    #problem1()
    #problem2()
    problem4c()
