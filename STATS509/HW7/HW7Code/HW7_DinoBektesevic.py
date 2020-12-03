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



if __name__ == "__main__":
    problem1()
    problem2()
