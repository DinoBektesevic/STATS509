import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


def problem1c(n = 1000000):
    """Performs n simulations of 5 coin tosses and returns the total winnings
    earned in each simulation. Win earnings scale as  2/3**k where k is the
    toss count, i.e. integer from 1 to 5, for each simulation.

    Parameters
    ----------
    n : `int`
        Number of simulations to run, a simulation being 5 random draws from
        the set [0,1].

    Returns
    -------
    outcome : `np.array`
        a list of n pmf values corresponding to each simulation.
    """
    # a matrix in which rows are simulations and columns are tosses
    sims = np.random.randint(2, size=(n, 5))

    # covnert tosses of each simulation into individual earnings by multiplying
    # column-wise with the earnings for that toss, i.e. 2/3**k
    tossEarnings = []
    for k in range(1, 6):
        tossEarnings.append(sims[:,k-1].astype(float) * 2/3**k)

    # add all individual toss earnings into a total earning per simulation.
    simEarnings = np.column_stack(tossEarnings)
    return simEarnings.sum(axis=1)


def plot1c():
    """Displays plots for problem 1c."""
    n = 1000000
    simEarnings = problem1c(n)
    possibleEarnings = set(simEarnings)

    fig, axes = plt.subplots(2,1)

    # PMF
    for earning in possibleEarnings:
        occurenceRate = np.sum(simEarnings == earning) / n
        axes[0].bar(earning, occurenceRate, width=0.005, color='darkgray')
        axes[1].bar(earning, occurenceRate, width=0.005, color='darkgray')
    # replot the last one again to get a legend entry correct
    axes[0].bar(earning, occurenceRate, width=0.005, color='darkgray', label="pmf")

    # CDF evaluated at equaly space points within the domain
    xs = np.linspace(0, max(simEarnings), 1000)
    cumsum = []
    for x in xs:
        cumsum.append(np.sum(simEarnings <= x))
    cumsum = np.array(cumsum)/n
    axes[1].plot(xs, cumsum, 'black', label="cdf")

    for ax in axes:
        ax.set_xlabel("$X_5$")
        ax.legend()
    axes[0].set_ylabel("Occurence Rate")
    plt.ylabel("Proportion of observations $\leq x$")
    plt.show()


if __name__ == "__main__":
    plot1c()
