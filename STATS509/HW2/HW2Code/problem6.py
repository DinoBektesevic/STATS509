import scipy.stats as stats
import numpy as np
np.random.seed(0)


def simulate(nSimulations, p):
    """Runs n simulations, where each simulation is a double coin toss, keeping
    track of how many attempts are requires untill the first case of HT or TH
    is observed.

    Parameters
    ----------
    nSimulations : `int`
        Number of simulations to run
    p : `float`
        Probability that a coin toss is either heads or tails.

    Returns
    -------
    nAttempts : `array`
        Number of coin tosses that were required until HT or TH were tossed.
    """
    nAttempts = []
    for sim in range(nSimulations):
        nTries = 0
        while True:
            flip1 = stats.binom.rvs(1, p)
            flip2 = stats.binom.rvs(1, p)
            nTries += 2
            if flip1 != flip2:
                break
        nAttempts.append(nTries)
    return nAttempts


if __name__ == "__main__":
    nSimulations = 1000
    p = 0.2
    nAttempts = simulate(nSimulations, p)
    print(f"Average number of coin tosses, across all simulations, until success was {np.mean(nAttempts)}")
