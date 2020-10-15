import scipy.stats as stats
import numpy as np
np.random.seed(0)


def simulate(nSimulations, p):
    nSims = 0

    nAttempts = []
    for sim in range(nSimulations):
        nTries = 1
        flip1, flip2 = 0, 1
        while (flip1 != flip2):
            # stats.binom.rvs(1, 0.2, size=1000)
            flip1 = stats.binom.rvs(1, 0.2)
            flip2 = stats.binom.rvs(1, 0.2)
            nTries += 1
        nAttempts.append(nTries)
    return nAttempts


if __name__ == "__main__":
    nSimulations = 1000
    p = 0.2
    nAttempts = simulate(nSimulations, p)
    print(f"Average number of attempts to reach success for all simulations was {np.mean(nAttempts)}")
