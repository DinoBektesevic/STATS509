import scipy.stats as stats
import numpy as np

# a)
normalDraws = stats.norm.rvs(loc=5, scale=1, size=1000)


# b)
posExpDraws = np.exp(0.01*normalDraws)
print(f"Empirical expectation value is {np.mean(posExpDraws)}")


# c)
negExpDraws = np.exp(-0.01*normalDraws)
print(f"Empirical expectation value is {np.mean(negExpDraws)}")


# d)
partialExp = (posExpDraws - negExpDraws)/(2*0.01)
print(f"Empirical expectation value is {np.mean(partialExp)}")
