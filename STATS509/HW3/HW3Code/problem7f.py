import scipy.stats as stats
import numpy as np
k = 3
lambd = 2
print(f"Upper bound as estimated in problem c: {1/k}")
print(f"Upper bound as estimated in problem E: {np.exp(lambd*k*(1-np.log(k)))/np.e}")
print(f"Upper limit as evaluated via CDF: {1-stats.poisson(k, lambd).cdf(6)}")
