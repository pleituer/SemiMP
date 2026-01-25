import numpy as np
from scipy.stats import norm
from tqdm import tqdm

import time

SEED = 67

Phi = norm.cdf
Phi_inv = norm.ppf
phi = norm.pdf

def timer(func, perm_n=10):
    start = time.perf_counter()
    for i in tqdm(range(perm_n)): func(i)
    end = time.perf_counter()
    time_spent = end - start
    print(f"Total Time: {time_spent:.4f} seconds, {time_spent/perm_n:.4f} seconds per iter.")

def tukeysG_fromMeanVarSkew(mean, variance, skewness):
    ## np.divide to avoid division by zero
    A = 54 + 27 * skewness**2 + np.sqrt(2916 * skewness**2 + 729 * skewness**4)
    exp_g2 = -1 + 3*np.cbrt(2/A) + np.cbrt(A/2)/3
    g = np.sqrt(np.log(exp_g2)) * np.sign(skewness)
    sigma = np.sqrt(np.divide(variance * g**2, exp_g2 * (exp_g2 - 1), out=np.sqrt(variance), where=(g!=0))) 
    mu = mean - sigma * np.divide(np.sqrt(exp_g2) - 1, g, out=np.zeros_like(g), where=(g!=0))
    return mu, sigma, g