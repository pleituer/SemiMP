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