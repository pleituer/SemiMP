import numpy as np
import pandas as pd
from pydataset import data

from utils import SEED, tukeysG_fromMeanVarSkew, sample_tukeysG

class Dataset():
    def __init__(self, path, cols=None):
        self.path = path
        self.rng = np.random.default_rng(seed=SEED)

        self.load()
        if cols is not None: self.data_df.columns = cols
        self.standardize()
        self.data = self.data_df.to_numpy()
    
    def __len__(self):
        return len(self.data) 
    
    def load(self):
        self.data_df = pd.read_csv(self.path)
    
    def standardize(self):
        self.data_df = (self.data_df - self.data_df.mean()) / np.where(self.data_df.std() == 0, 1, self.data_df.std())

    def set_rng(self, seed):
        self.rng = np.random.default_rng(seed=seed)
    
    def reset_rng(self):
        self.rng = np.random.default_rng(seed=SEED)
    
    def shuffle(self):
        self.data = self.rng.permutation(self.data)

class LIDAR(Dataset):
    def __init__(self, path=None):
        if path is None: path = "datasets/lidar.csv"
        super().__init__(path, cols=["x", "y"])
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
    
    def shuffle(self):
        self.data = self.rng.permutation(self.data)
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]

class SamPoker(Dataset):
    def __init__(self, path=None):
        ## Example dataset form a friend
        if path is None: path = "datasets/sampoker.csv"
        super().__init__(path, cols=["y"])
        self.data = self.data.flatten()

class Galaxies(Dataset):
    def __init__(self):
        super().__init__(path=None, cols=["y"])
        self.data = self.data.flatten()
    
    def load(self):
        self.data_df = data("galaxies")

    def _old_standardize(self):
        self.data_df /= 1000

class SyntheticDataset(Dataset):
    def __init__(self, mean=(0, 1), variance=(0, 1), skewness=(0, 0), mean_shape=("constant", {}), variance_shape=("constant", {}), skewness_shape=("constant", {})):
        self.rng = np.random.default_rng(seed=SEED)

        ## amplitude doesnt matter cuz standardizing, we can assume E[X] = E[Y] = 0, Var(X) = Var(Y) = 1 or 0
        ## x has support [-1, 1]
        self.trend_shape = {
            "constant": lambda x, **_: np.zeros(x.shape),
            "poly": lambda x, degree=1, **_: x**degree,
            "sine": lambda x, freq=np.pi, **_: np.sin(freq * x)
        }
        mean_type, mean_kwargs = mean_shape
        variance_type, variance_kwargs = variance_shape
        skewness_type, skewness_kwargs = skewness_shape
        
        mean_offset, mean_scale = mean
        variance_offset, variance_scale = variance
        skewness_offset, skewness_scale = skewness

        self.mean = lambda x: self.trend_shape[mean_type](x, **mean_kwargs) * mean_scale + mean_offset
        self.variance = lambda x: np.abs(self.trend_shape[variance_type](x, **variance_kwargs) * variance_scale + variance_offset)
        self.skewness = lambda x: self.trend_shape[skewness_type](x, **skewness_kwargs) * skewness_scale + skewness_offset

    def load(self):
        pass

    def sample(self, n):
        self.x = np.random.uniform(-1, 1, size=(n,))
        means = self.mean(self.x)
        variances = self.variance(self.x)
        skewnesses = self.skewness(self.x)

        mus, sigmas, gs = tukeysG_fromMeanVarSkew(means, variances, skewnesses)
        samples = sample_tukeysG(mus, sigmas, gs, size=(n,))

        self.data_df = pd.DataFrame(data={
            "x": self.x,
            "y": samples
        })
        self.standardize()
        self.data = self.data_df.to_numpy()
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]

    def shuffle(self):
        self.data = self.rng.permutation(self.data)
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]