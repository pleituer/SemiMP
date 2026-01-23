import numpy as np
import pandas as pd
from pydataset import data

from utils import SEED

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
        self.data_df = (self.data_df - self.data_df.mean()) / self.data_df.std()

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
    def __init__(self, path=None):
        super().__init__(path, cols=["y"])
        self.data = self.data.flatten()
    
    def load(self):
        self.data_df = data("galaxies")

    def _old_standardize(self):
        self.data_df /= 1000