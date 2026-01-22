import pandas as pd
from pydataset import data

class Dataset():
    def __init__(self, path, cols=None, shuffle=True):
        self.path = path
        self.load()
        if shuffle: self.data_df = self.data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        if cols is not None: self.data_df.columns = cols
        self.standardize()
        self.data = self.data_df.to_numpy()
    
    def __len__(self):
        return len(self.data)
    
    def load(self):
        self.data_df = pd.read_csv(self.path)
    
    def standardize(self):
        self.data_df = (self.data_df - self.data_df.mean()) / self.data_df.std()

class LIDAR(Dataset):
    def __init__(self, path=None, shuffle=True):
        if path is None: path = "datasets/lidar.csv"
        super().__init__(path, cols=["x", "y"], shuffle=shuffle)
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]

class SamPoker(Dataset):
    def __init__(self, path=None, shuffle=True):
        ## Example dataset form a friend
        if path is None: path = "datasets/sampoker.csv"
        super().__init__(path, cols=["y"], shuffle=shuffle)
        self.data = self.data.flatten()

class Galaxies(Dataset):
    def __init__(self, path=None, shuffle=True):
        super().__init__(path, cols=["y"], shuffle=shuffle)
        self.data = self.data.flatten()
    
    def load(self):
        self.data_df = data("galaxies")

    #def standardize(self):
    #    self.data_df /= 1000