import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nonparametric import NonparametricMP
from alpha import TrivialAlpha

from utils import phi, Phi
from dataset import Galaxies

data = Galaxies()

alpha = TrivialAlpha(alpha=0.25)
rho = 0.93

mp = NonparametricMP(p0=phi, P0=Phi, rho=rho, alpha=alpha)
mp.train(data.data)

y_linspace = np.linspace(data.data.min()-1, data.data.max()+1, 200)

sim_n = len(data)
_, ps = mp.get(y_linspace, n=sim_n)
sns.scatterplot(x=data.data, y=0.01, s=10, color="k", label="Data")
sns.kdeplot(data.data[:sim_n], label="KDE", linestyle=":")
plt.plot(y_linspace, ps[-1], c="tab:orange", label="MP Estimate")
plt.xlabel("y")
plt.legend()
plt.show()