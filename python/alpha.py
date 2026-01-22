import numpy as np

from utils import Phi, Phi_inv

class GaussianCopula():
    def __init__(self, rho):
        self.rho = rho
    
    def c(self, u, v):
        u = Phi_inv(u)
        v = Phi_inv(v)
        return np.exp(-(pow(self.rho, 2) * (pow(u,2) + pow(v,2)) - 2*self.rho*u*v)/(2*(1 - pow(self.rho, 2))))/np.sqrt(1 - pow(self.rho, 2))
    
    def H(self, u, v):
        return Phi((Phi_inv(u) - self.rho*Phi_inv(v))/np.sqrt(1 - self.rho**2))
    
    def __call__(self, u, v):
        return self.c(u, v)
    
class GaussianCopulaD():
    def __init__(self, rho):
        self.rho = rho
    
    def c(self, x1, x2):
        return np.exp(-(pow(self.rho, 2) * (pow(x1,2) + pow(x2,2)) - 2*self.rho*x1*x2)/(2*(1 - pow(self.rho, 2))))/np.sqrt(1 - pow(self.rho, 2))
    
    def H(self, x1, x2):
        return Phi((x1 - self.rho*x2)/np.sqrt(1 - self.rho**2))
    
    def __call__(self, x1, x2):
        return self.c(x1, x2)
    
class TrivialAlpha():
    def __init__(self, alpha, d=None):
        self.d = None
        self.alpha = alpha
    
    def __call__(self, i, x1=None, x2=None):
        return self.alpha/(i+1)

class Alpha():
    def __init__(self, alpha, d):
        self.d = d
        self.alpha = alpha
    
    def __call__(self, i, x1, x2):
        alpha_i = self.alpha/(i+1)
        nume = alpha_i * self.d(x1, x2)
        return nume/(1 - alpha_i + nume)