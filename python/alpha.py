import numpy as np

from utils import Phi, Phi_inv

################################ Copulas ################################

class GaussianCopula():
    def __init__(self, rho):
        self.rho = rho
    
    def c(self, u, v):
        u = Phi_inv(u)
        v = Phi_inv(v)
        return np.exp(-(self.rho**2 * (u**2 + v**2) - 2*self.rho*u*v)/(2*(1 - self.rho**2)))/np.sqrt(1 - self.rho**2)
    
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
    
################################ Alpha ################################

def true_alpha_update(alpha, i):
    return (alpha - 1/i) / (i+1)

class TrivialAlpha():
    def __init__(self, alpha, d=None, alpha_updater=None):
        self.d = None
        self.alpha = alpha
        if alpha_updater is None: self.alpha_update = lambda alpha, i: alpha/(i+1)
        else: self.alpha_update = alpha_updater
    
    def __call__(self, i, x1=None, x2=None):
        return self.alpha_update(self.alpha, i)

class Alpha():
    def __init__(self, alpha, d, alpha_updater=None):
        self.d = d
        self.alpha = alpha
        if alpha_updater is None: self.alpha_update = lambda alpha, i: alpha/(i+1)
        else: self.alpha_update = alpha_updater
    
    def __call__(self, i, x1, x2):
        alpha_i = self.alpha_update(self.alpha, i)
        nume = alpha_i * self.d(x1, x2)
        return nume/(1 - alpha_i + nume)