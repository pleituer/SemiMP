import numpy as np

from alpha import GaussianCopula

class NonparametricMP():
    def __init__(self, p0, P0, rho, alpha):
        self.p0 = p0
        self.P0 = P0

        self.n = 0
        self.y = None

        self.Py = np.zeros(0)

        self.copula = GaussianCopula(rho=rho)
        self.alpha = alpha
    
    def train(self, y):
        self.n = len(y)
        self.y = y.copy()

        self.Py = np.zeros((self.n,))
        self.Py = self.P0(self.y)

        for n in range(1, self.n):
            alpha_n = self.alpha(n)
            self.Py[n:] = (1 - alpha_n)*self.Py[n:] + alpha_n*self.copula.H(self.Py[n:], self.Py[n-1])
    
    def get(self, y_test, n=None):
        if n is None: n = self.n
        n = min(self.n, n)

        Ps = np.zeros((n+1, len(y_test)))
        ps = np.zeros((n+1, len(y_test)))

        Ps[0] = self.P0(y_test)
        ps[0] = self.p0(y_test)

        for i in range(1, n+1):
            alpha_i = self.alpha(i)
            Ps[i] = (1 - alpha_i)*Ps[i-1] + alpha_i*self.copula.H(Ps[i-1], self.Py[i-1])
            ps[i] = (1 - alpha_i + alpha_i*self.copula.c(Ps[i-1], self.Py[i-1]))*ps[i-1]
        
        return Ps, ps
    
class NonparametricRegressionMP():
    def __init__(self, p0, P0, rho, alpha):
        self.p0 = p0
        self.P0 = P0

        self.n = 0
        self.x = None
        self.y = None

        self.Py = np.zeros(0)

        self.copula = GaussianCopula(rho=rho)
        self.alpha = alpha
    
    def train(self, x, y):
        self.n = len(y)
        self.x = x.copy()
        self.y = y.copy()

        self.Py = np.zeros((self.n,))
        self.Py = self.P0(x, y)

        for n in range(1, self.n):
            alpha_n = self.alpha(n, x[n:], x[n-1])
            self.Py[n:] = (1 - alpha_n)*self.Py[n:] + alpha_n*self.copula.H(self.Py[n:], self.Py[n-1])
    
    def get(self, x_test, y_test, n=None):
        ## x_test.shape = y_test.shape = (len(test), )
        if n is None: n = self.n
        n = min(self.n, n)

        Ps = np.zeros((n+1, len(y_test)))
        ps = np.zeros((n+1, len(y_test)))

        Ps[0] = self.P0(x_test, y_test) 
        ps[0] = self.p0(x_test, y_test)

        for i in range(1, n+1):
            alpha_i = self.alpha(i, x_test, self.x[i-1])
            Ps[i] = (1 - alpha_i)*Ps[i-1] + alpha_i*self.copula.H(Ps[i-1], self.Py[i-1])
            ps[i] = (1 - alpha_i + alpha_i*self.copula.c(Ps[i-1], self.Py[i-1]))*ps[i-1]
        
        return Ps, ps
    
    def sample(self, x_test, sample_size=67, n=None):
        pass
