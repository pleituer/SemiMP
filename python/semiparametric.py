import numpy as np

class SemiparametricRegressionMP():
    def __init__(self, ms_0, n_moments, rho, alpha):
        self.ms_0 = ms_0 ## moment init funcs (one func x -> (m1, m2, ...))
        self.n_moments = n_moments

        self.n = 0
        self.x = None
        self.y = None

        self.mu = np.zeros(0)
        self.sigma = np.zeros(0)
        self.ms = np.zeros((self.n_moments, 0))
        self.zs = np.zeros(0)

        self.rho = rho
        self.alpha = alpha
    
    def _old_update(self, ms, mu, sigma, n):
        ## m^k_n <- m^k_{n-1} + alpha * mk_update(mk, mu, sigma2, n)
        _C = (1 - self.rho**2) * sigma**2
        m1_update = mu + self.rho * sigma * self.zs[n-1]
        m2_update = m1_update**2 + _C
        m3_update = m1_update**3 + 3 * _C * m1_update 
        return np.array([m1_update, m2_update, m3_update]) - ms
    
    def _update(self, ms, mu, sigma, n):
        _mu = mu + self.rho * sigma * self.zs[n-1]
        _sigma2 = (1 - self.rho**2) * sigma**2
        _v = sigma/self.sigma[n-1]
        _kappa3 = _v**3 * (self.ms[2, n-1] - 3*self.mu[n-1]*self.sigma[n-1]**2 - self.mu[n-1]**3)
        m1_update = _mu
        m2_update = _mu**2 + _sigma2
        m3_update = _mu**3 + 3*_mu*_sigma2 - (mu**3 + 3*mu*sigma**2 + self.rho**3 * _kappa3)
        return np.array([m1_update, m2_update, m3_update]) - ms
    
    def update(self, ms, mu, sigma2, n):
        return self._update(ms, mu, sigma2, n)
    
    def set_update(self, new_update):
        ## first 2 must be first moment and second moment
        self.update = new_update
    
    def reset_update(self):
        self.update = self._update
    
    def train(self, x, y):
        self.n = len(y)
        self.x = x.copy()
        self.y = y.copy()

        self.mu = np.zeros((self.n, ))
        self.sigma = np.zeros((self.n,))
        self.ms = np.zeros((self.n_moments, self.n))
        self.zs = np.zeros((self.n,))

        self.ms = self.ms_0(x)
        self.mu = self.ms[0]
        self.sigma = np.sqrt(self.ms[1] - self.ms[0]**2)

        for n in range(1, self.n):
            alpha_n = self.alpha(n, x[n:], x[n-1])
            self.zs[n-1] = (y[n-1] - self.mu[n-1])/self.sigma[n-1]
            self.ms[:, n:] = self.ms[:, n:] + alpha_n * self.update(self.ms[:, n:], self.mu[n:], self.sigma[n:], n)
            self.mu[n:] = self.ms[0, n:]
            self.sigma[n:] = np.sqrt(self.ms[1, n:] - self.ms[0, n:]**2)

        self.zs[-1] = (y[-1] - self.mu[-1])/self.sigma[-1]

    def get(self, x_test, n=None):
        ## x_test is one dimensional
        if n is None: n = self.n
        n = min(self.n, n)

        mu = np.zeros((n+1, len(x_test)))
        sigma = np.zeros((n+1, len(x_test)))
        ms = np.zeros((self.n_moments, n+1, len(x_test)))

        ms[:, 0] = self.ms_0(x_test)
        mu[0] = ms[0, 0]
        sigma[0] = np.sqrt(ms[1, 0] -  ms[0, 0]**2)

        for i in range(1, n+1):
            alpha_i = self.alpha(i, x_test, self.x[i-1]).flatten()
            ms[:, i] = ms[:, i-1] + alpha_i * self.update(ms[:, i-1], mu[i-1], sigma[i-1], i)
            mu[i] = ms[0, i]
            sigma[i] = np.sqrt(ms[1, i] - ms[0, i]**2)
        
        return {
            "mu": mu,
            "sigma": sigma,
            "sigma2": sigma**2,
            "moment": ms
        }