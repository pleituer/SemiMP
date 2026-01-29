import numpy as np

from utils import sample_tukeysG, tukeysG_fromMeanVarSkew

class _SemiparametricRegressionMP():
    """
    A container for the means, variations, moments, and z_n's of a MP model
    """
    def __init__(self, n_moments, static=True):
        """
        :param n_moments: number of moments to estimate
        :param static: uses numpy array if True, otherwise python lists
        """
        if static:
            self.mu = np.zeros(0)
            self.sigma = np.zeros(0)
            self.ms = np.zeros((n_moments, 0))
            self.zs = np.zeros(0)
        else:
            self.mu = []
            self.sigma = []
            self.ms = [[] for _ in range(n_moments)]
            self.zs = []

class SemiparametricRegressionMP():
    """
    Semiparametric Regression MP model
    """
    def __init__(self, ms_0, n_moments, rho, alpha):
        """
        :param ms_0: Initializing function for moments, first two must be first and second moment, maps x -> [m1, m2, ...]
        :param n_moments: number of moments to estimate, aka the length of the output of ms_0
        :param rho: rho (see MP)
        :param alpha: alpha_n(x, x'), maps n, x, x' -> [0, 1]
        """
        self.ms_0 = ms_0 
        self.n_moments = n_moments

        self.n = 0
        self.x = None
        self.y = None

        self._train = _SemiparametricRegressionMP(n_moments, static=True)

        ## Predictive resampling
        self.pn = 0
        self.px = []
        self.py = []
        self._pr = _SemiparametricRegressionMP(n_moments, static=False)

        self.rho = rho
        self.alpha = alpha
    
    def _old_update(self, n, ms, mu, sigma, _mp):
        """
        Update rule for first 3 moments, directly derived without modification
        """
        _C = (1 - self.rho**2) * sigma**2
        m1_update = mu + self.rho * sigma * _mp.zs[n-1]
        m2_update = m1_update**2 + _C
        m3_update = m1_update**3 + 3 * _C * m1_update 
        return np.array([m1_update, m2_update, m3_update]) - ms
    
    def _update(self, n, ms, mu, sigma, _mp):
        """
        Update rule for first 3 moments, with correction to satisfy the martingale property
        """
        _mu = mu + self.rho * sigma * _mp.zs[n-1]
        _sigma2 = (1 - self.rho**2) * sigma**2
        _v = sigma/_mp.sigma[n-1]
        _kappa3 = _v**3 * (_mp.ms[2][n-1] - 3*_mp.mu[n-1]*_mp.ms[1][n-1] + 2*_mp.mu[n-1]**3)
        m1_update = _mu
        m2_update = _mu**2 + _sigma2
        m3_update = _mu**3 + 3*_mu*_sigma2 - (mu**3 + 3*mu*sigma**2 + self.rho**3 * _kappa3)
        return np.array([m1_update - ms[0], m2_update - ms[1], m3_update])
    
    def _sample(self, x, size=(1,), n=None, predictive_n=0):
        """
        Samples based on first 3 moments, uses Tukey's g or Shifted Lognormal distribution. `size` represents shape for **each individual** x
        """
        theta_fromMoments = tukeysG_fromMeanVarSkew
        sample_fromTheta = sample_tukeysG

        results = self.get(x, n=n, predictive_n=predictive_n)
        _means, _, _variances, _moments = results.values()
        means, variances, moments = _means[-1], _variances[-1], _moments[:, -1]

        skewnesses = (moments[2] - 3*moments[1]*moments[0] + 2*moments[0]**3)/(np.sqrt(variances)**3)

        return sample_fromTheta(*theta_fromMoments(means, variances, skewnesses), size=(*x.shape, *size))

    def update(self, n, ms, mu, sigma, _mp=None):
        """
        Wrapper for the update rule implementations. The update rules for each k-th moment is implemented as:
        
        **m^k_n <- m^k_{n-1} + self.alpha(...) * self.update(...)**
        
        :param n: n-th iteration
        :param ms: moments to compute, first two row must be first and second moments
        :param mu: mean to compute
        :param sigma: sd to compute
        :param _mp: a container containing means, variance, moments, z_n. By default `self._train` if None
        """
        if _mp == None: _mp = self._train
        return self._update(n, ms, mu, sigma, _mp)
    
    def sample(self, x, size=(1,), n=None, predictive_n=0):
        """
        Wrapper for the sampling function implementations.
        
        :param x: x to sample from
        :param size: number of samples
        :param n: n-th iteration, can be from [0, self.n], None is equivalent to self.n, default is None
        :param predictive_n: Number of predictive resampling values to use, None is equivalent to using all, default is 0
        """
        return self._sample(x, size=size, n=n, predictive_n=predictive_n)
    
    def set_update(self, new_update):
        """
        Setting self.update to be new_update
        
        :param new_update: new update function
        """
        self.update = new_update
    
    def reset_update(self):
        """
        Docstring for reset_update
        """
        def _reset_update(self, n, ms, mu, sigma, _mp=None):
            if _mp == None: _mp = self._train
            return self._update(n, ms, mu, sigma, _mp)
        self.update = _reset_update
    
    def train(self, x, y):
        """
        Fits the MP model, will refit if previously fitted and wipes previous fitted values.
        
        :param x: array of x_n
        :param y: array of y_n, must have same shape as `x`
        """
        self.n = len(y)
        self.x = x.copy()
        self.y = y.copy()

        self._train.ms = self.ms_0(x)
        self._train.mu = self._train.ms[0]
        self._train.sigma = np.sqrt(self._train.ms[1] - self._train.ms[0]**2)
        self._train.zs = np.zeros((self.n,))

        for n in range(1, self.n):
            alpha_n = self.alpha(n, x[n:], x[n-1])
            self._train.zs[n-1] = (y[n-1] - self._train.mu[n-1])/self._train.sigma[n-1]
            self._train.ms[:, n:] += alpha_n * self.update(n, self._train.ms[:, n:], self._train.mu[n:], self._train.sigma[n:], self._train)
            self._train.mu[n:] = self._train.ms[0, n:]
            self._train.sigma[n:] = np.sqrt(self._train.ms[1, n:] - self._train.ms[0, n:]**2)

        self._train.zs[-1] = (y[-1] - self._train.mu[-1])/self._train.sigma[-1]
    
    def set_mp(self, mp_obj):
        """
        Sets the mean, variance, moments, and z_n container for the training section, this does not affect predictive resampling values stored
        
        :param mp_obj: _SemiparametricRegressionMP Object
        """
        self._train = mp_obj

    def predictive_resample(self, x):
        """
        Predictivly resamples, does not reset previous resampling results
        
        :param x: list of x to do predictive resampling on
        """
        x = np.array(x).tolist()
        self.px += x
        self.pn += len(x)

        for _x in x:
            mu, sigma, y = self._fast_pr_sample(_x)
            self.py.append(y.item())
            self._pr.zs.append(((y - mu)/sigma).item())
    
    def reset_predictive_resample(self):
        """
        Resets and wipe any fitted values from predictive resampling
        """
        self.pn = 0
        self.px = []
        self.py = []
        self._pr = _SemiparametricRegressionMP(self.n_moments, static=False)

    def get(self, x_test, n=None, predictive_n=0):
        """
        Gets mean, variance, and moments m^k_n(x) from fitted values, returns a dictionary of said values
        
        :param x_test: list of x to obtain moments from
        :param n: n-th iteration, can be from [0, self.n], None is equivalent to self.n, default is None
        :param predictive_n: Number of predictive resampling values to use, None is equivalent to using all, default is 0
        """
        if n is None: n = self.n
        if predictive_n is None: predictive_n = self.pn
        n = min(self.n, n)
        predictive_n = min(self.pn, predictive_n)

        mu = np.zeros((n+predictive_n+1, len(x_test)))
        sigma = np.zeros((n+predictive_n+1, len(x_test)))
        ms = np.zeros((self.n_moments, n+predictive_n+1, len(x_test)))

        ms[:, 0] = self.ms_0(x_test)
        mu[0] = ms[0, 0]
        sigma[0] = np.sqrt(ms[1, 0] -  ms[0, 0]**2)

        for i in range(1, n+1):
            alpha_i = self.alpha(i, x_test, self.x[i-1]).flatten()
            ms[:, i] = ms[:, i-1] + alpha_i * self.update(i, ms[:, i-1], mu[i-1], sigma[i-1], self._train)
            mu[i] = ms[0, i]
            sigma[i] = np.sqrt(ms[1, i] - ms[0, i]**2)
        
        for i in range(n+1, n+predictive_n+1):
            alpha_i = self.alpha(i, x_test, self.px[i-self.n-1]).flatten()
            ms[:, i] = ms[:, i-1] + alpha_i * self.update(i-n, ms[:, i-1], mu[i-1], sigma[i-1], self._pr)
            mu[i] = ms[0, i]
            sigma[i] = np.sqrt(ms[1, i] - ms[0, i]**2)
        
        return {
            "mu": mu,
            "sigma": sigma,
            "sigma2": sigma**2,
            "moment": ms
        }
    
    def _fast_pr_sample(self, x_test):
        """
        Fast single-item sampling used in self.predictive_resample
        
        :param x_test: x value to sample from
        """
        theta_fromMoments = tukeysG_fromMeanVarSkew
        sample_fromTheta = sample_tukeysG

        ms = self.ms_0(np.array([x_test]))
        mu = ms[0]
        sigma = np.sqrt(ms[1] - ms[0]**2)

        for i in range(1, self.n+1):
            alpha_i = self.alpha(i, x_test, self.x[i-1])
            ms += alpha_i * self.update(i, ms, mu, sigma, self._train)
            mu = ms[0]
            sigma = np.sqrt(ms[1] - ms[0]**2)

        for i in range(self.n+1, self.n+len(self.py)+1):
            alpha_i = self.alpha(i, x_test, self.px[i-self.n-1])
            ms += alpha_i * self.update(i-self.n, ms, mu, sigma, self._pr)
            mu = ms[0]
            sigma = np.sqrt(ms[1] - ms[0]**2)
        
        self._pr.mu.append(mu)
        self._pr.sigma.append(sigma)
        for n_moment in range(self.n_moments): self._pr.ms[n_moment].append(ms[n_moment])
        
        skew = (ms[2] - 3*ms[1]*mu + 2*mu**3)/sigma**3
        return mu, sigma, sample_fromTheta(*theta_fromMoments(mu, sigma**2, skew), size=(1,))