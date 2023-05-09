import numpy as np
import numpy.random as rng

class Generator_doubleNormal(object):
    def __init__(self, p, theta, sigma0, sigma1, sigma=1) -> None:
        self.p = p 
        self.theta = theta 
        self.sigma0 = sigma0
        self.sigma1 = sigma1 
        self.sigma = sigma
        self.X = np.eye(p)
    
    def generate_samples(self, n):
        theta = np.ones((n, self.p)) * self.theta
        gamma = rng.binomial(1, theta)
        beta = np.zeros((n, self.p))
        beta[gamma == 1] = rng.randn(np.sum(gamma == 1)) * self.sigma1
        beta[gamma == 0] = rng.randn(np.sum(gamma == 0)) * self.sigma0 
        Y = beta@self.X.T + rng.randn(n, self.p) * self.sigma 
        return gamma, beta, Y 

class Generator_doubleNormal_lr(object):
    def __init__(self, X, theta, sigma0, sigma1, sigma=1) -> None:
        self.X = X
        self.N = X.shape[0]
        self.p = X.shape[1] 
        self.theta = theta 
        self.sigma0 = sigma0
        self.sigma1 = sigma1 
        self.sigma = sigma
  
    
    def generate_samples(self, n):
        theta = np.ones((n, self.p)) * self.theta
        gamma = rng.binomial(1, theta)
        beta = np.zeros((n, self.p))
        beta[gamma == 1] = rng.randn(np.sum(gamma == 1)) * self.sigma1
        beta[gamma == 0] = rng.randn(np.sum(gamma == 0)) * self.sigma0 
        Y = beta@self.X.T + rng.randn(n, self.N) * self.sigma 
        return gamma, beta, Y 
    
class Generator_logistic(object):
    def __init__(self, X, theta, sigma0, sigma1) -> None:
        self.X = X
        self.N = X.shape[0]
        self.p = X.shape[1] 
        self.theta = theta 
        self.sigma0 = sigma0
        self.sigma1 = sigma1
    
    def generate_samples(self, n):
        theta = np.ones((n, self.p)) * self.theta
        gamma = rng.binomial(1, theta)
        beta = np.zeros((n, self.p))
        beta[gamma == 1] = rng.randn(np.sum(gamma == 1)) * self.sigma1
        beta[gamma == 0] = rng.randn(np.sum(gamma == 0)) * self.sigma0 
        prob = np.exp(beta@self.X.T) / (1 + np.exp(beta@self.X.T))
        Y = rng.binomial(1, prob)
        return gamma, beta, Y
   