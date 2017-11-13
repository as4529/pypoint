import tensorflow as tf
import numpy as np
import itertools
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
import GPy

def sim_f(X, k=GPy.kern.RBF(input_dim=2, variance=1., lengthscale=10), mu=5):
    return np.random.multivariate_normal(np.ones(X.shape[0]) * mu, k.K(X, X))

def sim_X(D=2, N_dim=30, lower=0, upper=100):
    grid = [np.sort(np.random.uniform(lower, upper, size=N_dim)) for d in range(D)]

    return np.array(list(itertools.product(*grid)))

def poisson_draw(f):
    return np.random.poisson(np.exp(f + 0.2 * np.random.normal(0, 1, size=len(f))))
