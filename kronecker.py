import tensorflow as tf
import numpy as np
import optimizers
from kronecker_utils import kron_mvp, kron_list
import sys
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

"""
Class for Kronecker inference of GPs
"""

class KroneckerSolver:



    def __init__(self, kernel, likelihood, X, y, tau):
        """

        Args:
            kernel (kernels.Kernel): kernel function to use for inference
            likelihood (likelihoods.Likelihood): likelihood of observations given function values
            X (np.array): data
            y (np.array): output
            tau (float): Newton line search hyperparam
        """

        self.kernel = kernel
        self.likelihood = likelihood
        self.X = X
        self.y = y
        self.alpha = tf.zeros(shape=[X.shape[0]])
        self.Ks = self.construct_Ks()
        self.opt = optimizers.CG_optimizer(self.Ks, self.likelihood, tau)


    def construct_Ks(self):
        """

        Returns: list of kernel evaluations (tf.Variable) at each dimension

        """
        self.Ks = [tf.constant(self.kernel.K(np.expand_dims(np.unique(self.X[:, i]), 1)),
                            dtype=tf.float32) for i in range(self.X.shape[1])]

        return self.Ks


    def construct_B(self, W):

        """

        Args:
            W (tf.Variable): negative hessian of likelihood

        Returns: B matrix used in Newton step

        """
        return tf.ones(tf.shape(W)) + W * self.kernel.K(np.array([[1.0]]))


    def step(self, mu, max_it, it, f, delta):
        """
        Runs one step of Kronecker inference
        Args:
            mu (tf.Variable): mean of f at each point x_i
            max_it (int): maximum number of Kronecker iterations
            it (int): current iteration
            f (tf.Variable): current estimate of function values
            delta (tf.Variable): change in step size from previous iteration

        Returns: mean, max iteration, current iteration, function values, and step size

        """

        f = tf.squeeze(kron_mvp(self.Ks, tf.transpose(self.alpha))) + mu
        psi = -tf.reduce_sum(self.likelihood.log_like(self.y, f)) + 0.5*tf.reduce_sum(tf.multiply(self.alpha, f-mu))

        print "Iteration: ", it
        print " psi: ", psi

        grads = self.likelihood.grad(self.y, f)
        W = -self.likelihood.hess(self.y, f)


        B = tf.ones(W.shape[0]) + W
        b = tf.multiply(W, f - mu) + grads
        z = self.opt.cg(B, tf.multiply(tf.pow(W, -0.5), b))

        delta_alpha = tf.squeeze(tf.multiply(tf.pow(W, 0.5), z)) + self.alpha
        ls = self.opt.line_search(self.alpha, delta_alpha, self.y, psi, 20, mu)
        step_size = ls[1]
        print "step", step_size
        self.alpha = self.alpha + delta_alpha*step_size


        it = it + 1

        return mu, max_it, it, f, step_size


    def conv(self, mu, max_it, it, f, delta):
        """
        Assesses convergence of Kronecker inference
        Args:
            mu (tf.Variable): mean of f at each point x_i
            max_it (int): maximum number of Kronecker iterations
            it (int): current iteration
            f (tf.Variable): current estimate of function values
            delta (tf.Variable): change in step size from previous iteration

        Returns: true if continue, false if converged

        """
        return tf.logical_and(tf.less(it, max_it), tf.greater(delta, 1e-5))


    def run(self, mu, max_it, f):
        """
        Runs Kronecker inference
        Args:
            mu (tf.Variable): prior mean
            max_it (int): maximum number of iterations for Kronecker inference
            f (tf.Variable): uninitialized function values

        Returns:

        """
        delta = tfe.Variable(sys.float_info.max)
        it = tfe.Variable(0)
        f = mu

        return tf.while_loop(self.conv, self.step, [mu, max_it, it, f, delta])


    def marginal(self, f, mu, W):
        """
        (in progress) calculates marginal likelihood
        Args:
            f (tf.Variable): function values
            mu (tf.Variable): prior mean
            W (tf.Variable): negative Hessian of likelihood

        Returns: tf.Variable for marginal likelihood

        """
        eigs = []

        for K in self.Ks:
            eigs.append(tf.self_adjoint_eig(K))

        eig_K = kron_list(eigs)

        return 0.5 * tf.reduce_sum(tf.multiply(self.alpha, f - mu)) + \
                0.5*tf.reduce_sum(tf.log(1 + tf.multiply(eig_K, W))) - self.likelihood.log_like(f, self.y)


    def optimize_marginal(self):
        """
        IN PROGRESS
        Returns:

        """
        return 0
    
    def update(self, X, y):
        """
        IN PROGRESS
        Args:
            X ():
            y ():

        Returns:

        """
        return 0