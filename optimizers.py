import tensorflow as tf
import sys
from kronecker import kron_mvp

"""
    Conjugate gradient descent optimizer for Kronecker inference
"""
class CG_optimizer:

    def __init__(self, Ks, likelihood, tau):
        """

        Args:
            Ks (list of tf.Variable): list of kernel evaluations by dimension
            likelihood (likelihoods.Likelihood): likelihood function
            tau (float): hyperparameter for line search
        """
        self.likelihood = likelihood
        self.Ks = Ks
        self.tau = tau

    def cg_converged(self, A, p, r_k_norm, count, x, r, n):
        """
        Assesses convergence of CG
        Args:
            A (tf.Variable): matrix on left side of linear system
            p (tf.Variable): search direction
            r_k_norm (tf.Variable): norm of r_k
            count (int): iteration number
            x (tf.Variable): current estimate of solution to linear system
            r (tf.Variable): current residual (b - Ax)
            n (int): size of b

        Returns: false if converged, true if not

        """
        return tf.logical_and(tf.greater(r_k_norm, 1e-5), tf.less(count, n))

    def cg_body(self, A, p, r_k_norm, count, x, r, n):
        """

        Executes one step of conjugate gradient descent

        Args:
            A (tf.Variable): matrix on left side of linear system
            p (tf.Variable): search direction
            r_k_norm (tf.Variable): norm of r_k
            count (int): iteration number
            x (tf.Variable): current estimate of solution to linear system
            r (tf.Variable): current residual (p - Ax)
            n (int): size of b

        Returns: updated parameters for CG
        """
        count = count + 1
        Ap = tf.multiply(A, p)
        alpha = r_k_norm / tf.reduce_sum(tf.multiply(Ap, p))

        x += alpha * p
        r += alpha * Ap

        r_kplus1_norm = tf.reduce_sum(tf.multiply(r, r))

        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        p = beta * p - r

        return A, p, r_k_norm, count, x, r, n


    def cg(self, A, b, x=None):
        """
        solves linear system Ax = b
        Args:
            A (tf.Variable): matrix A
            b (tf.Variable): vector b
            x (): solution

        Returns: returns x that solves linear system

        """
        count = tf.constant(0)
        n = b.get_shape().as_list()[0]

        if not x:
            x = tf.ones(shape=[1, n])

        r = tf.multiply(A, x) - b
        p = - r
        r_k_norm = tf.reduce_sum(tf.multiply(r, r))

        fin = tf.while_loop(self.cg_converged, self.cg_body, [A, p, r_k_norm, count, x, r, 2 * n])

        return fin[4]


    def search_step(self, obj_prev, obj_search, min_obj, alpha, delta_alpha,
                    y, step_size, grad_norm, max_it, t, mu, opt_step):
        """
        Executes one step of a backtracking line search
        Args:
            obj_prev (tf.Variable): previous objective
            obj_search (tf.Variable): current objective
            min_obj (tf.Variable): current minimum objective
            alpha (tf.Variable): current search point
            delta_alpha (tf.Variable): change in step size from last iteration
            y (tf.Variable): realized function values from GP
            step_size (tf.Variable): current step size
            grad_norm (tf.Variable): norm of gradient
            max_it (int): maximum number of line search iterations
            t (tf.Variable): current line search iteration
            mu (tf.Variable): prior mean
            opt_step (tf.Variable): optimal step size until now

        Returns:

        """
        alpha_search = tf.squeeze(alpha + step_size * delta_alpha)
        f_search = tf.squeeze(kron_mvp(self.Ks, alpha_search)) + mu

        obj_search = -tf.reduce_sum(self.likelihood.log_like(y, f_search)) + 0.5 * tf.reduce_sum(
            tf.multiply(alpha_search, f_search - mu))

        opt_step = tf.cond(tf.greater(min_obj, obj_search), lambda: step_size, lambda: opt_step)
        min_obj = tf.cond(tf.greater(min_obj, obj_search), lambda: obj_search, lambda: min_obj)

        step_size = self.tau * step_size
        t = t + 1

        return obj_prev, obj_search, min_obj, alpha, delta_alpha, y,\
               step_size, grad_norm, max_it, t, mu, opt_step,


    def converge_cond(self, obj_prev, obj_search, min_obj, alpha,
                      delta_alpha, y, step_size, grad_norm, max_it, t, mu, opt_step):
        """

        Assesses convergence of line search. Same params as above.

        """
        return tf.logical_and(tf.less(t, max_it), tf.less(obj_prev - obj_search, 0.5 * step_size * grad_norm))


    def line_search(self, alpha, delta_alpha, y, obj_prev, max_it, mu):
        """
        Executes line search for optimal Newton step
        Args:
            alpha (tf.Variable): search direction
            delta_alpha (tf.Variable): change in search direction
            y (tf.Variable): realized values from GP point process
            obj_prev (tf.Variable): previous objective value
            max_it (int): maximum number of iterations
            mu (tf.Variable): prior mean

        Returns: (min objective, optimal step size)

        """
        obj_search = sys.float_info.max
        min_obj = obj_prev

        step_size = 2.0
        opt_step = 0.0

        grad_norm = tf.reduce_sum(tf.multiply(alpha, alpha))
        t = 1

        res = tf.while_loop(self.converge_cond, self.search_step, [obj_prev, obj_search, min_obj, alpha, delta_alpha,
                                                         y, step_size, grad_norm, max_it, t, mu, opt_step])

        return res[2], res[-1]