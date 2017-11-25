import tensorflow as tf
import numpy as np
import sys
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
from copy import deepcopy
from operator import mul

"""
Class for Kronecker inference of GPs

"""


class KroneckerSolver:

    def __init__(self, mu, kernel, likelihood, X, y, tau=0.5, k_diag=None, mask=None, verbose = False):
        """

        Args:
            kernel (kernels.Kernel): kernel function to use for inference
            likelihood (likelihoods.Likelihood): likelihood of observations given function values
            X (np.array): data
            y (np.array): output
            tau (float): Newton line search hyperparam
        """
        self.verbose = verbose
        self.X = X
        self.y = y

        self.kernel = kernel
        self.mu = mu
        self.likelihood = likelihood
        self.Ks = self.construct_Ks()
        self.K_eigs = [tf.self_adjoint_eig(K) for K in self.Ks]
        self.k_diag = k_diag

        if self.k_diag is not None:
            self.precondition = 1.0/tf.sqrt(self.k_diag)
        else:
            self.precondition = None
        self.mask = mask

        self.alpha = tf.zeros(shape=[X.shape[0]], dtype = tf.float32)
        self.W = tf.zeros(shape = [X.shape[0]])
        self.grads = tf.zeros(shape = [X.shape[0]])

        self.step_opt = CGOptimizer(self.cg_prod_step)
        self.var_opt = CGOptimizer(self.cg_prod_var)
        self.f = self.mu
        self.tau = tau
        #self.grad_func = tfe.gradients_function(self.likelihood.log_like, [1])
        #self.hess_func = tfe.gradients_function(self.grad_func, [1])
        self.grad_func = likelihood.grad
        self.hess_func = likelihood.hess


    def construct_Ks(self, kernel=None):
        """

        Returns: list of kernel evaluations (tf.Variable) at each dimension

        """
        if kernel == None:
            kernel = self.kernel

        Ks = [tfe.Variable(kernel.eval(np.expand_dims(np.unique(self.X[:, i]), 1)),
                            dtype=tf.float32) for i in range(self.X.shape[1])]

        if kernel == None:
            self.Ks = Ks

        return Ks

    def step(self, max_it, it, prev, delta):
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

        self.f = kron_mvp(self.Ks, self.alpha) + self.mu

        if self.k_diag is not None:
            self.f += tf.multiply(self.alpha, self.k_diag)

        if self.mask is not None:
            y_lim = tf.boolean_mask(self.y, self.mask)
            f_lim = tf.boolean_mask(self.f, self.mask)
            alpha_lim = tf.boolean_mask(self.alpha, self.mask)
            mu_lim = tf.boolean_mask(self.mu, self.mask)
            psi = -tf.reduce_sum(self.likelihood.log_like(y_lim, f_lim)) + \
                  0.5 * tf.reduce_sum(tf.multiply(alpha_lim, f_lim - mu_lim))
        else:
            psi = -tf.reduce_sum(self.likelihood.log_like(self.y, self.f)) +\
              0.5*tf.reduce_sum(tf.multiply(self.alpha, self.f-self.mu))

        if self.verbose:
            print "Iteration: ", it
            print " psi: ", psi

        self.grads = self.grad_func(self.y, self.f)
        hess = self.hess_func(self.y, self.f)
        self.W = -hess

        if self.precondition is not None:
            self.precondition = 1.0 / tf.multiply(tf.sqrt(self.k_diag), tf.sqrt(self.W))

        b = tf.multiply(self.W, self.f - self.mu) + self.grads

        if self.precondition is not None:
            z = self.step_opt.cg(tf.multiply(self.precondition,
                                             tf.multiply(1.0/tf.sqrt(self.W), b)))
        else:
            z = self.step_opt.cg(tf.multiply(1.0/tf.sqrt(self.W), b))

        delta_alpha = tf.multiply(tf.sqrt(self.W), z) - self.alpha
        ls = self.line_search(delta_alpha, psi, 40)
        step_size = ls[1]

        if self.verbose:
            print "step", step_size
            print ""

        delta = prev - psi
        prev = psi
        self.alpha = self.alpha + delta_alpha*step_size
        self.alpha = tf.where(tf.is_nan(self.alpha), tf.ones_like(self.alpha) * 1e-9, self.alpha)
        it = it + 1

        return max_it, it, prev, delta

    def conv(self, max_it, it, prev, delta):
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

    def run(self, max_it):
        """
        Runs Kronecker inference
        Args:
            mu (tf.Variable): prior mean
            max_it (int): maximum number of iterations for Kronecker inference
            f (tf.Variable): uninitialized function values

        Returns:

        """
        delta = tfe.Variable(sys.float_info.max)
        prev = tfe.Variable(sys.float_info.max)
        it = tfe.Variable(0)

        out = tf.while_loop(self.conv, self.step, [max_it, it, prev, delta])
        self.f = kron_mvp(self.Ks, self.alpha) + self.mu
        self.grads = self.grad_func(self.y, self.f)

        return out

    def cg_prod_step(self, p):

        if self.precondition is None:
            return p + tf.multiply(tf.sqrt(self.W), kron_mvp(self.Ks, tf.multiply(tf.sqrt(self.W), p)))

        Cp = tf.multiply(self.precondition, p)

        return tf.multiply(self.precondition, Cp) +\
            tf.multiply(tf.multiply(self.precondition, tf.multiply(self.W, self.k_diag)),
                             Cp) +\
            tf.multiply(tf.multiply(self.precondition, tf.sqrt(self.W)),
                            kron_mvp(self.Ks, tf.multiply(tf.sqrt(self.W), Cp)))

    def search_step(self, obj_prev, obj_search, min_obj, delta_alpha,
                   step_size, max_it, t, opt_step):
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
        alpha_search = tf.squeeze(self.alpha + step_size * delta_alpha)
        f_search = tf.squeeze(kron_mvp(self.Ks, alpha_search)) + self.mu

        if self.mask is not None:
            f_search += tf.multiply(self.k_diag, alpha_search)

        obj_search = self.eval_obj(f_search, alpha_search)
        opt_step = tf.cond(tf.greater(min_obj, obj_search), lambda: step_size, lambda: opt_step)
        min_obj = tf.cond(tf.greater(min_obj, obj_search), lambda: obj_search, lambda: min_obj)
        step_size = self.tau * step_size
        t = t + 1

        return obj_prev, obj_search, min_obj, delta_alpha,\
               step_size, max_it, t, opt_step

    def eval_obj(self, f = None, alpha = None):

        if self.mask is not None:
            y_lim = tf.boolean_mask(self.y, self.mask)
            f_lim = tf.boolean_mask(f, self.mask)
            alpha_lim = tf.boolean_mask(alpha, self.mask)
            mu_lim = tf.boolean_mask(self.mu, self.mask)
            return -tf.reduce_sum(self.likelihood.log_like(y_lim, f_lim)) + \
                         0.5 * tf.reduce_sum(tf.multiply(alpha_lim, f_lim - mu_lim))

        return -tf.reduce_sum(self.likelihood.log_like(self.y, f)) + 0.5 * tf.reduce_sum(
            tf.multiply(alpha, f - self.mu))

    def converge_cond(self, obj_prev, obj_search, min_obj,
                      delta_alpha, step_size, max_it, t, opt_step):
        """

        Assesses convergence of line search. Same params as above.

        """

        return tf.logical_and(tf.less(t, max_it), tf.less(obj_prev - obj_search, step_size*t))

    def line_search(self, delta_alpha, obj_prev, max_it):
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

        t = 1

        res = tf.while_loop(self.converge_cond, self.search_step, [obj_prev, obj_search, min_obj, delta_alpha,
                                                         step_size, max_it, t, opt_step])

        return res[2], res[-1]

    def marginal(self, Ks_new = None):
        """
        calculates marginal likelihood
        Args:
            f (tf.Variable): function values
            mu (tf.Variable): prior mean
            self.W (tf.Variable): negative Hessian of likelihood

        Returns: tf.Variable for marginal likelihood

        """

        if Ks_new == None:
            Ks = self.Ks
        else:
            Ks = Ks_new

        eigs = [tf.expand_dims(tf.self_adjoint_eig(K)[0], 1) for K in Ks]
        eig_K = tf.squeeze(kron_list(eigs))

        if self.mask is not None:

            y_lim = tf.boolean_mask(self.y, self.mask)
            f_lim = tf.boolean_mask(self.f, self.mask)
            alpha_lim = tf.boolean_mask(self.alpha, self.mask)
            mu_lim = tf.boolean_mask(self.mu, self.mask)
            W_lim = tf.boolean_mask(self.W, self.mask)
            eig_k_lim = tf.boolean_mask(eig_K, self.mask)

            return -0.5 * tf.reduce_sum(tf.multiply(alpha_lim, f_lim - mu_lim)) - \
                   0.5 * tf.reduce_sum(tf.log(1 + tf.multiply(eig_k_lim, W_lim))) + \
                   tf.reduce_sum(self.likelihood.log_like(y_lim, f_lim))

        return -0.5 * tf.reduce_sum(tf.multiply(self.alpha, self.f - self.mu)) - \
               0.5*tf.reduce_sum(tf.log(1 + tf.multiply(eig_K, self.W))) +\
               tf.reduce_sum(self.likelihood.log_like(self.y, self.f))

    def variance(self, n_s):

        self.root_eigdecomp = self.sqrt_eig()

        n = self.X.shape[0]
        var = tf.zeros([self.X.shape[0]])
        id_norm = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros([n]), tf.ones([n]))

        for i in range(n_s):

            g_m = tf.expand_dims(id_norm.sample(), 1)
            g_n = id_norm.sample()

            right_side = tf.squeeze(tf.matmul(self.root_eigdecomp, g_m)) + tf.multiply(tf.sqrt(self.k_diag), tf.squeeze(g_m)) + \
                         tf.multiply(1.0/tf.sqrt(self.W), g_n)

            r = self.var_opt.cg(right_side)
            var += tf.square(kron_mvp(self.Ks, r))

        return tf.ones([self.X.shape[0]]) - var/n_s*1.0

    def sqrt_eig(self):

        res = []

        for e, v in self.K_eigs:
            e_root_diag = tf.sqrt(e)
            e_root = tf.diag(tf.where(tf.is_nan(e_root_diag), tf.zeros_like(e_root_diag), tf.sqrt(e_root_diag)))
            res.append(tf.matmul(tf.matmul(v, e_root), tf.transpose(v)))

        res = tf.squeeze(kron_list(res))
        self.root_eigdecomp = tf.constant(res)

        return res

    def cg_prod_var(self, p):

        return tf.multiply(self.k_diag, p) + tf.squeeze(kron_mvp(self.Ks, p)) + tf.multiply(1.0 / self.W, p)

    def predict_mean(self, x_new):

        k_dims = [self.kernel.eval(np.expand_dims(np.unique(self.X[:, d]), 1), np.expand_dims(x_new[:, d], 1))
                  for d in self.X.shape[1]]

        kx = tf.squeeze(kron_list(k_dims))

        mean = tf.reduce_sum(tf.multiply(kx, self.alpha)) + self.mu[0]

        return mean



class CGOptimizer:

    def __init__(self, cg_prod = None):

        self.cg_prod = cg_prod

    def cg_converged(self, p, count, x, r, max_it):
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
        return tf.logical_and(tf.greater(tf.reduce_sum(tf.multiply(r, r)), 1e-3),
                                             tf.less(count, max_it))

    def cg_body(self, p, count, x, r, max_it):
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
        Bp = self.cg_prod(p)

        self.Bp = Bp
        self.p = p

        norm_k = tf.reduce_sum(tf.multiply(r, r))

        alpha = norm_k / tf.reduce_sum(tf.multiply(p, Bp))
        x += alpha * p
        r -= alpha * Bp

        if tf.reduce_sum(tf.multiply(r, r)).numpy() < 1e-3:
            return p, count, x, r, max_it

        norm_next = tf.reduce_sum(tf.multiply(r, r))
        beta = norm_next / norm_k

        p = r + beta*p

        return p, count, x, r, max_it

    def cg(self, b, x=None, z=None, max_it = None):
        """
        solves linear system Ax = b
        Args:
            A (tf.Variable): matrix A
            b (tf.Variable): vector b
            x (): solution
            precondition(): diagonal of preconditioning matrix

        Returns: returns x that solves linear system

        """

        count = tf.constant(0)
        n = b.get_shape().as_list()[0]

        if max_it is None:
            max_it = 2*n

        if not x:
            x = tf.zeros(shape=[n])

        r =  b - self.cg_prod(x)
        p = r

        fin = tf.while_loop(self.cg_converged, self.cg_body, [p, count, x,
                                                              r, max_it])

        return fin[2]


class KernelLearner:

    def __init__(self, mu, kernel, likelihood, X, y, tau,
                 k_diag = None, mask = None, eps = np.array([1e-5, 1])):

        self.kernel = kernel
        self.mu = mu
        self.likelihood = likelihood
        self.X = X
        self.y = y
        self.tau = tau
        self.k_diag = k_diag
        self.mask = mask
        self.eps = eps

    def optimize_marginal(self, init_params):

        return 0

    def gradient_step(self, params):

        for i in range(len(params)):

            fin_diff = self.finite_difference(self.eps[i], params, i)

        return 0

    def finite_difference(self, epsilon, params, i):

        param_step = deepcopy(params)

        param_step[i] += self.eps[i]
        marg_plus = self.get_marginal(param_step)

        param_step[i] -= 2 * self.eps[i]
        marg_minus = self.get_marginal(param_step)

        fin_diff = (marg_plus - marg_minus) / (2 * self.eps[i])

        return fin_diff

    def get_marginal(self, params):

        kernel = self.kernel(*params)
        solver = KroneckerSolver(self.mu, kernel, self.likelihood, self.X, self.y,
                                 self.tau, self.k_diag, self.mask)
        solver.run(10)
        marg = solver.marginal()
        return marg


def kron(A, B):
    """
    Kronecker product of two matrices
    Args:
        A (tf.Variable): first matrix for kronecker product
        B (tf.Variable): second matrix

    Returns: kronecker product of A and B

    """

    n_col = A.shape[1] * B.shape[1]
    out = tf.zeros([0, n_col])

    for i in range(A.shape[0]):

        row = tf.zeros([B.shape[0], 0])

        for j in range(A.shape[1]):
            row = tf.concat([row, A[i, j] * B], 1)

        out = tf.concat([out, row], 0)

    return out

def kron_list(matrices):
    """
    Kronecker product of a list of matrices
    Args:
        matrices (list of tf.Variable): list of matrices

    Returns:

    """
    out = kron(matrices[0], matrices[1])

    for i in range(2, len(matrices)):
        out = kron(out, matrices[i])

    return out


def kron_mvp(Ks, v):
    """
    Matrix vector product using Kronecker structure
    Args:
        Ks (list of tf.Variable): list of matrices corresponding to kronecker decomposition
        of K
        v (tf.Variable): vector to multiply K by

    Returns: matrix vector product of K and v

    """

    mvp = tf.transpose(tf.reshape(tf.expand_dims(v, 1), [-1, Ks[-1].shape.as_list()[0]]))

    for idx, k in enumerate(reversed(Ks)):
        if idx > 0:
            rows = k.shape.as_list()[0]
            mvp = tf.reshape(mvp, [rows, -1])
        mvp = tf.transpose(tf.matmul(k, mvp))

    return tf.reshape(tf.transpose(mvp), [-1])
