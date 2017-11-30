import tensorflow as tf
import numpy as np
import sys
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
from copy import deepcopy
from operator import mul

"""
Class for Kronecker inference of GPs. Inspiration from GPML.

For references, see:

Flaxman and Wilson (2014), Fast Kronecker Inference in Gaussian Processes with non-Gaussian Likelihoods
Rassmussen and Williams (2006), Gaussian Processes for Machine Learning
Wilson et al (2012), Fast Kernel Learning for Multidimensional Pattern Extrapolation
Wilson et al (2014). Thoughts on Massively Scalable Gaussian Processes

Most of the notation follows R and W chapter 2, and Flaxman and Wilson

"""


class KroneckerSolver:

    def __init__(self, mu, kernel, likelihood, X, y, tau=0.5, obs_idx=None, verbose = False):
        """

        Args:
            kernel (kernels.Kernel): kernel function to use for inference
            likelihood (likelihoods.Likelihood): likelihood of observations given function values
            X (np.array): data
            y (np.array): output
            tau (float): Newton line search hyperparam
            obs_idx (np.array): Indices of observed points if working with a partial grid
            verbose (bool): verbose or not
        """

        self.verbose = verbose
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        self.obs_idx = obs_idx

        self.kernel = kernel
        self.mu = mu
        self.likelihood = likelihood
        self.Ks = self.construct_Ks()
        self.K_eigs = [tf.self_adjoint_eig(K) for K in self.Ks]
        self.root_eigdecomp = None

        self.alpha = tf.zeros([X.shape[0]], tf.float32)
        self.W = tfe.Variable(tf.zeros([X.shape[0]], tf.float32))
        self.grads = tf.zeros([X.shape[0]], tf.float32)
        self.opt = CGOptimizer(self.cg_prod)

        self.f = self.mu
        self.f_pred = self.f
        self.tau = tau
        self.grad_func = tfe.gradients_function(self.likelihood.log_like, [1])
        self.hess_func = tfe.gradients_function(self.grad_func, [1])

    def construct_Ks(self, kernel=None):
        """

        Constructs kronecker-decomposed kernel matrix

        Args:
            kernel (): kernel (if not using kernel passed in constructor)

        Returns: List of kernel evaluated at each dimension

        """
        if kernel is None:
            kernel = self.kernel

        Ks = [tfe.Variable(kernel.eval(np.expand_dims(np.unique(self.X[:, i]), 1)),
                            dtype=tf.float32) for i in range(self.X.shape[1])]

        return Ks

    def sqrt_eig(self):
        """
        Calculates square root of kernel matrix using fast kronecker eigendecomp.
        This is used in stochastic approximations of the predictive variance.

        Returns: Square root of kernel matrix

        """
        res = []

        for e, v in self.K_eigs:
            e_root_diag = tf.sqrt(e)
            e_root = tf.diag(tf.where(tf.is_nan(e_root_diag), tf.zeros_like(e_root_diag), e_root_diag))
            res.append(tf.matmul(tf.matmul(v, e_root), tf.transpose(v)))

        res = tf.squeeze(kron_list(res))
        self.root_eigdecomp = tf.constant(res)

        return res

    def run(self, max_it):
        """
        Runs Kronecker inference. Updates instance variables.

        Args:
            max_it (int): maximum number of iterations for Kronecker inference

        Returns: max iterations, iteration number, objective

        """
        if self.obs_idx is not None:
            k_diag = np.ones(self.X.shape[0]) * 1e12
            k_diag[self.obs_idx] = 1.
            self.k_diag = tf.cast(tfe.Variable(k_diag, tf.float32), tf.float32)
            self.precondition = tf.clip_by_value(1.0 / tf.sqrt(self.k_diag), 0, 1)
        else:
            self.k_diag = None
            self.precondition = None

        delta = tfe.Variable(sys.float_info.max)
        it = tfe.Variable(0)

        out = tf.while_loop(self.conv, self.step, [max_it, it, delta])

        """
        if self.obs_idx is not None:
            W = self.W.numpy()
            W[list(set(range(self.n)) - set(self.obs_idx))] = 0.
            self.W = tfe.Variable(W, tf.float32)
        """
        return out

    def step(self, max_it, it, delta):
        """
        Runs one step of Kronecker inference
        Args:
            max_it (int): maximum number of Kronecker iterations
            it (int): current iteration
            delta (tf.Variable): change in step size from previous iteration

        Returns: max iteration, current iteration, previous objective, change in objective

        """

        self.f = kron_mvp(self.Ks, self.alpha) + self.mu
        if self.k_diag is not None:
            self.f += tf.multiply(self.alpha, self.k_diag)
        psi = self.eval_obj(self.f, self.alpha)

        if self.obs_idx is None:
            self.grads = self.grad_func(self.y, self.f)[0]
            hess = self.hess_func(self.y, self.f)[0]
            self.W = -hess
        else:
            self.grads, hess = self.gather_derivs()
            self.hess = hess
            self.W = tf.clip_by_value(tfe.Variable(-hess, tf.float32), 1e-9, 1e16)
        self.W = tf.where(tf.is_nan(self.W), tf.ones_like(self.W)*1e-9, self.W)

        b = tf.multiply(self.W, self.f - self.mu) + self.grads
        if self.precondition is not None:
            z = self.opt.cg(tf.multiply(self.precondition,
                                             tf.multiply(1.0/tf.sqrt(self.W), b)))
        else:
            z = self.opt.cg(tf.multiply(1.0/tf.sqrt(self.W), b))

        delta_alpha = tf.multiply(tf.sqrt(self.W), z) - self.alpha
        step_size = self.line_search(delta_alpha, psi, 20)

        if self.verbose:
            print "Iteration: ", it
            print " psi: ", psi
            print "step", step_size
            print ""

        delta = step_size

        if delta > 1e-9:
            self.alpha = self.alpha + delta_alpha*step_size
            self.alpha = tf.where(tf.is_nan(self.alpha), tf.ones_like(self.alpha) * 1e-9, self.alpha)
            self.f_pred = kron_mvp(self.Ks, self.alpha) + self.mu

        it = it + 1

        return max_it, it, delta

    def conv(self, max_it, it, delta):
        """
        Assesses convergence of Kronecker inference
        Args: Same as above function
        Returns: true if continue, false if converged

        """
        return tf.logical_and(tf.less(it, max_it), tf.greater(delta, 1e-9))

    def line_search(self, delta_alpha, obj_prev, max_it):
        """
        Executes line search for optimal Newton step
        Args:
            delta_alpha (tf.Variable): change in search direction
            obj_prev (tf.Variable): previous objective value
            max_it (int): maximum number of iterations

        Returns: optimal step size

        """
        obj_search = sys.float_info.max
        min_obj = obj_prev
        step_size = 2.0
        opt_step = 0.0
        t = 1

        res = tf.while_loop(self.converge_line, self.search_step, [obj_prev, obj_search, min_obj, delta_alpha,
                                                                   step_size, max_it, t, opt_step])

        return res[-1]

    def search_step(self, obj_prev, obj_search, min_obj, delta_alpha,
                   step_size, max_it, t, opt_step):
        """
        Executes one step of a backtracking line search
        Args:
            obj_prev (tf.Variable): previous objective
            obj_search (tf.Variable): current objective
            min_obj (tf.Variable): current minimum objective
            delta_alpha (tf.Variable): change in step size from last iteration
            step_size (tf.Variable): current step size
            max_it (int): maximum number of line search iterations
            t (tf.Variable): current line search iteration
            opt_step (tf.Variable): optimal step size until now

        Returns: updated parameters
        """
        alpha_search = tf.squeeze(self.alpha + step_size * delta_alpha)
        f_search = tf.squeeze(kron_mvp(self.Ks, alpha_search)) + self.mu

        if self.k_diag is not None:
            f_search += tf.multiply(self.k_diag, alpha_search)

        obj_search = self.eval_obj(f_search, alpha_search)
        opt_step = tf.cond(tf.greater(min_obj, obj_search), lambda: step_size, lambda: opt_step)
        min_obj = tf.cond(tf.greater(min_obj, obj_search), lambda: obj_search, lambda: min_obj)
        step_size = self.tau * step_size
        t = t + 1

        return obj_prev, obj_search, min_obj, delta_alpha,\
               step_size, max_it, t, opt_step

    def converge_line(self, obj_prev, obj_search, min_obj,
                      delta_alpha, step_size, max_it, t, opt_step):
        """
        Assesses convergence of line search. Same params as above.
        """

        return tf.logical_and(tf.less(t, max_it), tf.less(obj_prev - obj_search, step_size * t))

    def eval_obj(self, f = None, alpha = None):

        """
        Evaluates objective function (negative log likelihood plus GP penalty)
        Args:
            f (): function values (if not same as class variable)
            alpha (): alpha (if not same as class variable)

        Returns:
        """

        if self.obs_idx is not None:
            f_lim = tf.gather(f, self.obs_idx)
            alpha_lim = tf.gather(alpha, self.obs_idx)
            mu_lim = tf.gather(self.mu, self.obs_idx)
            return -tf.reduce_sum(self.likelihood.log_like(self.y, f_lim)) + \
                         0.5 * tf.reduce_sum(tf.multiply(alpha_lim, f_lim - mu_lim))

        return -tf.reduce_sum(self.likelihood.log_like(self.y, f)) + 0.5 * tf.reduce_sum(
            tf.multiply(alpha, f - self.mu))

    def marginal(self, Ks_new = None):
        """
        calculates marginal likelihood
        Args:
            Ks_new: new covariance if needed
        Returns: tf.Variable for marginal likelihood

        """

        if Ks_new == None:
            Ks = self.Ks
        else:
            Ks = Ks_new
        eigs = [tf.expand_dims(tf.self_adjoint_eig(K)[0], 1) for K in Ks]
        eig_K = tf.squeeze(kron_list(eigs))
        self.eig_K = eig_K

        if self.obs_idx is not None:
            f_lim = tf.gather(self.f, self.obs_idx)
            self.f_lim = f_lim
            alpha_lim = tf.gather(self.alpha, self.obs_idx)
            self.alpha_lim = alpha_lim
            mu_lim = tf.gather(self.mu, self.obs_idx)
            self.mu_lim = mu_lim
            W_lim = tf.gather(self.W, self.obs_idx)
            self.W_lim = W_lim
            eig_k_lim = tf.gather(eig_K, self.obs_idx)
            self.eig_k_lim = eig_k_lim

            pen = -0.5 * tf.reduce_sum(tf.multiply(alpha_lim, f_lim - mu_lim))
            pen = tf.where(tf.is_nan(pen), tf.zeros_like(pen), pen)
            eigs =  0.5 * tf.reduce_sum(tf.log(1 + tf.multiply(eig_k_lim, W_lim)))
            eigs = tf.where(tf.is_nan(eigs), tf.zeros_like(eigs), eigs)
            like = tf.reduce_sum(self.likelihood.log_like(self.y, f_lim))
            like = tf.where(tf.is_nan(like), tf.zeros_like(like), like)

            return pen+eigs+like

        return -0.5 * tf.reduce_sum(tf.multiply(self.alpha, self.f - self.mu)) - \
               0.5*tf.reduce_sum(tf.log(1 + tf.multiply(eig_K, self.W))) +\
               tf.reduce_sum(self.likelihood.log_like(self.y, self.f))

    def variance(self, n_s):
        """
        Stochastic approximator of predictive variance. Follows "Massively Scalable GPs"
        Args:
            n_s (int): Number of iterations to run stochastic approximation

        Returns: Approximate predictive variance at grid points

        """

        if self.root_eigdecomp is None:
            self.root_eigdecomp = self.sqrt_eig()

        WK = tf.matmul(tf.diag(tf.sqrt(self.W)), self.root_eigdecomp)

        if self.precondition is not None:
            W_kd = tf.multiply(tf.sqrt(self.W), tf.sqrt(self.k_diag))

        var = tf.zeros([self.n])
        id_norm = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros([self.n]), tf.ones([self.n]))

        for i in range(n_s):
            g_m = id_norm.sample()
            g_n = id_norm.sample()
            if self.precondition is None:
                right_side = tf.squeeze(tf.matmul(WK, tf.expand_dims(g_m, 1))) + tf.squeeze(g_n)
            else:
                cov_term =  tf.squeeze(tf.matmul(WK, tf.expand_dims(g_m, 1)))
                noise_term = tf.multiply(W_kd, g_n)
                right_side = tf.multiply(self.precondition, cov_term + noise_term)

            right_side = tf.where(tf.is_nan(right_side), tf.zeros_like(right_side), right_side)
            r = self.opt.cg(right_side)
            var += tf.square(tf.squeeze(kron_mvp(self.Ks, tf.multiply(tf.sqrt(self.W), r))))

        return tf.nn.relu(tf.squeeze(self.kernel.eval([[0.]],[[0.]])) - var/n_s*1.0)

    def predict_mean(self, x_new):

        k_dims = [self.kernel.eval(np.expand_dims(np.unique(self.X[:, d]), 1), np.expand_dims(x_new[:, d], 1))
                  for d in self.X.shape[1]]
        kx = tf.squeeze(kron_list(k_dims))
        mean = tf.reduce_sum(tf.multiply(kx, self.alpha)) + self.mu[0]

        return mean

    def cg_prod(self, p):
        """

        Args:
            p (tfe.Variable): potential solution to linear system

        Returns: product Ap (left side of linear system)

        """

        if self.precondition is None:
            return p + tf.multiply(tf.sqrt(self.W), kron_mvp(self.Ks, tf.multiply(tf.sqrt(self.W), p)))

        Cp = tf.multiply(self.precondition, p)
        noise = tf.multiply(tf.multiply(self.precondition, tf.multiply(self.W, self.k_diag)),
                             Cp)
        wkw = tf.multiply(tf.multiply(self.precondition, tf.sqrt(self.W)),
                            kron_mvp(self.Ks, tf.multiply(tf.sqrt(self.W), Cp)))

        return noise + wkw + tf.multiply(self.precondition, Cp)

    def gather_derivs(self):
        """

        Returns: sum of gradients if there are multiply hessians at single points

        """

        obs_f = tf.gather(self.f, self.obs_idx)
        obs_grad = self.grad_func(self.y, obs_f)[0]
        obs_hess = self.hess_func(self.y, obs_f)[0]

        agg_grad = np.zeros(self.n, np.float32)
        agg_hess = np.zeros(self.n, np.float32)

        for i, j in enumerate(self.obs_idx):
            agg_grad[j] += obs_grad[i]
            agg_hess[j] += obs_hess[i]

        return agg_grad, agg_hess

class CGOptimizer:

    def __init__(self, cg_prod = None, tol = 1e-3):

        self.cg_prod = cg_prod
        self.tol = tol

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
        return tf.logical_and(tf.greater(tf.reduce_sum(tf.multiply(r, r)), self.tol),
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

        norm_k = tf.reduce_sum(tf.multiply(r, r))
        alpha = norm_k / tf.reduce_sum(tf.multiply(p, Bp))
        x += alpha * p
        r -= alpha * Bp

        if tf.reduce_sum(tf.multiply(r, r)).numpy() < 1e-5:
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
        b = tf.where(tf.is_nan(b), tf.ones_like(b) * 1e-9, b)

        if max_it is None:
            max_it = 2*n

        if not x:
            x = tf.zeros(shape=[n])
            r = b
        else:
            r = b - self.cg_prod(x)

        p = r

        fin = tf.while_loop(self.cg_converged, self.cg_body, [p, count, x, r, max_it])

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
