import numpy as np
import tensorflow as tf
import itertools
from kern import RBF
from tensorflow.contrib.distributions import Bernoulli
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
from kronecker import KroneckerSolver
import kernels
from likelihoods import BernoulliSigmoidLike


class ThinnedEventsSampler:

    def __init__(self, kern=None, f_lambda=None, dim=2, N_dim=20,
            measure=None, rate=10, bern_p=0.5, n_iter=10):

        self.dim = dim
        self.N_dim = N_dim
        self.measure = measure
        self.rate = rate
        if kern:
            self.kern = kern
        else:
            self.kern = RBF(input_dim=self.dim, variance=1.0, lengthscale=5.0)
        if f_lambda:
            self.gen_from_lambda(f_lambda)
            self.S_k, self.G_k = self.constructS_k(sim_data=False)
        else:
            self.gen_grid()
            self.S_k, self.G_k = self.constructS_k(sim_data=True)
        self.bern_p = bern_p
        self.bern = Bernoulli(probs=self.bern_p)
        self.n_iter = n_iter
        self.x_K = tf.constant(self.S_k, dtype=tf.float32)
        self.y_K = tf.constant(self.G_k, dtype=tf.float32)
        self.x_M = tfe.Variable(tf.zeros((0, self.dim)), validate_shape=False)
        self.y_M = tfe.Variable(tf.zeros((0, 1)), validate_shape=False)

    def gen_from_lambda(self, f_lambda):
        """Generate values from intensity function.

        Args:
            f_lambda (function) : A callable function for intensity

        """
        N = np.random.poisson(self.measure*self.rate)
        self.S = np.expand_dims(np.sort(np.linspace(0, self.measure, N)), axis=1)
        self.Z = f_lambda(self.S)
        self.gridPoints = self.S.reshape(1, -1)
        self.dim = 1
        self.gridN = [len(self.gridPoints[0])]
        self.type = "C"

    def gen_grid(self, lower=0, upper=20):
        """Generate a complete grid.

        Args:
            lower (float) = lower bound for draw
            upper (float) = upper bound for draw

        """
        D = self.dim
        self.S = np.random.uniform(size=(self.N_dim, self.dim), low=lower, high=upper)
#gridPoints = [np.arange(lower, upper, (1.0 * upper - lower)/(self.N_dim * 1.0)) for i in range(D)]
        self.measure = 1.0 * D * (upper - lower)
        self.gridPoints = self.S.reshape(self.dim, self.N_dim)
        self.gridN = [len(self.gridPoints[i]) for i in range(D)]
        self.type = "C"

    def constructS_k(self, sim_data):
        """Construct set of observed events.

        Retrieves set of observed events given a region and
        an intensity function. The observed events are draws from
        and inhomogenuous poisson process

        Args:
            sim_data (bool) : True if using simulated data

        Returns:
            Locations and initial function values of observed events.
        """

        N = len(self.S)
        R = np.random.uniform(0, 1, N)
        C = self.kern.K(self.S, self.S)
        if not sim_data:
            accept = np.where(R < (self.Z.flatten() / self.rate))
        else:
            self.G = np.random.multivariate_normal(np.zeros((N)), C)
            accept = np.where(R < 1 / (1.0 + np.exp(-self.G)))
        G_k = np.take(np.ones((N)), accept, axis=0).reshape(-1, 1)
        S_k = np.take(self.S, accept, axis=0).squeeze(axis=0)
        return S_k, G_k

    def update(self, x_K, x_M, y_K, y_M):
        """Update event locations and function values.

        Args:
            x_K (np.array) : Locations of observed events
            x_M (np.array) : Locations of thinned events
            y_K (np.array) : Functions values at observed locations
            y_M (np.array) : Functions values at thinned locations

        """
            
        self.x_K = x_K
        self.x_M = tfe.Variable(x_M, validate_shape=False)
        self.y_K = y_K
        self.y_M = tfe.Variable(y_M, validate_shape=False)

    def get_values(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run([tf.concat([self.x_K, self.x_M], 0), tf.concat([self.y_K, self.y_M], 0)])

    def sample_point(self, x_K, dist = "Uniform", mean = None):

        """Sample new point from region.

        Args:
            x_K (np.array) : Locations of all events observed
            dist (str) : Type of distribution to sample from
            mean (np.array) : Mean for sampling from normal distribution

        Returns:
            A point from the region

        """
        vec = np.zeros((1, self.dim), dtype=np.float32)
        x_K = np.array(x_K)
        if dist == "Uniform":
            if self.type=="C":
                return tf.random_uniform((1,self.dim), minval=0.0, maxval=self.measure)
            while(True):
                for i in range(len(self.gridN)):
                    vec[0][i] = self.gridPoints[i][np.random.choice(self.gridN[i], 1)]
                if np.min((x_K - vec[0])**2) > 1e-3:
                    return tf.convert_to_tensor(vec, dtype=tf.float32)
        elif dist == "Gaussian":
            if self.type=="C":
                return tf.random_normal((1,self.dim), mean=mean, stddev=np.sqrt(self.measure/100.0)*tf.eye(tf.shape(mean)[0]))
            while(True):
                vec = np.random.multivariate_normal(mean, np.sqrt(self.measure/10.0)*tf.eye(tf.shape(mean)[0]))
                vec = np.expand_dims(self.S[np.argmin(np.linalg.norm(self.S - vec, axis=1))], axis=0)
                if np.min((x_K - vec[0])**2) > 1e-3:
                    return tf.convert_to_tensor(vec, dtype=tf.float32)

    def conditional(self, x_new, x, y, kernel):
        """Conditional for multivariate gaussian distribution.

        Args:
            x_new (np.array) : Set of points for function value predicting
            x (np.array) : Set of observed points
            y (np.array) : Function value of observed points
            kernel : Kernel for covariance function

        Returns:
            Predicted function values for x_new

        """

        B = kernel.K(x, x_new)
        A = kernel.K(x_new, x_new)
        X = kernel.K(x, x)
        N = tf.shape(X)[0]
        mu = tf.matmul(B, tf.matmul(tf.matrix_inverse(X + 1e-6*tf.eye(N)), y), transpose_a=True)
        sigma = A - tf.matmul(B, tf.matmul(tf.matrix_inverse(X + 1e-6*tf.eye(N)), B), transpose_a=True)
        return tf.squeeze(mu), tf.squeeze(sigma)

    def add_event(self, x_new, y_new, x_M, y_M):

        """
        Add location to set of thinned events

        x_new (tf.constant) : Location of the event
        y_new (tf.constant) : Function value at x_new
        x_M (tf.Variable) : Locations of thinned events
        y_M (tf.Variable) : Values at thinned events
        
        Returns:
            Updated locations and functions values of thinned events

        """
        x_M = tf.concat([x_M, x_new], 0)
        y_M = tf.concat([y_M, y_new], 0)
        return x_M, y_M

    def erase_event(self, x_M, y_M, c):

        """
        Deletes location from set of thinned events

        Args:
            x_M (tf.Variable) : Locations of thinned events
            y_M (tf.Variable) : Values at thinned events
            c (tf.constant) : Index of the event to be deleted

        Returns:
            Updated locations and functions values of thinned events

        """

        x_M = tf.concat([tf.slice(x_M, [0, 0], [c, self.dim]), tf.slice(x_M, [c+1, 0], [-1, self.dim])], 0)
        y_M = tf.concat([tf.slice(y_M, [0, 0], [c, 1]), tf.slice(y_M, [c+1, 0], [-1, 1])], 0)
        return x_M, y_M

    def insert_event(self, x_K, y_K, x_M, y_M):

        """
        Insert event based on acceptance ratio

        Args:
            x_K (np.array) : Locations of observed events
            x_M (np.array) : Locations of thinned events
            y_K (np.array) : Functions values at observed locations
            y_M (np.array) : Functions values at thinned locations
        
        Returns:
            Updated locations and functions values of thinned events
            
        """

        M = tf.shape(x_M)[0]
        x_new = self.sample_point(tf.concat([x_K, x_M], 0))  # tf.random_uniform((1,1), minval=0.0, maxval=self.measure)
        mu_new, sigma_new = self.conditional(x_new, tf.concat([x_K, x_M], 0), tf.concat([y_K, y_M], 0), self.kern)
        y_new = tf.random_normal((1, 1), mean=mu_new, stddev=tf.sqrt(sigma_new))
        ratio = tf.log(float(self.rate * self.measure))
        ratio -= tf.log(tf.cast(M+1, tf.float32))
        ratio -= tf.log(1+tf.exp(y_new))
        a = tf.random_uniform((1,))
        x_M, y_M = tf.cond(tf.squeeze(tf.less(tf.log(a), ratio)), lambda: self.add_event(x_new, y_new, x_M, y_M), lambda: (x_M, y_M))
        return x_M, y_M

    def delete_util(self, x_M, y_M):

        """
        Delete event based on acceptance ratio

        Args:
            x_K (np.array) : Locations of observed events
            x_M (np.array) : Locations of thinned events
            y_K (np.array) : Functions values at observed locations
            y_M (np.array) : Functions values at thinned locations
        
        Returns:
            Updated locations and functions values of thinned events
            
        """

        M = tf.shape(x_M)[0]
        c = tf.random_uniform((1,), minval=0, maxval=M, dtype=tf.int32)
        c = tf.squeeze(c)
        ratio = tf.log(tf.cast(M, tf.float32))
        ratio += tf.log(1 + tf.exp(tf.slice(y_M, [c, 0], [1, 1])))
        ratio -= tf.log(float(self.rate * self.measure))
        a = tf.random_uniform((1,))
        x_M, y_M = tf.cond(tf.squeeze(tf.less(tf.log(a), ratio)), lambda: self.erase_event(x_M, y_M, c), lambda: (x_M, y_M))
        return x_M, y_M

    def delete_event(self, x_K, y_K, x_M, y_M):

        """Utility function for delete step"""
        M = tf.shape(x_M)[0]
        x_M, y_M = tf.cond(tf.equal(M, tf.constant(0)), lambda: (x_M, y_M), lambda: self.delete_util(x_M, y_M))
        return x_M, y_M

    def sample_cond(self, x_K, y_K, x_M, y_M, i):

        """Checks if all thinned locations have been iterated over"""
        return tf.less(i, tf.shape(x_M)[0])

    def sample_step(self, x_K, y_K, x_M, y_M, i):

        """
        Samples the location of a thinned event

        Args:
            x_K (tf.constant) : Locations of observed events
            y_K (tf.constant) : Functions values at observed locations
            x_M (tf.Variable) : Locations of thinned events
            y_M (tf.Variable) : Functions values at thinned locations
            i (int) : Index of the thinned event

        Returns:
            Updated locations and function values of thinned events

        """
            
        x_new = self.sample_point(tf.concat([x_K, x_M], 0), mean=x_M[i], dist="Gaussian")
# x_new = tf.random_normal((1,1), mean=x_M[i], stddev=np.sqrt(self.measure/100.0))#self.sample_point(tf.concat([x_K, x_M], 0), mean=x_M[i], dist="Gaussian")
        mu_new, sigma_new = self.conditional(x_new, tf.concat([x_K, x_M], 0), tf.concat([y_K, y_M], 0), self.kern)
        y_new = tf.random_normal((1, 1), mean=mu_new, stddev=tf.sqrt(sigma_new))
        ratio = tf.log(1 + tf.exp(y_M[i]))
        ratio -= tf.log(1 + tf.exp(y_new))
        a = tf.random_uniform((1,))
        accept = tf.squeeze(tf.less(tf.log(a), ratio))
        x_M = tf.cond(accept, lambda: tf.concat([tf.slice(x_M, [0, 0], [i, self.dim]), tf.concat([x_new, tf.slice(x_M, [i+1, 0], [-1, self.dim])], 0)], 0), lambda: x_M)
        y_M = tf.cond(accept, lambda: tf.concat([tf.slice(y_M, [0, 0], [i, 1]), tf.concat([y_new, tf.slice(y_M, [i+1, 0], [-1, 1])], 0)], 0), lambda: y_M)
        i = tf.add(i, 1)
        return x_K, y_K, x_M, y_M, i

    def thinned_cond(self, x_K, y_K, x_M, y_M, i):
        """Assesses end of while loop for sampling number of thinned events"""

        return tf.less(i, tf.constant(10))

    def thinned_step(self, x_K, y_K, x_M, y_M, i):
        """Runs one step of sampling number of thinned events

        Args:
            x_K (tf.constant) : Locations of observed events
            y_K (tf.constant) : Functions values at observed locations
            x_M (tf.Variable) : Locations of thinned events
            y_M (tf.Variable) : Functions values at thinned locations
            i (int) : Loop iterator

        Returns:
            Updated locations and function values of thinned events

        """
        x_M, y_M = tf.cond(tf.equal(self.bern.sample(), 1), lambda: self.insert_event(x_K, y_K, x_M, y_M), lambda: self.delete_event(x_K, y_K, x_M, y_M))
        i = tf.add(i, 1)
        return x_K, y_K, x_M, y_M, i

    def loop_cond(self, n_iter, it, x_K, y_K, x_M, y_M):
        """Terminates after execution of n_iter steps"""
        return tf.less(it, n_iter)

    def run(self):
    
        """Samples the number of thinned locations and its locations

        Returns:
            Set of observed and thinned events and their function values

        """

        i = tfe.Variable(0)
        self.x_K, self.y_K, self.x_M, self.y_M, i = tf.while_loop(self.thinned_cond, self.thinned_step, [self.x_K, self.y_K, self.x_M, self.y_M, i])

        # Sample thinned locations
        it = tfe.Variable(0)
        self.x_K, self.y_K, self.x_M, self.y_M, it = tf.while_loop(self.sample_cond, self.sample_step, [self.x_K, self.y_K, self.x_M, self.y_M, it])
        res = self.x_K, self.y_K, self.x_M, self.y_M

        return res

def f(x):
    return 2*np.exp(-x/15) + np.exp(-((x-25)/10.0)**2)

def run_thinnedEventsSolver(sim_data = False):
    
    kern = RBF(input_dim=1, variance=1.0, lengthscale=5.0)
    if sim_data == False:
        sampler = ThinnedEventsSampler(f_lambda=f, kern=kern, measure=50, rate=2, dim=1, N_dim=100)
    else:
        sampler = ThinnedEventsSampler(kern=kern, dim=1, N_dim=100)
        
    n_iter = 30
    for i in range(n_iter):
        x_K, y_K, x_M, y_M = sampler.run()
        K_i = len(x_K.numpy())
        M_i = len(x_M.numpy())
        S_i = tf.concat([x_K, x_M], 0)
        ind = np.argsort(S_i.numpy().flatten())
        y = np.concatenate((np.ones(K_i), np.zeros(M_i))) + 1e-4
        S_i = S_i.numpy()[ind]
        y = y[ind]
        kron = KroneckerSolver(tf.ones([S_i.shape[0]], tf.float32)*np.log((np.mean(y) + 1e-3/(1 - np.mean(y) + 1e-3))), kernels.RBF(variance=1.0, length_scale=5.0) , BernoulliSigmoidLike(), S_i, tfe.Variable(y, dtype=tf.float32))
        kron.run(20)
        val = kron.f_pred
        val = val.numpy().reshape(-1,1)
        sampler.update(x_K, x_M, val[y>1.0], val[y<1.0])
        
    return sampler, S_i, val
