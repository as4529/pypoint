import numpy as np
import tensorflow as tf
import itertools
from kern import RBF
from tensorflow.contrib.distributions import Bernoulli
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()


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

        N = np.random.poisson(self.measure*self.rate)
        self.S = np.expand_dims(np.sort(np.linspace(0, self.measure, N)), axis=1)
        self.Z = f_lambda(self.S)
        self.gridPoints = self.S.reshape(1, -1)
        self.dim = 1
        self.gridN = [len(self.gridPoints[0])]

    def gen_grid(self, lower=0, upper=10):
        D = self.dim
        gridPoints = [np.arange(lower, upper, (1.0 * upper - lower)/(self.N_dim * 1.0)) for i in range(D)]
        self.gridN = [len(gridPoints[i]) for i in range(D)]
        self.S = np.array(list(itertools.product(*gridPoints)))
        self.measure = D * (gridPoints[0][-1] - gridPoints[0][0])
        self.gridPoints = gridPoints

    def constructS_k(self, sim_data):

        N = len(self.S)
        R = np.random.uniform(0, 1, N)
        C = self.kern.K(self.S, self.S)
        G = np.random.multivariate_normal(np.zeros((N)), C)
        if not sim_data:
            accept = np.where(R < (self.Z.flatten() / self.rate))
        else:
            accept = np.where(R < 1 / (1.0 + np.exp(-G)))
        S_k = np.take(self.S, accept, axis=0).squeeze(axis=0)
        G_k = np.take(G, accept, axis=0).reshape(-1, 1)
        return S_k, G_k

    def update(self, x_K, x_M, y_K, y_M):

        self.x_K = x_K
        self.x_M = tfe.Variable(x_M, validate_shape=False)
        self.y_K = y_K
        self.y_M = tfe.Variable(y_M, validate_shape=False)

    def get_values(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run([tf.concat([self.x_K, self.x_M], 0), tf.concat([self.y_K, self.y_M], 0)])

    def sample_point(self, X, dist="Uniform", mean=None):
        vec = np.zeros((1, len(self.gridN)), dtype=np.float32)
        X = np.array(X)
        if dist == "Uniform":
            while(True):
                for i in range(len(self.gridN)):
                    vec[0][i] = self.gridPoints[i][np.random.choice(self.gridN[i], 1)]
                if not (X == vec[0]).all(-1).any():
                    return tf.constant(vec, dtype=tf.float32)
        elif dist == "Gaussian":
            while(True):
                vec = np.random.multivariate_normal(mean, np.sqrt(self.measure/10.0)*tf.eye(tf.shape(mean)[0]))
                vec = np.expand_dims(self.S[np.argmin(np.linalg.norm(self.S - vec, axis=1))], axis=0)
                if not (X == vec[0]).all(-1).any():
                    return tf.constant(vec, dtype=tf.float32)

    def conditional(self, x_new, x, y, kernel):

        B = kernel.K(x, x_new)
        A = kernel.K(x_new, x_new)
        X = kernel.K(x, x)
        N = tf.shape(X)[0]
        mu = tf.matmul(B, tf.matmul(tf.matrix_inverse(X + 1e-5 * tf.eye(N)), y), transpose_a=True)
        sigma = A - tf.matmul(B, tf.matmul(tf.matrix_inverse(X + 1e-5*tf.eye(N)), B), transpose_a=True)
        return tf.squeeze(mu), tf.squeeze(sigma)

    def add_event(self, x_new, y_new, x_M, y_M):

        x_M = tf.concat([x_M, x_new], 0)
        y_M = tf.concat([y_M, y_new], 0)
        return x_M, y_M

    def erase_event(self, x_M, y_M, c):

        x_M = tf.concat([tf.slice(x_M, [0, 0], [c, self.dim]), tf.slice(x_M, [c+1, 0], [-1, self.dim])], 0)
        y_M = tf.concat([tf.slice(y_M, [0, 0], [c, 1]), tf.slice(y_M, [c+1, 0], [-1, 1])], 0)
        return x_M, y_M

    def insert_event(self, x_K, y_K, x_M, y_M):

        M = tf.shape(x_M)[0]
        x_new = self.sample_point(tf.concat([x_K, x_M], 0))  # tf.random_uniform((1,1), minval=0.0, maxval=self.measure)
        print(x_new.shape)
        mu_new, sigma_new = self.conditional(x_new, tf.concat([x_K, x_M], 0), tf.concat([y_K, y_M], 0), self.kern)
        y_new = tf.random_normal((1, 1), mean=mu_new, stddev=tf.sqrt(sigma_new))
        ratio = tf.log(float(self.rate * self.measure))
        ratio -= tf.log(tf.cast(M+1, tf.float32))
        ratio -= tf.log(1+tf.exp(y_new))
        a = tf.random_uniform((1,))
        x_M, y_M = tf.cond(tf.squeeze(tf.less(tf.log(a), ratio)), lambda: self.add_event(x_new, y_new, x_M, y_M), lambda: (x_M, y_M))
        return x_M, y_M

    def delete_util(self, x_M, y_M):

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

        M = tf.shape(x_M)[0]
        x_M, y_M = tf.cond(tf.equal(M, tf.constant(0)), lambda: (x_M, y_M), lambda: self.delete_util(x_M, y_M))
        return x_M, y_M

    def sample_cond(self, x_K, y_K, x_M, y_M, i):

        return tf.less(i, tf.shape(x_M)[0])

    def sample_step(self, x_K, y_K, x_M, y_M, i):

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

        return tf.less(i, tf.constant(10))

    def thinned_step(self, x_K, y_K, x_M, y_M, i):

        x_M, y_M = tf.cond(tf.equal(self.bern.sample(), 1), lambda: self.insert_event(x_K, y_K, x_M, y_M), lambda: self.delete_event(x_K, y_K, x_M, y_M))
        i = tf.add(i, 1)
        return x_K, y_K, x_M, y_M, i

    def loop_cond(self, n_iter, it, x_K, y_K, x_M, y_M):

        return tf.less(it, n_iter)

    def run(self):

        i = tfe.Variable(0)
        self.x_K, self.y_K, self.x_M, self.y_M, i = tf.while_loop(self.thinned_cond, self.thinned_step, [self.x_K, self.y_K, self.x_M, self.y_M, i])

        # Sample thinned locations
        it = tfe.Variable(0)
        self.x_K, self.y_K, self.x_M, self.y_M, it = tf.while_loop(self.sample_cond, self.sample_step, [self.x_K, self.y_K, self.x_M, self.y_M, it])
        res = self.x_K, self.y_K, self.x_M, self.y_M

        return res

#class ThinnedEventsSolver():
#
#   def __init__(self, S, kern, measure, rate):
#
#       self.S_ph = tf.placeholder(dtype=tf.float32, name="S")
#       self.C = kern.K(self.S_ph, self.S_ph)
#       self.K_ph = tf.placeholder(dtype=tf.int32, name="K")
#       self.M_ph = tf.placeholder(dtype=tf.int32, name="M")
#       self.N = self.K_ph + self.M_ph
#       self.S = S
#       self.measure = measure
#       self.kern = kern
#       self.rate = rate
#
#   def get_optimizer(self, G):
#   
#       F = tfe.Variable(G, validate_shape=False)
#       prior_loss = 0.5 * tf.matmul(tf.transpose(F), tf.matmul(tf.matrix_inverse(self.C + 1e-6*tf.eye(self.N)), F))
#       prior_loss = tf.squeeze(prior_loss)
#       likelihood_loss = tf.reduce_sum(tf.log(tf.ones([self.K_ph,1]) + tf.exp(-tf.slice(F, [0,0], [self.K_ph,1]))))
#       likelihood_loss += tf.reduce_sum(tf.log(tf.ones([self.M_ph,1]) + tf.exp(tf.slice(F, [self.K_ph,0], [self.M_ph,1]))))
#       loss = prior_loss + likelihood_loss
#       train_op = tf.train.AdadeltaOptimizer(0.1, 0.95, 1e-5).minimize(loss)
#       init_op = tf.global_variables_initializer()
#       self.F = F
#       return init_op, train_op, loss
#
#   def step(self, opt_iter):
#
#
#       x_K, y_K, x_M, y_M = self.Sampler.run()
#       S_i = np.concatenate([x_K, x_M], axis=0)
#       
#       G_i = np.concatenate([y_K, y_M], axis=0)
#       K_i = len(x_K)
#       M_i = len(x_M)
#       init_op, train_op, loss = self.get_optimizer(G_i)
#       
#       with tf.Session() as sess:
#           sess.run(init_op)
#           for i in range(opt_iter):
#               sess.run(train_op, feed_dict={self.S_ph:S_i, self.K_ph:len(x_K), self.M_ph:len(x_M)})
#       
#           val = sess.run(self.F)
#
#       self.Sampler.update(x_K, x_M, val[:K_i], val[K_i:])
#       
#
#   def solve(self, n_iter = 30, opt_iter = 500):
#
#       self.Sampler = ThinnedEventsSampler(self.S, self.kern, self.measure, self.rate)
#       
#       for i in range(n_iter):
#
#           print(i)
#           self.step(opt_iter)
#       
#       res_S, res_G = self.Sampler.get_values()
#       return res_S, res_G
