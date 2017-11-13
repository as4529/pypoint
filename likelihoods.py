import tensorflow as tf


class Likelihood:

    def log_like(self, y, param):
        pass

    def grad(self, y, param):
        pass

    def hess(self, y, param):
        pass

class PoissonLike(Likelihood):

    def log_like(self, y, log_rate):

        return tf.multiply(y, log_rate) - tf.exp(log_rate)

    def grad(self, y, log_rate):

        return y - tf.exp(log_rate)

    def hess(self, y, log_rate):

        return -tf.exp(log_rate)