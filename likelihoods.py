import tensorflow as tf

"""
basic likelihood class
"""

class PoissonLike:
    """
    Implements Poisson likelihood
    """
    def log_like(self, y, log_rate):

        return tf.multiply(y, log_rate) - tf.exp(log_rate)

    def grad(self, y, log_rate):

        return y - tf.exp(log_rate)

    def hess(self, y, log_rate):

        return -tf.exp(log_rate)

class GaussianLike:

    def __init__(self, variance):

        self.variance = variance

    def log_like(self, y, mean):

        return -1/(2*self.variance)*tf.square(y-mean)

