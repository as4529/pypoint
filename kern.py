import numpy as np
import tensorflow as tf
class RBF():
    
    def __init__(self, input_dim, variance=1., lengthscale=1.):
        self.input_dim = input_dim
        self.variance = variance
        self.lengthscale = lengthscale
        
    def K(self, X, X2):
        return self.variance * tf.exp(-self.square_dist(X, X2) / 2)
        
    def square_dist(self, X, X2):
        X = X / self.lengthscale
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, tf.transpose(X)) + tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscale
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, tf.transpose(X2)) + tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

class Periodic():
    
    def __init__(self, input_dim, period=1.0, variance=1.0, lengthscales=1.0):
        self.input_dim = input_dim
        self.period = period
        self.variance = variance
        self.lengthscales = lengthscales
        
    def K(self, X, X2):
        
        if X2 is None:
            X2 = X
            
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D

        r = np.pi * (f - f2) / self.period
        r = tf.reduce_sum(tf.square(tf.sin(r) / self.lengthscales), 2)

        return self.variance * tf.exp(-0.5 * r)
