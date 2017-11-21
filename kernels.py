import tensorflow as tf

class RBF:

    def __init__(self, variance, length_scale):
        """
        modified from Edward to work with eager excecution
        Args:
            sigma ():
            length_scale ():
        """

        self.length_scale = length_scale
        self.variance = variance

    def eval(self, X, X2 = None):

        """
        Taken from Edward, without parts that break during eager excecution
        Args:
            X ():
            X2 ():

        Returns:

        """

        X = tf.convert_to_tensor(X, dtype = tf.float32)
        X = X / self.length_scale
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            X2 = X
            X2s = Xs
        else:
            X2 = tf.convert_to_tensor(X2, dtype = tf.float32)
            X2 = X2 / self.length_scale
            X2s = tf.reduce_sum(tf.square(X2), 1)

        square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - \
                 2 * tf.matmul(X, X2, transpose_b=True)
        output = self.variance * tf.exp(-square / 2)

        return output

    def params(self):

        return [self.variance, self.length_scale]

class SpectralMixture:

    def __init__(self, w, mu, v):

        self.w = w
        self.mu = mu
        self.v = v

    def eval(self, X_1, X_2):

        return 0
