import tensorflow as tf
import sys
from kronecker_utils import kron_mvp

class CG_optimizer:

    def __init__(self, Ks, likelihood, tau):

        self.likelihood = likelihood
        self.Ks = Ks
        self.tau = tau

    def cg_converged(self, A, p, r_k_norm, count, x, r, n):
        return tf.logical_and(tf.greater(r_k_norm, 1e-5), tf.less(count, n))

    def cg_body(self, A, p, r_k_norm, count, x, r, n):

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

        return tf.logical_and(tf.less(t, max_it), tf.less(obj_prev - obj_search, 0.5 * step_size * grad_norm))


    def line_search(self, alpha, delta_alpha, y, obj_prev, max_it, mu):
        obj_search = sys.float_info.max
        min_obj = obj_prev

        step_size = 2.0
        opt_step = 0.0

        grad_norm = tf.reduce_sum(tf.multiply(alpha, alpha))
        t = 1

        res = tf.while_loop(self.converge_cond, self.search_step, [obj_prev, obj_search, min_obj, alpha, delta_alpha,
                                                         y, step_size, grad_norm, max_it, t, mu, opt_step])

        return res[2], res[-1]