import tensorflow as tf
import numpy as np


def kron(A, B):

    return tf.py_func(np.kron, [A, B], tf.float32)


def kron_mvp(Ks, v):

    if len(Ks) == 1:
        return tf.matmul(Ks[0], tf.expand_dims(v, 1))

    k_0 = Ks[0]
    V_rows = k_0.shape[0]
    V_cols = v.shape[0] / V_rows
    V = tf.transpose(tf.reshape(v, (V_rows, V_cols)))

    prod = tf.matmul(V, tf.transpose(k_0))
    mvp = tf.zeros(shape=[0, 1])

    for col in range(prod.shape[1]):
        mvp = tf.concat([mvp, kron_mvp(Ks[1:], prod[:, col])], 0)

    return mvp


def kron_list(matrices):

    out = kron(matrices[0], matrices[1])

    for i in range(2, len(matrices)):
        out = kron(out, matrices[i])

    return out