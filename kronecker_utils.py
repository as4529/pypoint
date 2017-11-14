import tensorflow as tf
import numpy as np

"""
utility functions for kronecker inference
"""


def kron(A, B):
    """
    Kronecker product of two matrices
    TODO: implement this in tensorflow
    Args:
        A (tf.Variable): first matrix for kronecker product
        B (tf.Variable): second matrix

    Returns: kronecker product of A and B

    """
    return tf.py_func(np.kron, [A, B], tf.float32)


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

