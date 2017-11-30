import numpy as np
import itertools
import tensorflow as tf


def linear_interpolate(X, U, kernel):
    """
    performs linear kernel interpolation
    Note: this assumes a regular (equispaced) grid for U
    Args:
        X (): observed points
        U (): grid locations
        kernel (): kernel function

    Returns:

    """

    return 0


def find_nn(X, U, k):

    """
    Finds the k nearest neigbhors in U for each point in X
    Args:
        X (): observed points
        U (): grid locations
        N (): number of neighbors desired

    Returns:

    """

    distance = tf.reduce_sum(tf.square(tf.subtract(U, tf.expand_dims(X, 1))), axis=2)
    top_k_vals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)

    return top_k_vals, top_k_indices


def fill_grid(X, y):
    """
    Fills a partial grid with "imaginary" observations
    Args:
        X (np.array): data that lies on a partial grid

    Returns:
        X_grid: full grid X (including real and imagined points)
        y_full: full grid y (with zeros corresponding to imagined points)
        obs_idx: indices of observed points
        imag_idx: indices of imagined points

    """

    D = X.shape[1]
    x_dims = [np.unique(X[:, d]) for d in range(D)]

    X_grid = np.array(list(itertools.product(*x_dims)))

    d_indices = [{k: v for k, v in zip(x_dims[d], range(x_dims[d].shape[0]))} for d in range(D)]
    grid_part = np.ones([x_d.shape[0] for x_d in x_dims])*-1

    for i in range(X.shape[0]):
        idx = tuple([d_indices[d][X[i, d]] for d in range(D)])
        grid_part[idx] = 1

    obs_idx = np.where(grid_part.flatten() > -1)[0]
    imag_idx = np.where(grid_part.flatten() == -1)[0]

    y_full = np.zeros(X_grid.shape[0])
    y_full[obs_idx] = y

    return X_grid, y_full, obs_idx, imag_idx


def get_partial_grid(X, y=None, prop=0.3):

    D = X.shape[1]
    x_dims = [np.unique(X[:, d]) for d in range(D)]

    grid_part = np.zeros([x_d.shape[0] for x_d in x_dims])

    return 0
