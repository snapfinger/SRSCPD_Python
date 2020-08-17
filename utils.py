import numpy as np


def matricize(X, n):
    """
    Matricization along the n-th dim (N-dim supported)

    params:
        X (numpy array): the full tensor
        n (int): along which dim

    return:
        Xn (numpy array): the matricized tensor X (a matrix)

    """

    N = X.ndim
    if n < 0 or n >= N:
        raise ValueError("mode error")

    Y = np.moveaxis(X, n, 0)
    Xn = np.reshape(Y, (Y.shape[0], -1), order='F')

    return Xn
