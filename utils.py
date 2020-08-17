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


def KrProd(U):
    """
    Khatri-Rao product (N-dim supported)

    params:
        U (list of lists) the components

    return:
        KR:
    """
    N = len(U)
    cols = np.zeros((N, 1))
    for m in np.arange(N):
        cols[m] = U[m].shape[1]

    KR = U[0]
    R = cols[0]

    for m in np.arange(1, N):
        KR =


    KR = np.reshape(KR, (-1, R), order='F')

    return KR
