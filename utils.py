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
        U (list of numpy arrays): the components

    return:
        KR (numpy array): Khatri-Rao product
    """
    N = len(U)
    cols = np.zeros((N, 1))
    for m in np.arange(N):
        cols[m] = np.array(U[m]).shape[1]

    if not np.all(cols == cols[0]):
        raise ValueError("number of columns does not match")

    KR = U[0]
    R = int(cols[0])

    for m in np.arange(1, N):
        # element-wise product between reshaped tensors
        KR = np.multiply(np.reshape(U[m], (-1, 1, R), order='F'), np.reshape(KR, (1, -1, R), order='F'))

    KR = np.reshape(KR, (-1, R), order='F')

    return KR


def cpFull(U, lambda_, isLimitedMem=True):
    """
    Reconstruct the full tensor based on the components (N-D)

    params:
        U (list of (2d) numpy arrays): the decomposed compoents
        lambda_ (1d numpy array of float): the scales corresponding to the components
        isLimitedMem (bool): whether use limited memory mode or not

    return:
        Y (numpy array): the reconstructed full tensor
    """
    N = len(U)
    sz = np.zeros(N).astype(np.int32)
    R = U[0].shape[1]

    for m in np.arange(N):
        sz[m] = U[m].shape[0]

    if isLimitedMem:
        Y = np.zeros(sz)

        for m in range(R):
            T = U[0][:, m]
            if T.ndim == 1:
                T = np.reshape(T, (T.shape[0], 1))
            for n in range(1, N):
                s = np.ones(N).astype(np.int32)
                s[n] = sz[n]
                v = np.reshape(U[n][:, m], s, order='F')
                v = np.squeeze(v)
                T = np.tensordot(T, v, axes=0)
                T = np.squeeze(T)
            Y += lambda_[m] * T

    else:
        Y = np.reshape(np.matmul(KrProd(U[::-1]), lambda_), sz, order='F')

    return Y


# TODO: include option of using sparse matrix representation
def nBlockDiag(A, n):
    """
    Construct block diagonal matrix

    params:
        A (numpy array): the dense block matrix
        n (int): how many blocks to repeat

    return:
        (numpy array): the constructed block diagonal matrix
    """

    return np.kron(np.eye(n), A)


# TODO
def linop_diff_h(sz):
    """
    construct a linear differential operator

    params:
        sz (tuple): size of the matrix

    return:
        op (): TODO
    """
    pass
