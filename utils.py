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
        KR = np.multiply(np.reshape(U[m], (-1, 1, R)), np.reshape(KR, (1, -1, R)))

    KR = np.reshape(KR, (-1, R), order='F')

    return KR


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
