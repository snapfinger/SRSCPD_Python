import numpy as np
import numpy.linalg as LA


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


def tsFroNorm(X, mode=1):
	"""
	Frobenius norm (or square) of the tensor (N-D supported)
	Args:
		X (numpy array): the full tensor
		mode (int): 1 for norm, 2 for squared norm
	Returns:
		fbn (float): (squared) frobenius norm
	"""
	X2 = np.reshape(X, (X.size, 1))

	if mode == 1:
		fbn = np.sqrt(np.dot(X2.T, X2))
	elif mode == 2:
		fbn = np.dot(X2.T, X2)
	else:
		raise ValueError("Incorrect mode input")

	return fbn[0][0]


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


def cpDiff(X, U, lamb):
	"""
	Calculate the difference between tensor and approximations (N-D supported)
	Args:
		X: (numpy array) the full tensor
		U: (list of numpy arrays) the decomposed components
		lamb: ()
	Returns:
		df (float):
		err (float):
		rela_err (float):
	"""
	U_tmp = [U[i].copy() for i in range(len(U))]
	N = len(U_tmp)
	Xnorm = tsFroNorm(X)
	lmd2 = lamb ** (1/N)
	for m in range(N):
		U_tmp[m] *= lmd2.T

	A = matricize(X, 0) - np.matmul(U_tmp[0], KrProd(U_tmp[len(U_tmp)-1: 0: -1]).T)
	df = LA.norm(A, 'fro')

	rela_err = df / Xnorm
	varExp = 1 - rela_err ** 2

	return df, rela_err, varExp


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
