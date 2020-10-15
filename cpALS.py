import numpy as np
import copy

from utils import matricize, KrProd, cpDiff


def cpALS(TS=None, R=None, option=None):
    """
    Alternating least sqaure (ALS) algorithm for CP decomposition

    params:
        TS (numpy array): N-way tensor
        R (int): desired rank
        option (dictionary): contains more settings

    return:
        U (list of numpy arrays): the componetns
        lambda (numpy array): scales corresponding to the components
        output (dictionary): record of algo related info
    """
    if not option:
        option = {'init': 'random',
                  'firstItrDims': [],
                  # the overall iterating dimensions
                  # (fixed dimensions should be the complement of this set from the whole set 0:N-1)
                  'itrDims': [],
                  # convergence criterion
                  'tol': 1e-5,
                  'maxNumItr': 100,
                  # whether to cache the matricized tensor or not
                  # caching will speed up the computation but require more memory
                  'cacheMTS': True,
                  # non-negative constraint
                  'nonnegative': [],
                  # if output the history of cost function value
                  'isFC': False,
                  # print every #printItv, the first 5 iters always printed
                  'printItv': 10}

        return option

    # get dimensions for init
    N = TS.ndim

    # to check with matlab version
    np.random.seed(1337)

    # init set
    if isinstance(option['init'], str):
        # TODO: add other choice of init (e.g. use SVD), now it's just random init
        # transpose so to generate same random matrix as matlab
        UInit = [np.random.rand(R, TS.shape[m]).T  for m in range(N)]
        # UInit = [np.random.rand(TS.shape[m], R) for m in range(N)]
    else:
        # or with specified input
        UInit = option['init']

    if option['cacheMTS']:
        MTS = [[] for i in range(N)]
        for m in range(N):
            MTS[m] = matricize(TS, m)
    else:
        MTS = []

    if not option['firstItrDims']:
        firstItrDims = np.arange(N)
    else:
        firstItrDims = option['firstItrDims']

    if not option['itrDims']:
        itrDims = np.arange(N)
    else:
        itrDims = option['itrDims']

    if not len(option['nonnegative']):
        option['nonnegative'] = np.zeros((N, 1), dtype=bool)
    else:
        if len(option['nonnegative']) != N:
            raise ValueError("nonnegativity constraint dimension mismatches the data")

    # init params
    maxNumItr = option['maxNumItr']

    isFC = option['isFC']
    printItv = option['printItv']

    # start ALS
    U = UInit

    if isFC: FC = np.zeros((maxNumItr, 1))

    V = np.zeros((R, R, N))
    for m in range(N):
        V[:, :, m] = np.matmul(np.array(U[m]).T, np.array(U[m]))

    for m in range(maxNumItr):
        UOld = copy.deepcopy(U)

        # iterate over modes specified for first iter
        for n in firstItrDims:
            idx = np.concatenate((np.arange(0, n), np.arange(n+1, N)))

            A = np.prod(V[:, :, idx], axis=2)

            if option['cacheMTS']:
                B = np.matmul(MTS[n], KrProd([U[i] for i in idx[::-1]])).T
            else:
                B = np.matmul(matricize(TS, n), KrProd([U[i] for i in idx[::-1]])).T

            X = np.matmul(np.linalg.pinv(A), B)
            if n >= 0:
                X[X < 0] = 0

            lamb = np.sqrt(np.diag(np.matmul(X, X.T)))
            lamb = np.reshape(lamb, (lamb.shape[0], 1), order='F')
            X = np.divide(X, lamb)

            U[n] = X.T
            V[:, :, n] = np.matmul(X, X.T)

        # check factor convergence (abs diff per element)
        facCvg, numE = 0.0, 0.0
        for n in range(N):
            facCvg += np.sum(np.abs(U[n] - UOld[n]))
            numE += U[n].size
        facCvg /= float(numE)

        if isFC: FC[m] = facCvg

        if printItv and ((m <= 4) or ((m + 1) % printItv == 0)):
            print("Iter%d: eps = %.3e" % (m + 1, facCvg))

        firstItrDims = itrDims

        if (m > 1) and (facCvg < option['tol']):
            break

    dfFro, _, EV = cpDiff(TS, U, lamb)

    if printItv > 0:
        if (m + 1) == maxNumItr:
            print("reached the max number of iterations")
            print("eps = %.3e\n", facCvg)
        else:
            print("f = %.3f, eps = %.3e  EV = %.2f\n" % (dfFro**2, facCvg, EV))

    output = {'Flag': True,
              'numItr': m + 1}

    return U, lamb, output
