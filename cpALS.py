import numpy as np
import scipy.io
import copy
from subprocess import call
import os

from utils import *


def cpALS(TS=None, R=None, option={}):
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
        option['init'] = 'random'
        option['firstItrDims'] = []
        # the overall iterating dimensions
        # (fixed dimensions should be the complement of this set from the whole set 0:N-1)
        option['itrDims'] = []
        # convergence criterion
        option['tol'] = 1e-5
        option['maxNumItr'] = 100
        # whether to cache the matricized tensor or not
        # caching will speed up the computation but require more memory
        option['cacheMTS'] = True
        # constrained dim
        # 0: no constraint, 1: smoothness, 2:sparseness, 3: TV
        option['const'] = []
        # regularization params for each dim
        option['regParam'] = []
        # non-negative constraint
        option['nonnegative'] = []
        # if output the history of cost func. value
        option['isCostFV'] = False
        # if output the history of the convergence
        option['isFC'] = False
        # print every #printItv, the first 5 iters always printed
        option['printItv'] = 10

        # settings for TFOCS
        option['epsHuber'] = 1e-3

        optTFOCS = {}
        optTFOCS['tol'] = 1e-6
        optTFOCS['restart'] = 5
        optTFOCS['maxIts'] = 100
        optTFOCS['alg'] = 'AT'
        optTFOCS['printEvery'] = 0
        optTFOCS['debug'] = False
        option['optTFOCS'] = optTFOCS

        lamb = []
        output = []

        return option

    # get dimensions for init
    sz = TS.shape
    N = TS.ndim

    # to check with matlab version
    np.random.seed(1337)

    # init set
    # TODO: add other choice of init (e.g. use SVD), now it's just random init
    UInit = [[] for i in range(N)]
    for m in range(N):
        # UInit[m] = np.random.rand(TS.shape[m], R)
        UInit[m] = np.random.rand(R, TS.shape[m]).T # so to generate same random matrix as matlba

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

    if not len(option['const']):
        option['const'] = np.zeros((N, 1))
    else:
        if len(option['const']) != N:
            raise ValueError("regularization type dimension mismatches the data")

    if not len(option['regParam']):
        option['regParam'] = np.zeros((N, 1))
    else:
        if len(option['regParam']) != N:
            raise ValueError("regularization parameter dimension mismatches the data")

    if not len(option['nonnegative']):
        option['nonnegative'] = np.zeros((N, 1), dtype=bool)
    else:
        if len(option['nonnegative']) != N:
            raise ValueError("nonnegativity constraint dimension mismatches the data")

    # init params
    maxNumItr = option['maxNumItr']
    const = option['const']
    mu = option['regParam']
    epsHuber = option['epsHuber']

    isCostFV = option['isCostFV']
    isFC = option['isFC']
    printItv = option['printItv']

    # start ALS
    U = UInit

    if isCostFV: CostFV = np.zeros((maxNumItr, 1))
    if isFC: FC = np.zeros((maxNumItr, 1))

    V = np.zeros((R, R, N))
    for m in range(N):
        V[:, :, m] = np.matmul(np.array(U[m]).T, np.array(U[m]))

    for m in range(maxNumItr):
        UOld = copy.deepcopy(U);

        # iterate over modes specified for first iter
        for n in firstItrDims:
            idx = np.concatenate((np.arange(0, n), np.arange(n+1, N)))

            A = np.prod(V[:, :, idx], axis=2)

            if option['cacheMTS']:
                cur_MTS = MTS[n]
                cur_krprod = KrProd([U[i] for i in idx[::-1]])
                B = np.matmul(MTS[n], KrProd([U[i] for i in idx[::-1]])).T
            else:
                B = np.matmul(matricize(TS, n), KrProd([U[i] for i in idx[::-1]])).T

            # TODO: no regularization options for now, include in the future
            A2 = nBlockDiag(A, sz[n])
            B2 = np.reshape(B, (B.size, 1), order='F')
            x0 = U[n].T
            x0 = np.reshape(x0, (x0.size, 1), order='F')

            # TODO: now enforces nonnegativity,
            # in the future includes the one w/o this constraint
            scipy.io.savemat("var.mat", mdict={'A2': A2, 'B2':B2, 'x0':x0})
            # call(["matlab",  "-nodisplay", "-nosplash", "-r", "TFOCS_LS(); exit;"])
            os.system('matlab -nojvm -nodisplay -nosplash -r "TFOCS_LS;quit;"')
            tfocs_res_mat = scipy.io.loadmat("tfocs_rst.mat")
            X2 = tfocs_res_mat['X2']

            X = np.reshape(X2, (R, sz[n]), order='F')
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
            print("Iter%d: eps = %.3e\n" % (m + 1, facCvg))

        if (m > 1) and (facCvg < option['tol']):
            break

    if printItv > 0:
        if (m + 1) == maxNumItr:
            print("reached the max number of iterations")
            print("eps = %.3e\n", facCvg)

            # TODO: add dfFRo, EV
            # print("f = %.3f, eps = %.3e  EV = %.2f\n", dfFro^2, facCvg, EV))

    output = {}
    output['Flag'] = True
    output['numItr'] = m + 1

    return U, lamb, output
