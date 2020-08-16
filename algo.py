import numpy as np


def cpALS(TS, R, option={}):
    if not option:
        option['init'] = 'random'
        option['firstItrDims'] = []
        option['itrDims'] = []
        option['tol'] = 1e-5
        option['maxNumItr'] = 100
        option['cacheMTS'] = True
        option['const'] = []
        option['printItv'] = 10

        # settings for TFOCS
        option['epsHuber'] = 1e-3

    # get dimensions for init
    sz = TS.shape
    N = TS.ndim

    # init set
    # TODO: add choice of init using SVD, now it's just random init
    UInit = [[] for i in range(N)]
    for m in range(N):
        UInit[m] = np.random.rand(TS.shape[m], R)

    return UInit, option


    # # init params
    # maxNumItr = option['maxNumItr']
    # epsHuber = option['epsHuber']
    # printItv = option['printItv']
    #
    # # start ALS
    # for m in range(maxNumItr):


a = np.ones((3, 2))
res, op = cpALS(a, 2)
print(res)
print(op)
