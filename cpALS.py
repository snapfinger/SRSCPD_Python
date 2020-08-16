import numpy as np


def cpALS(TS, R, option={}):
    if not option:
        option['init'] = 'random'
        option['firstItrDims'] = []
        option['itrDims'] = []
        option['tol'] = 1e-5
        option['maxNumItr'] = 100
        # whether to cache the matricized tensor or not
        # caching will speed up the computation but require more memory
        option['cacheMTS'] = True
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

        optTFOCS['tol'] = 1e-6
        optTFOCS['restart'] = 5
        optTFOCS['maxIts'] = 100
        optTFOCS['alg'] = 'AT'
        optTFOCS['printEvery'] = 0
        optTFOCS['debug'] = False
        option['optTFOCS'] = optTFOCS

        lamb = []
        output = []

    # get dimensions for init
    sz = TS.shape
    N = TS.ndim

    # init set
    # TODO: add other choice of init (e.g. use SVD), now it's just random init
    UInit = [[] for i in range(N)]
    for m in range(N):
        UInit[m] = np.random.rand(TS.shape[m], R)

    return UInit

    if option['cacheMTS']:
        MTS = [[] for i in range(N)]
        for m in range(N):
            MTS[m] = matricize(TS, m)
    else:
        MTS = []


    # # init params
    # maxNumItr = option['maxNumItr']
    # epsHuber = option['epsHuber']
    # printItv = option['printItv']
    #
    # # start ALS
    # for m in range(maxNumItr):


a = np.ones((3, 2))
res = cpALS(a, 3)
print(res)
