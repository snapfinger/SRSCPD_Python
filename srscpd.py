import numpy as np

from cpALS import *
from utils import *


def srscpd(TS=None, R=None, option={}):
    """
    SRSCPD framework
    
    params:
        TS (numpy array): tensor
        R (int): desired rank
        option (dictionary): contains more settings

    return:
        result (dictionary): decomposition results from rank 1 to R
    """
    # TODO: expand, now just perform with ALS
    if not option:
        option = {}
        option['isStats'] = False
        option['maxNumFitRes'] = 10
        option['isVerbose'] = True
        option['rank1Method'] = 'als'
        option['rank1Init'] = 'random'
        option['alg'] = 'als'
        option['optAlg'] = cpALS()

        optALS = option['optAlg']
        if option['alg'] == 'als':
            optALS['printItv'] = optAlg['printItv']
            optALS['maxNumItr'] = optAlg['maxNumItr']
            optALS['cacheMTS'] = optAlg['cacheMTS']
        else:
            raise ValueError("To be implemented")

        optALS['nonnegative'] = optAlg['nonnegative']
        optALS['init'] = optAlg['init']

        if not option['isVerbose']: optALS['printItv'] = 0

        return option

    isStats = option['isStats']
    maxNumFitRes = option['maxNumFitRes']
    isVerbose = option['isVerbose']
    rank1Method = option['rank1Method']
    rank1Init = option['rank1Init']

    # ------------- start SRSCPD -------------
    result = []

    if isVerbose: print("Fit rank 1 tensor as basis...")

    # fit the first rank-1 tensor
    if rank1Method == "als":
        U, lamb, output = cpALS(TS, 1, optALS)
    else:
        raise ValueError("To be implemented")

    # refit if failed
    c = 1
    while ((not output['Flag']) and (c < maxNumFitRes)):
        c += 1
        if isVerbose: print("R=1 decomposition failed, try again (%s)" % c)
        if rank1Method == "als":
            U, lamb, output = cpALS(TS, 1, optALS)
        else:
            raise ValueError("To be implemented")

    # if failed too many times at the first round, just quit
    if not output['Flag']:
        if isVerbose: print("Still failed to fit the first rank-1 tensor, quit")
        return

    dict = {}
    dict['U'] = U
    dict['Lambda'] = lamb
    dict['Output'] = output
    result.append(dict)

    # TODO: calculate stats if needed
    if isStats: pass

    # compute the residue
    TSRes = TS - cpFull(U, lamb, False)

    # iterate over the rest of the ranks
    N = len(U)

    for m in range(1, R):
        if isVerbose: print("Start fitting rank-1 tensor to residue as part of warm start...")

        # TODO: now only init rank1 using random in ALS, extend optiosn in the future
        optALS["init"] = rank1Init

        if rank1Method == "als":
            URes, lambdaRes, output = cpALS(TSRes, 1, optALS)
        else:
            raise ValueError("To be implemented")

        # re-fit if failed
        c = 1
        while (not output['Flag']) and (c < maxNumFitRes):
            c += 1
            if isVerbose: print("Didn't find good init from residue, try again (%s)" % c)
            if rank1Method == "als":
                URes, lambdaRes, output = cpALS(TSRes, 1, optALS)
            else:
                raise ValueError("To be implemented")

        np.random.seed(1337)
        # just use random if still failed
        if not output['Flag']:
            if isVerbose: print("Still could not find good init, just use random")
            URes = []
            for n in range(N):
                URes.append(np.random.rand(1, TS.shape[n]).T) # so to generate same random matrix as matlba)

        # equally spread the scale lambda to each dimension
        UInit = []
        for n in range(N):
            a = U[n]
            b = URes[n]
            # TODO soon
            part1 = np.tensordot(a, np.multiply(lamb,  1/N).T)
            part2 = np.tensordot(b, np.multiply(lambdaRes, 1/N).T)
            UInit.append(np.hstack(part1, part2))

        # fit rank-r tensor
        if isVerbose: print("Fit rank %s tensor" % m)

        # TODO: only use als here, add other option later
        optAlg2 = optALS
        optAlg2['init'] = UInit

        U, lamb, output = cpALS(TS, m, optAlg2)

        cur_dict = {}
        cur_dict['U'] = U
        cur_dict['Lambda'] = lamb
        cur_dict['Output'] = output
        result.append(cur_dict)

        # stats TODO

        # compute residue
        TSRes = TS - cpFull(U, lamb)

    return result
