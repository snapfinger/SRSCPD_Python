from cpALS import cpALS
from utils import *


def srscpd(TS=None, R=None, option=None):
    """
    SRSCPD framework

    params:
        TS (numpy array): tensor
        R (int): desired rank
        option (dictionary): contains more settings

    return:
        result (dictionary): decomposition results from rank 1 to R
    """
    if option is None:
        option = {'maxNumFitRes': 10,
                  'isVerbose': True,
                  'rank1Method': 'als',
                  'rank1Init': 'random',
                  'alg': 'als',
                  'optAlg': cpALS()
                }

        return option

    optALS = option['optAlg']

    maxNumFitRes = option['maxNumFitRes']
    isVerbose = option['isVerbose']
    rank1Method = option['rank1Method']
    rank1Init = option['rank1Init']

    if not isVerbose: optALS['printItv'] = 0

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
    while (not output['Flag']) and (c < maxNumFitRes):
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

    rst_dict = {'U': U,
                'Lambda': lamb,
                'Output': output}
    result.append(rst_dict)

    # compute the residual
    TSRes = TS - cpFull(U, lamb, False)

    # iterate over the rest of the ranks
    N = len(U)

    for m in range(2, R + 1):
        if isVerbose: print("Start fitting rank-1 tensor to residue as part of warm start...")

        # TODO: now only init rank1 using random in ALS, extend options in the future
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
            # tranpose so to generate same random matrix as matlab with this random seeed
            URes = [np.random.rand(1, TS.shape[n]).T for n in range(N)]

        # equally spread the scale lambda to each dimension
        UInit = []

        for n in range(N):
            a = U[n]
            b = URes[n]
            part1 = a * (lamb ** (1/N)).T
            part2 = b * (lambdaRes ** (1/N)).T
            UInit.append(np.hstack((part1, part2)))

        # fit rank-r tensor
        if isVerbose: print("Fit rank %s tensor" % m)

        # TODO: only use als here, add other option later
        optAlg2 = optALS
        optAlg2['init'] = UInit

        U, lamb, output = cpALS(TS, m, optAlg2)

        cur_dict = {'U': U,
                    'Lambda': lamb,
                    'Output': output}
        result.append(cur_dict)

        # compute residue
        TSRes = TS - cpFull(U, lamb, False)

    return result
