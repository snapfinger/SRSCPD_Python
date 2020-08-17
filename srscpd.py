import numpy as np


def srscpd(TS, R, option):
    """
    params:
        TS (numpy array): tensor
        R (int): desired rank
        option (dictionary): contains more settings
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
