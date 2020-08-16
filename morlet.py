import math
import numpy as np

PI = math.pi


def cMorletWavelet(t, fc, sigma):
    """
    params:
        t (numpy array): time axis
        fc (float): center frequency
        sigma (float): gassian sigma

    return:
        M (numpy array): morlet kernel
    """

    M = (2 * PI * sigma**2)**(-0.5) * np.exp(-(t**2) / (2 * sigma**2)) * np.exp(1j * 2 * PI * fc * t)

    return M
