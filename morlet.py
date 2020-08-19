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


def cMorletTransformS(data=None, time=None, option=None):
    """
    Simplified Morlet transform of time series, single thread and no progress tracking

    params:
        data (2d numpy array): times series of size N * T
        time (1d numpy array): time axis
        option (dictionary): parameter settings

    return:
        M (3d numpy array): Morlet wavelet coefficients
        timeDs (1d numpy array): downsampled time points
    """
    if not option:
        option = {}
        option['freqs'] = np.arange(1, )

        return option

    freqs = option['freqs']
    fc = option['fc']
    fwhm = option['fwhm']
    dsRate = option['dsRate']
    mode = option['mode']

    # interval between time points
    Ts = time[1] - time[0]

    scales = freqs / float(fc)
    numSc = len(scales)
    sigma_tc = fwhm / math.sqrt(8 * math.log(2))
    sigma_t = np.divide(sigma_tc, scales)

    morletKernel = [[] for i in range(numSc)]
    numSigma = 4

    for m in range(numSc):
        x = np.arange(-numSigma * sigma_t[m], numSigma * sigma_t[m], Ts)
        morletKernel[m] = np.sqrt(scales[m]) * cMorletWavelet(scales[m] * x, fc, sigma_tc)

    numChns = data.shape[0]
    numT = data.shape[1]
    numDsT = len(np.arange(1, numT + 1, dsRate))
    M = np.zeros((numChns, numSc, numDsT))

    for n in range(numSc):
        tempKernel = morletKernel(n)

        for m in range(numChns):
            u = data[m, :]
            npad = len(tempKernel) - 1
            u_padded = np.pad(u, (npad//2, npad - npad//2), mode='constant')
            MTemp = np.convolve(u_padded, tempKernel, 'valid') * Ts
            M[m, n, :] = MTemp[np.arange(0, numT, dsRate)]

    if mode == "power":
        M = np.abs(M)**2
    elif mode == "envelope":
        M = np.sqrt(np.abs(M)**2)

    M = np.swapaxes(M, 1, 2)
    timeDs = time[0: len(time): 2]

    return M, timeDs
