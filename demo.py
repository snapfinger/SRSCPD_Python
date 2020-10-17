"""
This demo shows the simulation example reported in
[1] J. Li, et al. "Scalable and robust tensor decomposition for brain network identification in spontaneous stereotactic EEG data", IEEE Trans. Biomed. Eng., 2018.
[2] J. Li, et al. "Robust tensor decomposition of resting brain networks in stereotactic EEG", IEEE 51st Asilomar Conference on Signal, System and Computers, Pacfic Grove, CA, 2017.
"""

import numpy as np
import h5py

from visualize import plotTensorComponents
from morlet import cMorletTransformS
from srscpd import srscpd


R = 5

# --------------- --------- data loading ---------------------------
data_path = "data.mat"
with h5py.File(data_path, 'r') as file:
    X_ref = file['data'][0, 0]
    Xn_ref = file['data'][1, 0]
    UGT_ref = file['data'][2, 0]
    xAxis_ref = file['data'][3, 0]

    X = np.array(file[X_ref]).T
    Xn = np.array(file[Xn_ref]).T

    UGT = file[UGT_ref]
    UGT_dict = {}
    UGT_dict['Channel'] = np.array(file[UGT[0, 0]]).T
    UGT_dict['Time'] = np.array(file[UGT[1, 0]]).T
    UGT_dict['Spectrum'] = np.array(file[UGT[2, 0]]).T

    xAxis = file[xAxis_ref]
    xAxis_list = [np.squeeze(file[xAxis[i, 0]]) for i in range(3)]

    t = xAxis_list[1]
    freq = xAxis_list[2]


#%% --------------- --------- data plotting ------------------------
plotTensorComponents(UGT_dict, xAxis_list)

# --------------- calculate the Morlet wavelet --------------------
option = cMorletTransformS()
option['fc'] = 1.0
option['fwhm'] = 2.0
option['mode'] = 'power'
option['freqs'] = freq

TF, tReturn = cMorletTransformS(Xn, t, option)


# --------------- CP decomposition using SRSCPD --------------------
option = srscpd()
option['isStats'] = False
option['optAlg']['nonnegative'] = [True, True, True]
result = srscpd(TF, R, option)


#%% -------------------- plot the results ----------------------------
U = result[R - 1]['U']
plotTensorComponents(U, xAxis_list)