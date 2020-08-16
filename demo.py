"""
This demo shows the simulation example reported in
[1] J. Li, et al. "Scalable and robust tensor decomposition for brain network identification in spontaneous stereotactic EEG data", IEEE Trans. Biomed. Eng., 2018.
[2] J. Li, et al. "Robust tensor decomposition of resting brain networks in stereotactic EEG", IEEE 51st Asilomar Conference on Signal, System and Computers, Pacfic Grove, CA, 2017.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py

R = 5;

data_path = "/hd2/research/brain/time_series/my_SRSCPD_ALS/data.mat"
with h5py.File(data_path, 'r') as file:
    X_ref = file['data'][0, 0]
    Xn_ref = file['data'][1, 0]
    UGT_ref = file['data'][2, 0]
    xAxis_ref = file['data'][3, 0]

    X = np.array(file[X_ref]).T
    Xn = np.array(file[Xn_ref]).T

    xAxis = file[xAxis_ref]
    t = np.array(file[xAxis[1, 0]]).T
    freq = np.array(file[xAxis[2, 0]]).T

print(t.shape)
print(freq.shape)

# plt.imshow(Xn, cmap="gray")
# plt.axis('off')
# plt.show()
