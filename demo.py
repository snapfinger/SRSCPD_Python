import matplotlib.pyplot as plt
import numpy as np
import h5py

data_path = "/hd2/research/brain/time_series/my_SRSCPD_ALS/data.mat"
# data = scipy.io.loadmat(data_path)

# print(data.shape)

with h5py.File(data_path, 'r') as file:
    print(file['data'].shape)
    print(file['data'])
    X_ref = file['data'][0][0]
    Xn_ref = file['data'][1][0]

    X = np.array(file[X_ref]).T
    Xn = np.array(file[Xn_ref]).T
    print(Xn[19, 1])
plt.imshow(Xn, cmap="gray")
plt.axis('off')
plt.show()
