import matplotlib.pyplot as plt
import numpy as np


# # TODO: add text annotations in graphs
# # TODO: stem plot for channel
def plotTensorComponents(U):
    ord = len(U)
    fig = plt.figure()

    if isinstance(U, dict):
        for i, name in enumerate(U.keys()):
            print(i, name)
            im = fig.add_subplot(4, 1, i + 1)
            plt.plot(U[name])
    else:
        for i, cur_arr in enumerate(U):
            im = fig.add_subplot(ord, 1, i + 1)
            plt.plot(cur_arr)

    plt.show()


# TODO: fix show length incompatibility
# def plotTensorComponents2(U, xAxis):
#     ord = len(U)
#     fig = plt.figure()
#
#     if isinstance(U, dict):
#         for i, name in enumerate(U.keys()):
#             print(i, name)
#             im = fig.add_subplot(4, 1, i + 1)
#             plt.plot(U[name])
#     else:
#         fig = plt.figure()
#         # xlabels = ['Channel (channel ID)', 'Time (second)', 'Spectrum (Hz)']
#
#         ax1 = fig.add_subplot(2, 2, 1)
#         ax1.plot(xAxis['t'], U[0])
#         ax1.set_xlabel('Channel (channel ID)')
#         ax1.set_xlim(xAxis['freq'], np.max(xAxis['freq']))
#         add_on1 = (np.max(U[0]) - np.min(U[0])) * 0.1
#         ax1.set_ylim(np.min(U[0]) - add_on1, np.max(U[0]) + add_on1)
#
#
#         ax2 = fig.add_subplot(2, 2, 2)
#         ax2.plot(xAxis['t'], U[1])
#         ax2.set_xlabel('Time (second)')
#         ax2.set_xlim(xAxis['t'], np.max(xAxis['t']))
#         add_on2 = (np.max(U[1]) - np.min(U[1])) * 0.1
#         ax2.set_ylim(np.min(U[1]) - add_on2, np.max(U[1]) + add_on2)
#
#         ax3 = fig.add_subplot(2, 2, 3)
#         ax3.plot(xAxis['freq'], U[2])
#         ax3.set_xlabel('Spectrum (Hz)')
#         ax3.set_xlim(xAxis['freq'], np.max(xAxis['freq']))
#         add_on3 = (np.max(U[2]) - np.min(U[2])) * 0.1
#         ax3.set_ylim(np.min(U[2]) - add_on3, np.max(U[2]) + add_on3)
#
#     plt.show()


