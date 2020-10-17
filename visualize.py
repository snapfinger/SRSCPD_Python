import matplotlib.pyplot as plt


# TODO: stem plot for channel
def plotTensorComponents(U, xAxis):
    fig = plt.figure()
    x_labels = ['Channel (channel ID)',
                'Time (second)',
                'Spectrum (Hz)']

    if isinstance(U, dict):
        for i, name in enumerate(U.keys()):
            fig.add_subplot(len(U), 1, i + 1)
            plt.plot(xAxis[i], U[name])
            plt.xlabel(x_labels[i])
            plt.grid(True)
    else:
        for i, cur_arr in enumerate(U):
            fig.add_subplot(len(U), 1, i + 1)
            plt.plot(cur_arr)
            plt.xlabel(x_labels[i])
            plt.grid(True)

    plt.tight_layout()
    plt.show()