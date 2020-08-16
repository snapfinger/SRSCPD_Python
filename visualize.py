import matplotlib.pyplot as plt


# TODO: add text annotations in graphs
# TODO: stem plot for channel
def plotTensorComponents(U, xAxis, option):
    ord = len(U)
    fig = plt.figure()
    for i, name in enumerate(U.keys()):
        print(i, name)
        im = fig.add_subplot(ord, 1, i + 1)
        plt.plot(U[name])

    plt.show()
