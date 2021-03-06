import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt


def show_sample(sample):
    """Shows the sample with tasks and answers"""
    print("Train:")
    for i in range(len(sample["train"])):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.matshow(np.array(sample["train"][i]["input"]), cmap="Set3", norm=mpl.colors.Normalize(vmin=0, vmax=9))

        ax2 = fig.add_subplot(122)
        ax2.matshow(np.array(sample["train"][i]["output"]), cmap="Set3", norm=mpl.colors.Normalize(vmin=0, vmax=9))

        plt.show()

    print("Test:")
    for i in range(len(sample["test"])):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.matshow(np.array(sample["test"][i]["input"]), cmap="Set3", norm=mpl.colors.Normalize(vmin=0, vmax=9))

        if "output" in sample["test"][i]:
            ax2 = fig.add_subplot(122)
            ax2.matshow(np.array(sample["test"][i]["output"]), cmap="Set3", norm=mpl.colors.Normalize(vmin=0, vmax=9))

        plt.show()


def matrix2answer(array):
    s = "|"
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            s = s + str(int(array[i, j]))
        s = s + "|"
    return str(s)
