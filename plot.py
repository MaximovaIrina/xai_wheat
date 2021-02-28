import json

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_research(info, title, xlabel, ylabel, file, col=[0, 2, 4, 6, 10, 13, 16, 19]):
    df = pd.DataFrame.from_dict(info, orient='index', columns=col)

    fig, ax = plt.subplots()
    df.boxplot(ax=ax)

    plt.title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax2 = ax.twinx()
    ax2.plot(ax.get_xticks(), df.mean(), color='orange')
    ax2.set_ylim(ax.get_ylim())
    plt.legend(['mean'])
    plt.tight_layout()

    plt.savefig('plot\\' + file + '.png')
    return


def plot_NDVI_T_info(loader):
    drought_part = {'box1': [], 'box2': [], 'box3': []}
    drought_mean_ndvi = {'box1': [], 'box2': [], 'box3': []}
    delta_ndvi = {'box1': [], 'box2': [], 'box3': []}

    for ndvi, name in loader:
        box = name.split('_')[0]

        h, w = ndvi.shape
        ndvi_0 = ndvi[:, : w // 2]
        ndvi_1 = ndvi[:, w // 2:]

        drought_part[box].extend([100 * np.sum((ndvi_1 > 0) & (ndvi_1 < 0.2)) / (h * w)])
        drought_mean_ndvi[box].extend([np.mean(ndvi_1[(ndvi_1 > 0) & (ndvi_1 < 0.2)])])
        delta_ndvi[box].extend([np.mean(ndvi_1[(ndvi_1 > 0) & (ndvi_1 < 0.2)]) -
                                np.mean(ndvi_0[(ndvi_0 < 0) & (ndvi_0 > -0.8)])])

    plot_research(drought_part, "Spread of drought", "Days without watering", "Druoght, %", "research_part_nvdi")
    plot_research(drought_mean_ndvi, "NDVI_T", "Days without watering", "Average NDVI_T in the drought zone", "research_mean_nvdi")
    plot_research(delta_ndvi, "Δ NDVI_T", "Days without watering", "Δ NDVI_T", "research_delta_nvdi")


def plot_hist(loader, name_samples):
    data = []
    for ndvi, name in loader:
        if name.find(name_samples) != -1:
            data += [ndvi.flatten()]
    data = np.mean(data, axis=0)
    sns.histplot(data=data, kde=True, stat='probability')
    plt.savefig("plot\\hist_" + name_samples + ".png")


def plot_bar(x, y, xlabel, ylabel, file):
    plt.clf()
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'plot\\{file}.png')
    return

def plot_confusion_mat(labels, prediction, index, xlabel, ylabel, file):
    plt.clf()
    columns = list(np.expand_dims(index, axis=0))
    df_cm = pd.DataFrame(confusion_matrix(labels, prediction), index=index, columns=columns)
    sns.heatmap(df_cm, annot=True, cmap="crest")
    plt.xlabel("Prediction")
    plt.ylabel('True labels')
    plt.savefig(f'plot\\{file}.png')
    return


def plot_compare_NBINS_FEACH_NNEURON_rmse_single(data, file):
    plt.clf()
    mse = np.asarray(data['mse'])
    stat = mse[:, 0]
    hist = mse[:, 1]
    glcm = mse[:, 2]
    allf = mse[:, 3]
    x = list(range(1, len(stat) + 1))

    plt.clf()
    plt.plot(x, stat, label='STAT')
    plt.plot(x, hist, label='STAT + HIST')
    plt.plot(x, glcm, label='GLCM')
    plt.plot(x, allf, label='ALL')
    plt.legend(loc='upper right')
    plt.xlabel('Number neurons of the inner layer')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig(f'plot\\{file}.png')


def plot_compare_NBINS_FEACH_NNEURON_rmse_FE(file):
    plt.clf()
    mse = []
    feach = []
    for i in range(3, 9):
        with open('mse_' + str(i) + '.json', 'r') as json_file:
            data = json.load(json_file)
        mse += [data['mse']]
        feach += [data['feach']]

    sh = np.asarray(mse).shape
    mse = np.asarray(mse).transpose(1, 0, 2)
    feach = np.asarray(feach).transpose(1, 0, 2)
    mse = np.asarray(mse).reshape(sh[1], -1)
    feach = np.asarray(feach).reshape(sh[1], -1)

    z = np.argmin(mse, axis=1)
    mse = mse.min(axis=1)
    f = [feach[i, z[i]] for i in range(len(feach))]
    # f = [f_.replace(' ', '')for f_ in f]
    f = [f_.replace('STAT + HIST', '') for f_ in f]
    z = z // 4 + 3
    x = np.asarray(list(range(len(f)))) + 1
    plt.plot(x, mse, 'bo-')
    for (X, Y, Z, L) in zip(x, mse, f, z):
        plt.annotate(Z + '(' + str(L) + ')', xy=(X, Y),
                     xytext=(11, -11), ha='right', textcoords='offset points')

    plt.xlabel('Number neurons of the inner layer')
    plt.ylabel('Best RMSE')
    plt.tight_layout()
    plt.savefig(f'plot\\{file}.png')

