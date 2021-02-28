import json
import re
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
import torch
import os

from tqdm import tqdm

from plot import plot_confusion_mat, plot_bar, plot_compare_NBINS_FEACH_NNEURON_rmse_FE, \
    plot_compare_NBINS_FEACH_NNEURON_rmse_single


def features_labels(file):
    dataset = torch.load(file)
    features = dataset['features']
    labels = dataset['labels']
    names = dataset['names']
    return features, labels, names


def scale_data(data, train_mean=None, train_std=None):
    if train_mean is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-10), mean, std
    else:
        return (data - train_mean) / (train_std + 1e-10)


def build_classifiers(n, mode):
    clf = None
    if mode == 'r':
        clf = MLPRegressor(hidden_layer_sizes=(n,), max_iter=int(1e5), random_state=1)
    if mode == 'c':
        clf = MLPClassifier(hidden_layer_sizes=(n,), max_iter=10000, random_state=1)
    return clf


def train_and_test(ind, clf, train_data, test_data, mode):
    train_features, train_labels, tr_names = train_data
    test_features, test_labels, ts_names = test_data

    train_features = train_features[:, ind]
    test_features = test_features[:, ind]

    train_features, train_mean, train_std = scale_data(train_features)
    test_features = scale_data(test_features, train_mean, train_std)

    clf.fit(train_features, train_labels)
    print(f'OUT ACTIVATION FUNCTION {clf.out_activation_}')


    # plt.clf()
    # x = ['mean', 'std', 'max', 'min', 'hist\n[0]', 'hist\n[1]', 'hist\n[2]', 'hist\n[3]', 'con', 'hom', 'eng', 'corr', 'enp']
    # w = np.mean(abs(clf.coefs_[0]), axis=1)
    # glcm_w = w[8:]
    # glcm_w = [np.mean(glcm_w[i:i+4]) for i in range(0, len(glcm_w), 4)]
    # y = list(w[:8]) + list(glcm_w)
    # plot_bar(x, y, 'Features', 'Average weight', 'weight_feach.png')


    prediction = clf.predict(test_features)

    # if mode == 'r':
    #     s_test_labels = np.sort(test_labels)
    #     s_ind = np.argsort(test_labels)
    #     s_pred = [prediction[i] for i in s_ind]
    #     y = s_pred - s_test_labels
    #     x = ['Img' + str(i) + '\n(' + str(label.item()) + ')' for label, i in zip(s_test_labels, list(range(len(s_test_labels))))]
    #     plot_bar(x, y, 'Test samples', 'Day', 'RMSE')

    acc = 0
    if mode == 'c':
        _, _, acc, _ = precision_recall_fscore_support(test_labels, prediction, zero_division=0, average='weighted')
    if mode == 'r':
        acc = sklearn.metrics.mean_squared_error(np.asarray(test_labels), prediction, squared=False)

    # plot_confusion_mat(test_labels, prediction, ['Drought', 'No drought'], 'Prediction', 'True labels', 'conf_mat')
    return acc


def classification_results(clf, train_file, test_file, n_bins, mode):
    properties = {'STAT': list(range(4)), 'STAT + HIST': list(range(4 + n_bins)),
                  'GLCM': list(range(4 + n_bins, 4 + n_bins + 20)), 'All': list(range(4 + n_bins + 20))}
    data = {'name': [], 'properties': [], 'avg f-score': []}

    train_data = features_labels(train_file)
    test_data = features_labels(test_file)

    for prop, ind in properties.items():
        acc = train_and_test(ind=ind, clf=clf, train_data=train_data, test_data=test_data, mode=mode)
        data['properties'] += [prop]
        data['avg f-score'] += [acc]

    if mode == 'r':
        print(f'Best score {np.min(data["avg f-score"])}')
        print(f'Best feach {data["properties"][np.argmin(data["avg f-score"])]}')
    else:
        print(f'Best score {np.max(data["avg f-score"])}')
        print(f'Best feach {data["properties"][np.argmax(data["avg f-score"])]}')

    # plot_bar(properties.keys(), data["avg f-score"], 'Features', 'F-score', 'fsore.png')
    return data['avg f-score'], properties



if __name__ == '__main__':
    feach = 'greenG_reg8'
    train_file = os.path.join(feach + '_train.pth')
    test_file = os.path.join(feach + '_test.pth')

    TASK = 'r'
    n_bins = int(re.findall('(\d+)', feach)[0])

    mse = []
    feach = []
    for n in tqdm(range(1, 35)):
        classifiers = build_classifiers(n, 'r')
        m, f = classification_results(classifiers, train_file, test_file, n_bins+2, 'r')
        mse += [m]
        feach += [f]
    data = {'mse': mse, 'feach': feach}
    with open(f'mse_{n_bins}.json', 'w') as json_file:
        json.dump(data, json_file)

    plot_compare_NBINS_FEACH_NNEURON_rmse_FE('compare_all')

    with open('mse_4.json', 'r') as json_file:
        data = json.load(json_file)
    plot_compare_NBINS_FEACH_NNEURON_rmse_single(data, 'mse_single')

