from tqdm import tqdm
import numpy as np
import random
import torch

from dataset import splitted_loaders, WheatDS
from model import ChannelFeatures


def get_and_save_features(data_loader, model, loader_name):
    data_to_save = {'features': [], 'labels': [], 'names': []}
    for images, labels, names in tqdm(data_loader, desc=f'Getting {loader_name} features'):
        features = model(images)
        data_to_save['features'] += [features]
        data_to_save['labels'] += labels
        data_to_save['names'] += names
    data_to_save['features'] = np.concatenate(data_to_save['features'], axis=0)
    torch.save(data_to_save, loader_name + '.pth')
    return


if __name__ == '__main__':
    '''TESTING SETTINGS'''
    np.random.seed(42)
    random.seed(1001)
    torch.manual_seed(1002)

    '''DATASET'''
    dataset = WheatDS('dataset_div2.json')
    print(f"Daratset: {dataset}")

    '''LOADERS OF DATA'''
    train_loader, test_loader = splitted_loaders(dataset, train_size=0.8)

    '''CHANNEL MODEL'''
    bins = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    model = ChannelFeatures(n_jobs=8,
                            mode='global',
                            bins=bins,
                            dist=[1],
                            theta=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    '''GETTING AND SAVING DATA'''
    get_and_save_features(train_loader, model, str(dataset) + "_" + str(len(bins) - 1) + '_train')
    get_and_save_features(test_loader, model, str(dataset) + "_" + str(len(bins) - 1) + '_test')

