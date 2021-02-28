from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import cv2 as cv


def splitted_loaders(dataset, train_size, batch_size=1):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_size * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)
    return train_loader, test_loader



class SourceDS(Dataset):
    def __init__(self, json_file):
        self.paths = self._load_info(json_file)
        self.mode = None

    @staticmethod
    def _load_info(json_file):
        with open(json_file) as f:
            data = json.load(f)
        paths = [[ir, rgb] for ir, rgb in zip(data['IR'], data['RGB'])]
        return paths

    def __str__(self):
        return "Dataset properties: task - " + self.task + ", channels - " + self.channel + "\n"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = path[0].split('\\')[-1:][0].split('.')[0]

        tir = cv.imread(path[0])
        rgb = cv.imread(path[1])
        tir = cv.cvtColor(tir, cv.COLOR_BGR2GRAY)
        red = rgb[:, :, 2]
        tir = np.asarray(tir, dtype=np.float32)
        red = np.asarray(red, dtype=np.float32)
        ndvi = (tir - red) / (tir + red + 1e-7)

        if self.mode == 'NDVI':
            return ndvi, name
        else:
            return tir, rgb, name



class WheatDS(Dataset):
    def __init__(self, json_file):
        self.paths = SourceDS._load_info(json_file)
        self.channel = 'RGB'
        self.task = 'REG'

    def __str__(self):
        return self.task + ' ' + self.channel

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        tir = cv.imread(path[0])
        rgb = cv.imread(path[1])

        tir = cv.cvtColor(tir, cv.COLOR_BGR2GRAY)
        red = rgb[:, :, 2]
        green = rgb[:, :, 1]

        tir = np.asarray(tir, dtype=np.float32)
        red = np.asarray(red, dtype=np.float32)
        green = np.asarray(green, dtype=np.float32)

        ndvi = []
        if self.channel == 'TIR':
            ndvi = (tir - red) / (tir + red + 1e-7)
        else:
            red = (red - red.mean()) / red.std()
            green = (green - green.std()) / green.std()
            ndvi = (green - red) / (green + red + 1e-7)
            ndvi[(ndvi > 1) | (ndvi < -1)] = 0

        ndvi = np.expand_dims(ndvi, axis=0)

        name = path[0].split('\\')[-1:][0].split('.')[0]
        pos, day = name.split('_')[0], int(name.split('_')[-1:][0])

        if self.task == 'BIN':
            # TODO: Prove day >= 5
            label = True if pos == 'r' and day >= 5 else False
        else:
            label = day - 1 if pos == 'r' else 0

        return ndvi, label, name


