import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler

from dataset import WheatDS, splitted_loaders, SourceDS
from plot import plot_NDVI_T_info, plot_hist


def getNDVIseg(ndvi):
    seg = np.zeros(ndvi.shape)
    seg[(ndvi < -0.8)] = np.asarray([19, 69, 139])
    seg[(ndvi < 0) & (ndvi > -0.8)] = np.asarray([34, 139, 34])
    seg[(ndvi > 0) & (ndvi < 0.2)] = np.asarray([0, 255, 255])
    seg[(ndvi > 0.2) & (ndvi < 0.6)] = np.asarray([30, 105, 210])
    return seg


def Kmeans(img):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    color = np.asarray([[100, 100, 0], [100, 0, 0], [0, 0, 0], [0, 0, 100], [0, 100, 100]])
    res = color[label.flatten()]
    res2 = res.reshape(img.shape)
    return res2


def quantize(loader, bins, example_img):
    l = None
    r = None
    for img, _, name in loader:
        img = np.squeeze(img.numpy())

        ''' normalization [0; 1] ~ [-3std; 3std] '''
        mean = img.mean()
        std = img.std()
        img = (img - mean) / std
        img = (img + 3) / 6

        ''' quntization '''
        mean = img.mean()
        std = img.std()

        bins_ = np.full((len(bins), ), mean) + np.asarray(bins) * std
        q = np.digitize(img, bins_)

        sh = q * (255 // len(bins))
        if name[0].find('l_' + example_img) != -1:
            l = sh
        if name[0].find('r_' + example_img) != -1:
            r = sh

    cat = np.concatenate([l, r], axis=1)
    cv.imwrite("research_quant_" + str(len(bins) - 1) + ".png", cat)
    return


def correlation(loader):
    r = []
    g = []
    b = []
    for tir, rgb, _ in loader:
        r += np.corrcoef(tir.flatten(), rgb[:, :, 2].flatten())[0][1]
        g += np.corrcoef(tir.flatten(), rgb[:, :, 2].flatten())[0][1]
        b += np.corrcoef(tir.flatten(), rgb[:, :, 0].flatten())[0][1]
    print(f'Correlation TIR & :\b R = {np.mean(r)}, G = {np.mean(g)}, B = {np.mean(b)}')


if __name__ == '__main__':
    dataset = SourceDS('dataset.json')
    dataset.mode = 'NDVI'
    print(dataset)
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset))
    plot_NDVI_T_info(loader)
    plot_hist(loader, name_samples='IR_90_2020_12_11')

    dataset = SourceDS('dataset.json')
    print(dataset)
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset))
    correlation(loader)

    dataset = WheatDS('dataset_div2.json')
    print(dataset)
    loader, _ = splitted_loaders(dataset, train_size=0.8)
    quantize(loader, bins=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], example_img='box3_IR_90_2020_12_11')







