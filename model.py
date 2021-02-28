import cv2
from skimage.feature import greycomatrix, greycoprops
from torch.nn import Module, Parameter
from joblib import delayed, Parallel
import numpy as np
import torch


class Features(Module):
    """ ABSTRACT CLASS """

    STAT_AXIS = (-2, -1)

    def __init__(self, n_jobs, mode, bins):
        super().__init__()
        ''' ALGORITHM PARAMS '''
        stride = [4, 4] if mode == 'local' else [1, 1]
        mask_size = [17, 17] if mode == 'local' else [220, 155]
        self.stride = Parameter(torch.tensor(stride), requires_grad=False)
        self.mask_size = Parameter(torch.tensor(mask_size), requires_grad=False)
        self.n_jobs = Parameter(torch.tensor(n_jobs), requires_grad=False)
        self.bins = Parameter(torch.tensor(bins), requires_grad=False)


    def all_sliding_windows(self, imgs):
        """RETURNS ARRAY OF SHAPE (B, C, Nx, Ny, MaskX, MaskY)"""
        shape = imgs.shape[:-2] + ((imgs.shape[-2] - self.mask_size[-2]) // self.stride[-2] + 1,) + \
                ((imgs.shape[-1] - self.mask_size[-1]) // self.stride[-1] + 1,) + tuple(self.mask_size)
        strides = imgs.strides[:-2] + (imgs.strides[-2] * self.stride[-2],) + (imgs.strides[-1] * self.stride[-1],) + \
                  imgs.strides[-2:]
        return np.lib.stride_tricks.as_strided(imgs, shape=shape, strides=strides)

    def quantize(self, img):
        mean = img.mean()
        std = img.std()
        ''' image normalization [0; 1] ~ [-3std; 3std] '''
        img = (img - mean) / std
        img = (img + 3) / 6
        mean = img.mean()
        std = img.std()
        bins_ = np.full((len(self.bins), ), mean) + np.asarray(self.bins) * std
        q = np.digitize(img, bins_)
        return q

    def statistics(self, imgs):
        mean = imgs.mean(axis=self.STAT_AXIS)
        std = imgs.std(axis=self.STAT_AXIS)
        max = (imgs.max(axis=self.STAT_AXIS) - mean) / std
        min = (mean - imgs.min(axis=self.STAT_AXIS)) / std
        return np.concatenate([mean, std, max, min], axis=1)

    def hist(self, q_img):
        if np.max(q_img) > 0:
            levels = len(set(np.flatten(q_img)))
            hist, _ = np.histogram(q_img.reshape(-1), bins=np.arange(levels + 1), density=True)
        else:
            hist = np.zeros(self.levels)
        return hist

    def parallel_image_processing(self, q_imgs, function):
        sh = np.shape(q_imgs)
        gc = q_imgs.reshape((-1,) + sh[-2:])
        gcs = Parallel(n_jobs=self.n_jobs)(delayed(function)(img) for img in gc)
        gcs = np.stack(gcs, axis=0)
        if function != self.quantize:
            gcs = gcs.reshape(sh[:-2] + (-1,))
            gcs = gcs.transpose(0, -1, 2, 3, 1)
            gcs = gcs.squeeze(-1)
        else:
            gcs = gcs.reshape(sh)
        return gcs

    def forward(self, inputs):
        raise NotImplementedError("Subclass must implement abstract method")


class ChannelFeatures(Features, Module):
    GLCM_PROPERTIES = ('contrast', 'homogeneity', 'energy', 'correlation')

    def __init__(self, n_jobs, mode, bins, dist, theta):
        Module.__init__(self)
        Features.__init__(self, n_jobs, mode, bins)
        self.dist = Parameter(torch.tensor(dist), requires_grad=False)
        self.theta = Parameter(torch.tensor(theta), requires_grad=False)

    def glcm(self, q_img):
        levels = len(set(np.flatten(q_img)))
        g = greycomatrix(q_img, self.dist, self.theta, levels, normed=True, symmetric=True)
        props = np.array([greycoprops(g, p) for p in self.GLCM_PROPERTIES]).reshape(-1)
        entropy = -np.sum(np.multiply(g, np.log2(g + 1e-8)), axis=(0, 1)).reshape(-1)
        props = np.concatenate([props, entropy], axis=0)
        return props

    # x: B, C, H, W
    def forward(self, x):
        # x: B, 1, H, W
        x = x.numpy()
        # x = np.expand_dims(x, 1)

        # x: B, 1, Nx, Ny, MaskX, MaskY
        x = self.all_sliding_windows(x)

        # stat: B, [MEAN, STD, MAX, MIN], Nx, Ny
        stat = self.statistics(x)

        # quant: B, 1, Nx, Ny, MaskX, MaskY
        quant = self.parallel_image_processing(x, self.quantize)

        # hist: B, [HIST], Nx, Ny
        hist = self.parallel_image_processing(quant, self.hist)

        # glcm: B, [GLCM], Nx, Ny
        glcm = self.parallel_image_processing(quant, self.glcm)

        # features: B, [STAT, HIST, GLCM], Nx, Ny
        features = np.concatenate([stat, hist, glcm], axis=1)

        # features: B, [STAT, HIST, GLCM]
        features = np.mean(features, axis=(-2, -1), keepdims=False)  # np average pooling
        return features

