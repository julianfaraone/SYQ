#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bsds500.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os, glob
import cv2
import numpy as np

from ...utils import logger, get_rng, get_dataset_path
from ...utils.fs import download
from ..base import RNGDataFlow

try:
    from scipy.io import loadmat
    __all__ = ['BSDS500']
except ImportError:
    logger.warn_dependency('BSDS500', 'scipy.io')
    __all__ = []

DATA_URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
IMG_W, IMG_H = 481, 321

class BSDS500(RNGDataFlow):
    """
    `Berkeley Segmentation Data Set and Benchmarks 500
    <http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500>`_.

    Produce (image, label) pair, where image has shape (321, 481, 3) and
    ranges in [0,255]. Label is binary and has shape (321, 481).
    Those pixels annotated as boundaries by <=2 annotators are set to 0.
    This is used in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    """

    def __init__(self, name, data_dir=None, shuffle=True):
        """
        :param name: 'train', 'test', 'val'
        :param data_dir: a directory containing the original 'BSR' directory.
        """
        # check and download data
        if data_dir is None:
            data_dir = get_dataset_path('bsds500_data')
        if not os.path.isdir(os.path.join(data_dir, 'BSR')):
            download(DATA_URL, data_dir)
            filename = DATA_URL.split('/')[-1]
            filepath = os.path.join(data_dir, filename)
            import tarfile
            tarfile.open(filepath, 'r:gz').extractall(data_dir)
        self.data_root = os.path.join(data_dir, 'BSR', 'BSDS500', 'data')
        assert os.path.isdir(self.data_root)

        self.shuffle = shuffle
        assert name in ['train', 'test', 'val']
        self._load(name)

    def _load(self, name):
        image_glob = os.path.join(self.data_root, 'images', name, '*.jpg')
        image_files = glob.glob(image_glob)
        gt_dir = os.path.join(self.data_root, 'groundTruth', name)
        self.data = np.zeros((len(image_files), IMG_H, IMG_W, 3), dtype='uint8')
        self.label = np.zeros((len(image_files), IMG_H, IMG_W), dtype='float32')

        for idx, f in enumerate(image_files):
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            assert im is not None
            if im.shape[0] > im.shape[1]:
                im = np.transpose(im, (1,0,2))
            assert im.shape[:2] == (IMG_H, IMG_W), "{} != {}".format(im.shape[:2], (IMG_H, IMG_W))

            imgid = os.path.basename(f).split('.')[0]
            gt_file = os.path.join(gt_dir, imgid)
            gt = loadmat(gt_file)['groundTruth'][0]
            n_annot = gt.shape[0]
            gt = sum(gt[k]['Boundaries'][0][0] for k in range(n_annot))
            gt = gt.astype('float32')
            gt *= 1.0 / n_annot
            if gt.shape[0] > gt.shape[1]:
                gt = gt.transpose()
            assert gt.shape == (IMG_H, IMG_W)

            self.data[idx] = im
            self.label[idx] = gt

    def size(self):
        return self.data.shape[0]

    def get_data(self):
        idxs = np.arange(self.data.shape[0])
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.data[k], self.label[k]]


if __name__ == '__main__':
    a = BSDS500('val')
    for k in a.get_data():
        cv2.imshow("haha", k[1].astype('uint8')*255)
        cv2.waitKey(1000)
