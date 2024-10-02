# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.utils.data as data
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import open3d
from pathlib import Path
from . import mask as util_
from .imdb import imdb


class OCIDObject(data.Dataset, imdb):
    def __init__(self, image_set, ocid_object_path = None):

        self._name = 'ocid_object_' + image_set
        self._image_set = image_set
        self._ocid_object_path = self._get_default_path() if ocid_object_path is None \
                            else ocid_object_path
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        self._width = 640
        self._height = 480
        self.image_paths = self.list_dataset()

        print('%d images for dataset %s' % (len(self.image_paths), self._name))
        self._size = len(self.image_paths)
        assert os.path.exists(self._ocid_object_path), \
                'ocid_object path does not exist: {}'.format(self._ocid_object_path)


    def list_dataset(self):
        data_path = Path(self._ocid_object_path)
        seqs = sorted(list(Path(data_path).glob('**/*seq*')))

        image_paths = []
        for seq in seqs:
            paths = sorted(list((seq / 'rgb').glob('*.png')))
            image_paths += paths
        return image_paths


    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels


    def __getitem__(self, idx):

        # BGR image
        filename = str(self.image_paths[idx])
        im = cv2.imread(filename)
        im_tensor = torch.from_numpy(im)
        image_blob = im_tensor.permute(2, 0, 1)

        # Label
        labels_filename = filename.replace('rgb', 'label')
        foreground_labels = util_.imread_indexed(labels_filename)
        # mask table as background
        foreground_labels[foreground_labels == 1] = 0
        if 'table' in labels_filename:
            foreground_labels[foreground_labels == 2] = 0
        foreground_labels = self.process_label(foreground_labels)
        label_blob = torch.from_numpy(foreground_labels).unsqueeze(0)

        index = filename.find('OCID')
        sample = {'image_color': image_blob,
                  'label': label_blob,
                  'filename': filename[index+5:]}

        # Depth image
        pcd_filename = filename.replace('rgb', 'pcd')
        pcd_filename = pcd_filename.replace('png', 'pcd')
        pcd = open3d.io.read_point_cloud(pcd_filename, remove_nan_points=False, remove_infinite_points=False, print_progress=False)
        pcloud = np.asarray(pcd.points).astype(np.float32)
        pcloud[np.isnan(pcloud)] = 0
        xyz_img = pcloud.reshape((self._height, self._width, 3))
        depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
        sample['depth'] = depth_blob

        return sample


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where ocid_object is expected to be installed.
        """
        ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
        return os.path.join(ROOT_DIR, 'data', 'OCID')