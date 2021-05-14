import torch
import os
import pandas as pd
import pathlib
from torch.utils.data import Dataset
import numpy as np
import cv2
import copy

class JCNN:
    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False) -> None:
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        self.ids = [info['image_id'] for info in self.data]

    def __len__(self):
        return len(self.data)

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def _read_image(self, image_id):
        image = cv2.imread(f'{self.root}/{image_id}')
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _read_data(self):
        path = self.root
        gt = pd.read_csv(f'{path}/gt.txt', sep=';', names=['Filename', 'Roi.X1','Roi.Y1','Roi.X2','Roi.Y2', 'ClassId'])
        ds = self.root / f'{self.dataset_type}.txt'
        with open(ds) as d:
            ds_images = d.read().split('\n')
        names = pd.read_csv(f'{self.root}/signnames.csv')
        class_names = ['BACKGROUND'] + sorted(list(names['SignName'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        selected_objs = gt.loc[gt['Filename'].isin(ds_images)]
        selected_objs.apply(lambda x: data.append({'image_id': x['Filename'], 'boxes': np.array([x[['Roi.X1','Roi.Y1','Roi.X2','Roi.Y2']].values.astype(np.float32)]), 'labels': np.array([x['ClassId']])}), axis=1)
        return data, class_names, class_dict