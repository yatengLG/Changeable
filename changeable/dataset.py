# -*- coding: utf-8 -*-
# @Author  : LG

import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from numpy import ndarray

class VOCDataset(Dataset):
    def __init__(self, root:str, classes_name:tuple, is_train:bool=True, transforms=None, keep_difficult:bool=True):
        """
        voc格式数据集
        :param root:        数据集根目录
        :param classes_name:    类别集合
        :param is_train:        是否是训练集
        :param transform:       数据增强
        :param keep_difficult:  保留难识别目标
        """
        self.classes_name = classes_name
        self.num_classes = len(self.classes_name)
        self.is_train = is_train
        self.root = root
        self.transforms = transforms
        self.keep_difficult = keep_difficult

        sets_file = 'train.txt' if is_train else 'val.txt'
        with open(os.path.join(self.root, "ImageSets", "Main", sets_file)) as f:
            self.ids = [line.rstrip('\n') for line in f.readlines()]

        self.class_dict = {class_name: i+1 for i, class_name in enumerate(self.classes_name)}
        self.class_dict['__background__'] = 0

    def __getitem__(self, index: int):
        image_name = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_name)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_name)
        if self.transforms is not None:
            image, boxes, labels = self.transforms(image, boxes, labels)
        return image, boxes, labels, image_name

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, img_name: str) -> (ndarray, ndarray, ndarray):
        annotation_file = os.path.join(self.root, "Annotations", "{}.xml".format(img_name))
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            # .encode('utf-8').decode('UTF-8-sig') 解决Windows下中文编码问题
            class_name = obj.find('name').text.encode('utf-8').decode('UTF-8-sig')
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text.encode('utf-8').decode('UTF-8-sig')) - 1
            y1 = float(bbox.find('ymin').text.encode('utf-8').decode('UTF-8-sig')) - 1
            x2 = float(bbox.find('xmax').text.encode('utf-8').decode('UTF-8-sig')) - 1
            y2 = float(bbox.find('ymax').text.encode('utf-8').decode('UTF-8-sig')) - 1
            difficult = int(obj.find('difficult').text)
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult.append(difficult)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, img_name:str) -> ndarray:
        image_file = os.path.join(self.root, "JPEGImages", "{}.jpg".format(img_name))
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def get_one_image(self,image_name:str = None):
        import random
        if not image_name:
            image_name = random.choice(self.ids)
        boxes, labels, is_difficult = self._get_annotation(image_name)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_name)
        if self.transforms is not None:
            image, boxes, labels = self.transforms(image, boxes, labels)
        return image, boxes, labels, image_name