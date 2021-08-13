# -*- coding: utf-8 -*-
# @Author  : LG

from numpy import ndarray
from typing import Tuple
from PIL import Image, ImageOps
import numpy as np
import random
from .utils.boxes import boxes_iou

def adaptive_resize(image:ndarray, size: Tuple[int, int]=None, boxes:ndarray=None, value=(114, 114, 114)):
    """
    自适应resize，保持保持图片原始比例,扩充短边
    :param image:
    :param size:
    :param boxes:
    :param value:
    :return:
    """
    image = image.copy()
    h, w, _ = image.shape
    if size is not None:
        ratio = min(size[0]/w, size[1]/h)
        rh, rw = int(ratio*h), int(ratio*w)
        image = Image.fromarray(np.uint8(image)).resize((rw, rh))
        crop_l, crop_t = round((size[0]-rw)/2), round((size[1]-rh)/2)
        crop_r, crop_d = size[0]-crop_l-rw, size[1]-crop_t-rh
        image = ImageOps.expand(image, (crop_l, crop_t, crop_r, crop_d), value)
        image = np.array(image, dtype=np.float32)

        if boxes is not None:
            boxes = boxes.copy()
            boxes *= np.array((ratio, ratio, ratio, ratio))
            boxes += np.array((crop_l, crop_t, crop_l, crop_t))

    if boxes is not None:
        return image, boxes
    else:
        return image

def crop_iou(image: ndarray, boxes: ndarray, labels: ndarray, min_iou: float=None) -> Tuple[ndarray, ndarray, ndarray]:
    """
    iou裁剪。指定iou阈值，在原图上进行随机裁剪。
    :param image:
    :param boxes:
    :param labels:
    :param min_iou:
    :return:
    """
    assert boxes.shape[0] == labels.shape[0] > 0
    image, boxes, labels = image.copy(), boxes.copy(), labels.copy()
    if min_iou is None:
        return image, boxes, labels
    h, w, _ = image.shape
    rect = np.array([0, 0, w, h])
    mask = labels > -1
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    for _ in range(50):
        rw = random.uniform(0.3 * w, w)
        rh = random.uniform(0.3 * h, h)
        if not (0.5 < rw/rh < 2):
            continue
        lm = max(0, np.min(centers[:, 0]) - rw)
        tm = max(0, np.min(centers[:, 1]) - rh)
        left = random.uniform(lm, min(w-rw, np.max(centers[:, 0])))
        top = random.uniform(tm, min(h-rh, np.max(centers[:, 1])))
        rect_ = np.array([int(left), int(top), int(left+rw), int(top+rh)])
        overlap = boxes_iou(boxes, rect_)
        if overlap.max() < min_iou:
            continue
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        m1 = (centers[:, 0] > rect_[0]) * (centers[:, 0] < rect_[2])
        m2 = (centers[:, 1] > rect_[1]) * (centers[:, 1] < rect_[3])
        mask_ = m1 * m2
        mask = mask_
        rect = rect_
        if np.any(mask_):
            break
    image = image[rect[1]:rect[3], rect[0]:rect[2], :]
    boxes = boxes[mask, :].copy()
    labels = labels[mask]
    boxes[:, :2] = np.maximum(boxes[:, :2], rect[:2])
    boxes[:, :2] -= rect[:2]
    boxes[:, 2:] = np.minimum(boxes[:, 2:], rect[2:])
    boxes[:, 2:] -= rect[:2]
    return image, boxes, labels

def crop_size(image: ndarray, boxes: ndarray, labels: ndarray, size: Tuple[int, int]=None, mode: str='resize'):
    """
    尺寸裁剪。指定尺寸，在原图上进行随机裁剪，若原图尺寸小于裁剪尺寸，则对原图进行扩充或resize
    :param image:
    :param boxes:
    :param labels:
    :param size:    裁剪尺寸
    :param mode:    图片尺寸小于裁剪尺寸所选用的方式, 'expand' or 'resize', 扩充灰边或resize
    :return:
    """
    assert boxes.shape[0] == labels.shape[0] > 0
    image, boxes, labels = image.copy(), boxes.copy(), labels.copy()

    if size is None:
        return image, boxes, labels
    h, w, _ = image.shape

    if size[0] > w or size[1] > h:
        if mode == 'expand':    # 不改变比例和大小，在四周填充灰边
            crop_l, crop_t = max(0, round((size[0] - w) / 2)), max(0, round((size[1] - h) / 2))
            crop_r, crop_d = max(0, size[0] - crop_l - w), max(0, size[1] - crop_t - h)
            image = Image.fromarray(np.uint8(image))
            image = ImageOps.expand(image, (crop_l, crop_t, crop_r, crop_d), (114, 114, 114))
            image = np.array(image, dtype=np.float32)
            h, w, _ = image.shape
            boxes += (crop_l, crop_t, crop_r, crop_d)
        elif mode == 'resize':  # 保持宽高比，resize
            ratio = max(size[0] / w, size[1] / h)
            rw, rh = int(ratio * w)+1, int(ratio * h)+1
            image = Image.fromarray(np.uint8(image))
            image = np.array(image.resize(size=(rw, rh)), dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / w * rw
            boxes[:, 1] = boxes[:, 1] / h * rh
            boxes[:, 2] = boxes[:, 2] / w * rw
            boxes[:, 3] = boxes[:, 3] / h * rh
    h, w, _ = image.shape

    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    lm = max(0, np.min(centers[:, 0])-size[0])
    tm = max(0, np.min(centers[:, 1])-size[1])
    for _ in range(50):
        left = random.uniform(lm, min(np.max(centers[:, 0]), w-size[0]))
        top = random.uniform(tm, min(np.max(centers[:, 1]), h-size[1]))
        rect = np.array([int(left), int(top), int(left+size[0]), int(top+size[1])])
        m1 = (centers[:, 0] > rect[0]) * (centers[:, 0] < rect[2])
        m2 = (centers[:, 1] > rect[1]) * (centers[:, 1] < rect[3])
        mask = m1 * m2
        if np.any(mask):
            image = image[rect[1]:rect[3], rect[0]:rect[2], :]
            boxes = boxes[mask, :].copy()
            labels = labels[mask]
            boxes[:, :2] = np.maximum(boxes[:, :2], rect[:2])
            boxes[:, :2] -= rect[:2]
            boxes[:, 2:] = np.minimum(boxes[:, 2:], rect[2:])
            boxes[:, 2:] -= rect[:2]
            return image, boxes, labels
    # 如果50次尝试后，没有合适的截取框，则返回原图
    image, boxes = adaptive_resize(image, size, boxes)
    return image, boxes, labels

def generate_mosaic(images4: Tuple[ndarray, ...], boxes4: Tuple[ndarray, ...], labels4: Tuple[ndarray, ...],
                    mosaic_size: Tuple[int, int], center_range: Tuple[float, float]=(0.3, 0.7)):
    """
    将四张图片合成一张图片图片
    :param images4:         图片，四张
    :param boxes4:          标注框
    :param labels4:         标签
    :param mosaic_size:     合成的mosaic图片尺寸
    :param center_range:    合成的mosaic四张图片的中心浮动范围
    :return:
    """
    assert len(images4) == len(boxes4) == len(labels4) == 4
    w, h = mosaic_size
    cx, cy = int(random.uniform(center_range[0]*w, center_range[1]*w)), \
             int(random.uniform(center_range[0]*h, center_range[1]*h))
    sizes = ((cx, cy), (w - cx, cy), (cx, h - cy), (w - cx, h - cy))
    shifts = ((0, 0, 0, 0), (cx, 0, cx, 0), (0, cy, 0, cy), (cx, cy, cx, cy))
    images, boxes, labels = [], [], []
    for img, box, lab, size, shift in zip(images4, boxes4, labels4, sizes, shifts):
        img, box, lab = crop_size(img, box, lab, size)
        images.append(img)
        boxes.append(box + shift)
        labels.append(lab)
    image = np.concatenate((np.concatenate((images[0], images[1]), axis=1),
                            np.concatenate((images[2], images[3]), axis=1)),
                           axis=0)
    boxes = np.concatenate(boxes, axis=0)
    labels = np.concatenate(labels, axis=0)
    return image, boxes, labels
