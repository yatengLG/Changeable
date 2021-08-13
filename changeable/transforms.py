# -*- coding: utf-8 -*-
# @Author  : LG

import torch
import numpy as np
from .utils.boxes import convert_cxcywh_to_xyxy, convert_xyxy_to_cxcywh, boxes_cover
from .transfunc import crop_iou, crop_size, adaptive_resize
import random
import cv2

from PIL import Image
from numpy import ndarray
from typing import Union, Tuple


class ToTensor(object):
    def __call__(self, image: Union[ndarray, Image.Image], boxes: ndarray=None, labels: ndarray=None):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float(), \
               boxes if boxes is None else torch.from_numpy(boxes.copy()).float(), \
               labels if labels is None else torch.from_numpy(labels.copy()).float()

class ConvertBoxesToValue(object):
    """将包围盒从百分比转换为值形式"""
    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        boxes = boxes.copy()
        h, w, _ = image.shape
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        return image, boxes, labels

class ConvertBoxesToPercentage(object):
    """包围盒转换为百分比形式, [x, y, x, y]"""
    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        boxes = boxes.copy()
        h, w, _ = image.shape
        boxes[:, 0] = boxes[:, 0] / w
        boxes[:, 2] = boxes[:, 2] / w
        boxes[:, 1] = boxes[:, 1] / h
        boxes[:, 3] = boxes[:, 3] / h
        return image, boxes, labels

class ConvertBoxesForm(object):
    """包围盒形式转换：cxcywh与xyxy"""
    def __init__(self, from_form: str, to_form: str):
        assert from_form in ('xyxy', 'cxcywh')
        convert_fn = None
        if from_form == 'xyxy' and to_form == 'cxcywh':
            convert_fn = convert_xyxy_to_cxcywh
        elif from_form == 'cxcywh' and to_form == 'xyxy':
            convert_fn = convert_cxcywh_to_xyxy
        self.convert_fn = convert_fn

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray = None) -> Tuple[ndarray, ndarray, ndarray]:
        boxes = boxes.copy()
        if self.convert_fn is not None:
            boxes = self.convert_fn(boxes)
        return image, boxes, labels

class Resize(object):
    """resize"""
    def __init__(self, size: Tuple[int, int]):
        """

        :param size:    一组值 (w, h)
        """
        self.size = size

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray=None):
        image, boxes = image.copy(), boxes.copy()
        h, w, _ = image.shape
        image = Image.fromarray(np.uint8(image))
        image = np.array(image.resize(size=self.size), dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] / w * self.size[0]
        boxes[:, 1] = boxes[:, 1] / h * self.size[1]
        boxes[:, 2] = boxes[:, 2] / w * self.size[0]
        boxes[:, 3] = boxes[:, 3] / h * self.size[1]
        return image, boxes, labels

class AdaptiveResize(object):
    """自适应resize"""
    def __init__(self, size: Tuple[int, int],
                 value: Union[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]]=(114, 114, 114)):
        """
        短边自动扩充，不改变图片比例
        :param size:    resize尺寸    一组值, (w, h)
        :param value:   短边填充值     一组值或多组值
        """
        self.size = size
        self.value = value

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray=None):
        image, boxes = image.copy(), boxes.copy()
        value = random.choice(self.value) if isinstance(self.value[0], tuple) else self.value
        image, boxes = adaptive_resize(image, self.size, boxes, value)
        return image, boxes, labels

class Scaled(object):
    def __init__(self, scale: Union[float, Tuple[float, float]]):
        """

        :param scale:   缩放幅度。指定一个值或一个范围
        """
        self.scale = scale

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray=None):
        image, boxes = image.copy(), boxes.copy()
        scale = random.uniform(self.scale[0], self.scale[1]) if isinstance(self.scale, tuple) else self.scale
        h, w, _ = image.shape
        image = Image.fromarray(np.uint8(image))
        image = image.resize(size=(int(w*scale), int(h*scale)))
        image = np.array(image, dtype=np.float32)
        boxes = boxes * scale
        return image, boxes, labels

class CropIou(object):
    """IOU裁剪"""
    def __init__(self, iou:Union[float, Tuple[float, ...]]):
        """

        :param iou: iou阈值。指定一个值或多个值
        """
        self.iou = iou

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        image, boxes, labels = image.copy(), boxes.copy(), labels.copy()
        if boxes.shape[0] == 0:
            return image, boxes, labels
        h, w, _ = image.shape
        iou = random.choice(self.iou) if isinstance(self.iou, tuple) else self.iou
        return crop_iou(image, boxes, labels, min_iou=iou)

class CropSize(object):
    """随机尺寸裁剪"""
    def __init__(self, size:Union[Tuple[int, int], Tuple[Tuple[int, int], ...]]):
        """

        :param size:    裁剪尺寸。指定一个值或多个值
        """
        self.size = size

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        image, boxes, labels = image.copy(), boxes.copy(), labels.copy()
        if boxes.shape[0] == 0:
            return image, boxes, labels
        size = random.choice(self.size) if isinstance(self.size[0], tuple) else self.size
        return crop_size(image, boxes, labels, size=size)

class SubtractMeans(object):
    """减均值"""
    def __init__(self, mean: Tuple[float, float, float]):
        self.mean = mean

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        image = image.copy() - self.mean
        return image.astype(np.float32), boxes, labels

class DivideStds(object):
    """除方差"""
    def __init__(self, std: Tuple[float, float, float]):
        self.std = std

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        image = image.copy() / self.std
        return image.astype(np.float32), boxes, labels

class GaussNoise(object):
    """添加高斯噪声"""
    def __init__(self, scale: Union[float, Tuple[float, float]]=(0, 0.1), probability: float=0.5):
        """

        :param scale:       标准差 指定一个值或给定一个范围
        :param probability:
        """
        assert 0 <= np.min(scale) <= np.max(scale) <= 1
        self.scale = scale
        self.probability = probability
        if isinstance(self.scale, tuple):
            assert len(self.scale) == 2 and self.scale[0] <= self.scale[1] <= 1

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        image = image.copy()
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        scale = random.uniform(self.scale[0], self.scale[1]) if isinstance(self.scale, tuple) else self.scale
        image = image/255 + np.random.normal(0, scale=scale, size=image.shape)
        image = np.clip(image, 0, 1) * 255
        return image.astype(np.float32), boxes, labels

class SalePepperNoise(object):
    """添加椒盐噪声"""
    def __init__(self, sale_scale: Union[float, Tuple[float, float]]=(0, 0.03),
                       pepper_scale: Union[float, Tuple[float, float]]=(0, 0.03), probability: float=0.5):
        """
        先撒盐再撒椒,可能覆盖
        :param sale:    盐概率 指定一个值或一个范围
        :param pepper:  椒概率 指定一个值或一个范围
        :param probability:
        """
        self.sale = sale_scale
        self.pepper = pepper_scale
        self.probability = probability
        assert 0 <= np.min(self.sale) <= np.max(self.sale) <= 1
        assert 0 <= np.min(self.pepper) <= np.max(self.pepper) <= 1

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        image = image.copy()
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        sale = random.uniform(self.sale[0], self.sale[1]) if isinstance(self.sale, tuple) else self.sale
        pepper = random.uniform(self.pepper[0], self.pepper[1]) if isinstance(self.pepper, tuple) else self.pepper
        sale_mask = np.bool8(np.random.binomial(1, sale, size=image.shape[:2]))
        pepper_mask = np.bool8(np.random.binomial(1, pepper, size=image.shape[:2]))
        image[sale_mask] = 255
        image[pepper_mask] = 0

        return image, boxes, labels

class GaussBlur(object):
    """高斯模糊"""
    def __init__(self, ksize: Union[int, Tuple[int, ...]]=(3, 5, 7, 9), probability: float=0.5):
        """

        :param ksize:      核大小 越大模糊程度越高，必须为奇数。指定一组或多组值
        :param probability:
        """
        assert np.all(np.array(ksize) % 2 == 1)
        assert 1<= np.min(ksize) <= np.max(ksize) <= 9
        self.ksize = ksize
        self.probability = probability

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        image = image.copy()
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        ksize = self.ksize if isinstance(self.ksize, int) else random.choice(self.ksize)
        image = cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=2)
        return image, boxes, labels

class MotionBlue(object):
    """运动模糊"""
    def __init__(self, ksize: Union[int, Tuple[int, ...]]=(3, 5, 7, 9, 11),
                 angle: Union[int, Tuple[int, int]] = (0, 30),
                 probability: float=0.5):
        """

        :param ksize:      核大小，越大模糊程度越高，必须是奇数。指定一组或多组值
        :param angle:       运动模糊角度。指定一个值或一个范围
        :param probability:
        """
        assert np.all(np.array(ksize) % 2 == 1)
        self.ksize = ksize
        self.angle = angle
        self.probability = probability

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        image = image.copy()
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        ksize = random.choice(self.ksize) if isinstance(self.ksize, tuple) else self.ksize
        angle = random.randint(self.angle[0], self.angle[1]) if isinstance(self.angle, tuple) else self.angle
        rm= cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle+45, 1)
        kernel = cv2.warpAffine(np.eye(ksize, ksize), rm, (ksize, ksize))
        kernel = kernel / np.sum(kernel)
        image = cv2.filter2D(image, -1, kernel)
        return image, boxes, labels

class Cutout(object):
    """随机遮挡"""
    def __init__(self, num: Union[int, Tuple[int, ...]]=(1, 2, 3, 4),
                 size: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]=((0.2, 0.2), (0.3, 0.3)),
                 value:Union[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]]=(114, 114, 114),
                 cover: float=0.5, probability: float=0.5):
        """
        被遮挡率大于阈值的目标，将被舍弃
        :param num:     遮挡块数量, 指定一个或多个值
        :param size:    遮挡块对应原图比例， 指定一组或多组值
        :param value:   遮挡块填充值, 指定一组或多组值
        :param cover:   遮挡阈值，被遮挡率大于该值的框将被舍弃
        """
        self.num = num
        self.size = size
        self.value = value
        self.probability = probability
        self.cover = cover

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        image, boxes, labels = image.copy(), boxes.copy(), labels.copy()
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        h, w, _ = image.shape
        num = random.choice(self.num) if isinstance(self.num, tuple) else self.num
        size = random.choice(self.size) if isinstance(self.size[0], tuple) else self.size
        value = random.choice(self.value) if isinstance(self.value[0], tuple) else self.value
        ch, cw = int(h * size[1]/2), int(w * size[0]/2)
        xs, ys = random.choices(range(w), k=num), random.choices(range(h), k=num)
        cutouts = np.array([[max(0, x-cw), max(0, y-ch), min(x+cw, w), min(y+ch, h)] for x, y in zip(xs, ys)])
        covers = boxes_cover(boxes, cutouts)
        mask = np.max(covers, 1) < self.cover
        if not np.all(mask):    # 无目标时，直接返回原数据
            return image, boxes, labels
        for cutout in cutouts:
            image[cutout[1]:cutout[3], cutout[0]:cutout[2], :] = value
        boxes = boxes[mask]
        labels = labels[mask]
        return image, boxes, labels

class RandomFlipLR(object):
    """随机左右翻转"""
    def __init__(self, probability: float=0.5):
        self.probability = probability

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        image, boxes = image.copy(), boxes.copy()
        _, w, _ = image.shape
        image = image[:, ::-1]
        boxes[:, 0::2] = w - boxes[:, 2::-2]
        return image, boxes, labels

class RandomFlipUD(object):
    """随机上下翻转"""
    def __init__(self, probability: float=0.5):
        self.probability = probability

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        image, boxes = image.copy(), boxes.copy()
        h, _, _ = image.shape
        image = image[::-1]
        boxes[:, 1::2] = h - boxes[:, 3::-2]
        return image, boxes, labels

class ShuffleChannels(object):
    """随机交换通道"""
    def __init__(self, mode: Union[Tuple[int, int, int], Tuple[Tuple[int, int, int], ]]=((0, 2, 1), (1, 0, 2), (1, 2, 0),(2, 0, 1), (2, 1, 0)),
                 probability: float=0.5):
        """

        :param mode:        通道交换模式。指定一组值或多组值。
        :param probability:
        """
        self.mode = mode
        self.probability = probability

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        mode = random.choice(self.mode) if isinstance(self.mode[0], tuple) else self.mode
        image = image[:, :, mode]
        return image, boxes, labels

class ChangeContrast(object):
    """随机图片对比度"""
    def __init__(self, scale:Union[float, Tuple[float, float]]=(-0.5, 0.5), probability: float=0.5):
        """

        :param scale:       对比度变化幅度。指定一个值或一个范围。[-1, 1]，-1时，图片为同色。
        :param probability:
        """
        self.scale = scale
        self.probability = probability
        assert -0.8 <= np.min(scale) <= np.max(scale) <= 1

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        image = image.copy()
        scale = random.uniform(self.scale[0], self.scale[1]) if isinstance(self.scale, tuple) else self.scale
        means = np.mean(image, axis=(0, 1), keepdims=True)
        image = (image-means) * scale + image
        image = np.clip(image, 0, 255)
        return image, boxes, labels

class ChangeHue(object):
    """随机图片色调"""
    def __init__(self, scale: Union[int, Tuple[int, int]]=(0, 360), probability: float=0.5):
        """

        :param scale:       色调变化幅度，HSV色调360度色值。指定一个值或一个范围，360度与0度是一个色调。
        :param probability:
        """
        self.scale = scale
        self.probability = probability
        assert 0 <= np.min(scale) <= np.max(scale) <= 360

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        image = image.copy()
        scale = random.uniform(self.scale[0], self.scale[1]) if isinstance(self.scale, tuple) else self.scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 0] = image[:, :, 0] + scale
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, boxes, labels

class ChangeSaturation(object):
    """随机图片饱和度"""
    def __init__(self, scale: Union[float, Tuple[float, float]]=(-1, 1), probability: float=0.5):
        """

        :param scale:       饱和度变化幅度。指定一个值或一个范围
        :param probability:
        """
        self.scale = scale
        self.probability = probability
        assert -1 <= np.min(scale) <= np.max(scale) <= 1

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        image = image.copy()
        scale = random.uniform(self.scale[0], self.scale[1]) if isinstance(self.scale, tuple) else self.scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 1] = image[:, :, 1] + scale
        image[:, :, 1] = np.clip(image[:, :, 1], 0, 1)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, boxes, labels

class ChangeBrightness(object):
    """随机图片亮度"""
    def __init__(self, scale: Union[float, Tuple[float, float]]=(-0.3, 0.3), probability: float=0.5):
        """

        :param scale:       亮度变化幅度。指定一个值或一个范围。-1时，亮度为0，全黑。
        :param probability:
        """
        self.scale = scale
        self.probability = probability
        assert -0.5 <= np.min(scale) <= np.max(scale) <= 1, ''

    def __call__(self, image: ndarray, boxes: ndarray=None, labels: ndarray=None) -> Tuple[ndarray, ndarray, ndarray]:
        if random.uniform(0, 1) > self.probability:
            return image, boxes, labels
        image = image.copy()
        scale = random.uniform(self.scale[0], self.scale[1]) if isinstance(self.scale, tuple) else self.scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) / 255
        image[:, :, 2] = image[:, :, 2] + scale
        image[:, :, 2] = np.clip(image[:, :, 2], 0, 1)
        image = cv2.cvtColor(image*255, cv2.COLOR_HSV2BGR)
        return image, boxes, labels

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: ndarray, boxes: ndarray, labels: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        image = np.float32(image)
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '\t{:<30}    # {}'.format(t.__class__.__name__, t.__class__.__doc__)
        format_string += '\n)'
        return format_string