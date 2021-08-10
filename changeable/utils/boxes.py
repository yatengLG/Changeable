# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np
import torch
from torch import Tensor
from numpy import ndarray
from typing import Union

def convert_xyxy_to_cxcywh(boxes: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    将包围盒格式从[xmin, ymin, xmax, ymax]转为[cx, cy, w, h]
    :param boxes:   shape[..., 4]
    :return:
    """
    cat = np.concatenate if isinstance(boxes, np.ndarray) else torch.cat
    return cat(((boxes[..., :2] + boxes[..., 2:]) / 2, (boxes[..., 2:] - boxes[..., :2])), -1)

def convert_cxcywh_to_xyxy(boxes: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    将包围盒格式从[cx, cy, w, h]转为[xmin, ymin, xmax, ymax]
    :param boxes:   shape[..., 4]
    :return:
    """
    cat = np.concatenate if isinstance(boxes, np.ndarray) else torch.cat
    return cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:]/2), -1)

def boxes_area(boxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    计算包围盒面积     框体格式[xmin, ymin, xmax, ymax]
    :param boxes:   shape[..., 4]
    :return:
    """
    area_wh = boxes[..., 2:] - boxes[..., :2]
    area_wh[area_wh<0] = 0
    return area_wh[..., 0] * area_wh[..., 1]

def boxes_iou(boxes1: Union[torch.Tensor, np.ndarray],
              boxes2: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    计算两个框的交并比   框体格式[xmin, ymin, xmax, ymax]
    :param boxes1:  shape[..., 4]
    :param boxes2:  shape[..., 4]
    :return:
    """
    area1 = boxes_area(boxes1)
    area2 = boxes_area(boxes2)
    if isinstance(boxes1, np.ndarray) and isinstance(boxes2, np.ndarray):
        lt = np.maximum(boxes1[..., np.newaxis, :2], boxes2[..., :2])
        rb = np.minimum(boxes1[..., np.newaxis, 2:], boxes2[..., 2:])
        wh = np.clip(rb-lt, 0, np.inf)
        area1 = area1[..., np.newaxis]
        area2 = area2[np.newaxis, ...]

    elif isinstance(boxes1, torch.Tensor) and isinstance(boxes2, torch.Tensor):
        lt = torch.max(boxes1[..., None, :2], boxes2[..., :2])
        rb = torch.min(boxes1[..., None, 2:], boxes2[..., 2:])
        wh = torch.clip(rb-lt, 0)
        area1 = area1[..., None]
        area2 = area2[None, ...]
    else:
        raise ValueError
    overlap_area = wh[..., 0] * wh[...,1]
    return overlap_area / (area1 + area2 - overlap_area)

def boxes_cover(boxes1: Union[torch.Tensor, np.ndarray],
                boxes2: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    计算第一个框体被第二个框体遮挡部分占第一个框体面积的百分比   框体格式[xmin, ymin, xmax, ymax]
    :param boxes1:  shape[..., 4]
    :param boxes2:  shape[..., 4]
    :return:
    """
    area1 = boxes_area(boxes1)
    if isinstance(boxes1, np.ndarray) and isinstance(boxes2, np.ndarray):
        lt = np.maximum(boxes1[..., np.newaxis, :2], boxes2[..., :2])
        rb = np.minimum(boxes1[..., np.newaxis, 2:], boxes2[..., 2:])
        wh = np.clip(rb-lt, 0, np.inf)
        area1 = area1[..., np.newaxis]

    elif isinstance(boxes1, torch.Tensor) and isinstance(boxes2, torch.Tensor):
        lt = torch.max(boxes1[..., None, :2], boxes2[..., :2])
        rb = torch.min(boxes1[..., None, 2:], boxes2[..., 2:])
        wh = torch.clip(rb-lt, 0)
        area1 = area1[..., None]
    else:
        raise ValueError
    overlap_area = wh[..., 0] * wh[...,1]
    return overlap_area / area1