# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np
import torch
from changeable.utils.boxes import convert_xyxy_to_cxcywh, convert_cxcywh_to_xyxy, boxes_iou

from typing import Tuple, Union
from numpy import ndarray
from torch import Tensor

class AnchorsGenerator(object):
    """
    anchor生成器

    eg:
        anchors = AnchorsGenerator(image_size=(300, 300),
                               feature_maps_size=((76, 76), (38, 38), (19, 19)),
                               anchors_size=(((10, 13), (16, 30), (33, 23)),
                                             ((30, 61), (62, 45), (59, 119)),
                                             ((116, 90), (156, 198), (373, 326))),
                               form='xyxy',
                               clip=True
                               )
        -> anchors:
        [[  0.           0.           6.97368421   8.47368421]
         [  0.           0.           9.97368421  16.97368421]
         [  0.           0.          18.47368421  13.47368421]
         ...
         [234.10526316 247.10526316 300.         300.        ]
         [214.10526316 193.10526316 300.         300.        ]
         [105.60526316 129.10526316 300.         300.        ]]

    """
    def __init__(self, image_size: Tuple[int, int],
                 feature_maps_size: Tuple[Tuple[int, int], ...],
                 anchors_size: Tuple[Tuple[Tuple[float, float], ...], ...],
                 form: str='xyxy',
                 clip: bool=True):
        """

        :param image_size:          输入图片尺寸
        :param feature_maps_size:   多层特征图尺寸
        :param anchors_size:        每层特征图上anchor尺寸
        :param form:                xyxy or cxcywh
        :param clip:                超出图片anchor是否截断
        """
        self.image_size = image_size
        self.feature_maps_size = feature_maps_size
        self.anchors_size = anchors_size
        self.form = form
        self.clip = clip

        assert len(feature_maps_size) == len(anchors_size)
        assert self.form in ["xyxy", "cxcywh"]

        self.anchors = self.generate_prior()

    def generate_prior(self) -> ndarray:
        """
        生成先验框
        :return: xyxy
        """
        priors = []
        for k, (feature_map_w, feature_map_h) in enumerate(self.feature_maps_size):
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h
                    priors_size = self.anchors_size[k]
                    for w, h in priors_size:
                        priors.append([cx, cy, w / self.image_size[0], h / self.image_size[1]])
        priors = np.array(priors)
        # cxcywh 转 xyxy
        priors = convert_cxcywh_to_xyxy(priors)
        # 截断超出图片范围的框体
        if self.clip:
            priors = priors.clip(0, 1)
        priors = priors * np.array([self.image_size[0], self.image_size[1], self.image_size[0], self.image_size[1]])

        if self.form == "cxcywh":
            priors = convert_xyxy_to_cxcywh(priors)
        return np.array(priors)

    def __repr__(self):
        return "{}".format(self.anchors)

    def detail(self):
        return "image_size: {}\n" \
               "feature_maps_size: {}\n" \
               "priors_size: \n{}" \
               "anchor: \n{}".format(self.image_size,
                                     self.feature_maps_size,
                                     "".join(
                                         [" ".join(["({:>6.2f}, {:>6.2f})".format(p[0], p[1]) for p in ps]) + "\n" for
                                          ps in self.anchors_size]),
                                     self.anchors)

class AnchorsAssigner(object):
    def __init__(self, anchors: AnchorsGenerator, threshold: float):
        self.anchors = anchors
        self.threshold = threshold

    def __call__(self, gt_boxes: ndarray, gt_labels: ndarray):
        raise NotImplementedError

class AnchorsAssignerIOU(AnchorsAssigner):
    """通过iou分配检测框"""
    def __init__(self, anchors: AnchorsGenerator, threshold: float=0.6):
        """
        :param anchors:     检测框 xyxy
        :param threshold:   iou阈值
        """
        assert 0 < threshold <= 1, 'Iou must > 0 and <= 1'
        super(AnchorsAssignerIOU, self).__init__(anchors, threshold)

    def __call__(self, gt_boxes: Union[ndarray, Tensor], gt_labels: Union[ndarray, Tensor]):
        """

        :param gt_boxes:    标注框 xyxy
        :param gt_labels:   标签
        :return:            shape: [n_anchors, 4], [n_anchors]
        """
        anchors = self.anchors.anchors
        if isinstance(gt_boxes, torch.Tensor) and isinstance(gt_labels, torch.Tensor):
            anchors = torch.tensor(anchors)
            ious = boxes_iou(anchors, gt_boxes)
            # size: num_priors
            # 每个检测框对应标注框,及对应标注框id
            best_target_per_prior, best_target_per_prior_index = ious.max(1)
            # size: num_targets
            # 每个标注框对应的最好的检测框,及对应检测框id
            best_prior_per_target, best_prior_per_target_index = ious.max(0)
            # 为每个标注框分配检测框，id
            for target_index, prior_index in enumerate(best_prior_per_target_index):
                best_target_per_prior_index[prior_index] = target_index
            # 替换检测框对应标注框iou， 保证每个标注框都对应一个检测框,iou
            best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)

            labels = gt_labels[best_target_per_prior_index]
            labels[best_target_per_prior < self.threshold] = 0  # 背景
            boxes = gt_boxes[best_target_per_prior_index]

        elif isinstance(gt_boxes, ndarray) and isinstance(gt_labels, ndarray):
            ious = boxes_iou(anchors, gt_boxes)
            best_target_per_prior, best_target_per_prior_index = ious.max(1), ious.argmax(1)
            best_prior_per_target, best_prior_per_target_index = ious.max(0), ious.argmax(0)
            for target_index, prior_index in enumerate(best_prior_per_target_index):
                best_target_per_prior_index[prior_index] = target_index
            for target_index in np.unique(best_prior_per_target_index):
                best_target_per_prior[target_index] = 2
            labels = gt_labels[best_target_per_prior_index]
            labels[best_target_per_prior < self.threshold] = 0  # 背景
            boxes = gt_boxes[best_target_per_prior_index]
        else:
            raise ValueError

        return boxes, labels

class AnchorsAssignerWH(AnchorsAssigner):
    def __init__(self, anchors: AnchorsGenerator, threshold: float=3):
        """

        :param anchors:     检测框 xyxy
        :param threshold:   宽高比阈值
        """
        assert threshold >= 1, 'WH threashold must >= 1'
        super(AnchorsAssignerWH, self).__init__(anchors, threshold)

    def __call__(self, gt_boxes: Union[ndarray, Tensor], gt_labels: Union[ndarray, Tensor]):
        """

        :param gt_boxes:    标注框 xyxy
        :param gt_labels:   标签
        :return:
        """
        input_size = self.anchors.image_size
        anchors = self.anchors.anchors
        feature_maps_size = self.anchors.feature_maps_size
        features_anchor_num = [len(a) for a in self.anchors.anchors_size]

        assert len(anchors) == sum([s[0] * s[1] * n for s, n in zip(feature_maps_size, features_anchor_num)])

        gt_boxes = convert_xyxy_to_cxcywh(gt_boxes)
        anchors = convert_xyxy_to_cxcywh(anchors)

        if isinstance(gt_boxes, torch.Tensor):
            anchors = torch.tensor(anchors)
            gt_boxes = gt_boxes.unsqueeze(0)
            anchors = anchors.unsqueeze(1)
            wh = (gt_boxes[:, :, 2:] / anchors[:, :, 2:])  # [np, nt, 2]
            wh = torch.max(wh, 1. / wh).max(2)[0]  # [np, nt]
            xy = (gt_boxes[:, :, :2] - anchors[:, :, :2]).abs()  # [np, nt, 2]
            xyr = torch.tensor(
                [(input_size[0] // w // 2) ** 2 + (input_size[1] // h // 2) ** 2 for w, h in feature_maps_size])
            index = torch.tensor([w * h * n for (w, h), n in zip(feature_maps_size, features_anchor_num)])
            xy_threshold = torch.repeat_interleave(xyr, repeats=index, dim=0)  # [na]

            xy = xy[..., 0] ** 2 + xy[..., 1] ** 2
            wh[xy > xy_threshold.unsqueeze(1)] = np.inf  # 中心点偏移较大的框，宽高比置为极大
            xy[wh > self.threshold] = np.inf  # 宽高比差异较大的框，中心偏移置为极大

            # 每个检测框对应的最好的标注框
            best_target_per_prior, best_target_per_prior_index = xy.min(1)
            # 每个标注框对应的最好的检测框
            best_prior_per_target, best_prior_per_target_index = wh.min(0)

            for target_index, prior_index in enumerate(best_prior_per_target_index):
                best_target_per_prior_index[prior_index] = target_index
            # 替换检测框对应标注框iou， 保证每个标注框都对应一个检测框,iou
            best_target_per_prior.index_fill_(0, best_prior_per_target_index, 0)

            boxes = gt_boxes.squeeze(0)[best_target_per_prior_index]
            labels = gt_labels[best_target_per_prior_index]
            labels[best_target_per_prior > xy_threshold] = 0

        elif isinstance(gt_boxes, np.ndarray):
            gt_boxes = gt_boxes[np.newaxis, :]
            anchors = anchors[:, np.newaxis]
            wh = (gt_boxes[:, :, 2:] / anchors[:, :, 2:])  # [np, nt, 2]
            wh = np.maximum(wh, 1. / wh).max(2)

            xy = np.abs(gt_boxes[:, :, :2] - anchors[:, :, :2])  # [np, nt, 2]
            xyr = np.array(
                [(input_size[0] // w // 2) ** 2 + (input_size[1] // h // 2) ** 2 for w, h in feature_maps_size])
            index = np.array([w * h * n for (w, h), n in zip(feature_maps_size, features_anchor_num)])
            xy_threshold = np.repeat(xyr, index, 0)

            xy = xy[..., 0] ** 2 + xy[..., 1] ** 2
            wh[xy > xy_threshold[:, np.newaxis]] = np.inf   # 中心点偏移较大的框，宽高比置为极大
            xy[wh > self.threshold] = np.inf    # 宽高比差异较大的框，中心偏移置为极大

            # 每个检测框对应的最好的标注框
            best_target_per_prior, best_target_per_prior_index = xy.min(1), xy.argmin(1)
            best_prior_per_target, best_prior_per_target_index = wh.min(0), wh.argmin(0)

            for target_index, prior_index in enumerate(best_prior_per_target_index):
                best_target_per_prior_index[prior_index] = target_index
            for target_index in np.unique(best_prior_per_target_index):
                best_target_per_prior[target_index] = 0

            boxes = gt_boxes.squeeze(0)[best_target_per_prior_index]
            labels = gt_labels[best_target_per_prior_index]
            labels[best_target_per_prior > xy_threshold] = 0

        else:
            raise ValueError
        boxes = convert_cxcywh_to_xyxy(boxes)
        return boxes, labels
