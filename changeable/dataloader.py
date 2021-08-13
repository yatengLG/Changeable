# -*- coding: utf-8 -*-
# @Author  : LG
from .transfunc import adaptive_resize, generate_mosaic
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler
from changeable.anchor import AnchorsAssigner
from typing import Tuple

class InfiniteBatchSampler(Sampler):
    def __init__(self, sampler, batch_size):
        super(InfiniteBatchSampler, self).__init__(None)
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        while True:
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

class MosaicBatchCollator:
    def __init__(self, size: Tuple[int, int], anchors_assigner: AnchorsAssigner,
                 use_mosaic: bool=False):
        self.anchors_assigner = anchors_assigner
        self.size = size
        self.use_mosaic = use_mosaic

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = transposed_batch[0]
        boxes = transposed_batch[1]
        labels = transposed_batch[2]
        img_ids = transposed_batch[3]
        if self.use_mosaic:
            images, boxes, labels, img_ids = self.mosaic_batch(images, boxes, labels, img_ids)
        else:
            images, boxes, labels, img_ids = self.general_batch(images, boxes, labels, img_ids)
        images = default_collate(images)
        boxes = default_collate(boxes)
        labels = default_collate(labels)
        img_ids = default_collate(img_ids)
        return images.permute((0, 3, 1, 2)), boxes, labels, img_ids

    def general_batch(self, images, boxes, labels, img_ids):
        images_batch = []
        boxes_batch = []
        labels_batch = []
        ids_batch = []
        for img, box, lab, id in zip(images, boxes, labels, img_ids):
            img, box, lab = img.copy(), box.copy(), lab.copy()
            if len(box) == 0:
                raise ValueError('Pic {} has no object!')
            if self.size is not None:
                img, box = adaptive_resize(img, self.size, box)
            if self.anchors_assigner is not None:
                box, lab = self.anchors_assigner(box, lab)
            images_batch.append(img)
            boxes_batch.append(box)
            labels_batch.append(lab)
            ids_batch.append(id)
        return images_batch, boxes_batch, labels_batch, ids_batch

    def mosaic_batch(self, images, boxes, labels, img_ids):
        images_batch = []
        boxes_batch = []
        labels_batch = []
        ids_batch = []
        for i in range(0, len(images), 4):
            img, box, lab = generate_mosaic(images[i:i+4], boxes[i:i+4], labels[i:i+4], mosaic_size=self.size)
            if self.anchors_assigner is not None:
                box, lab = self.anchors_assigner(box, lab)
            images_batch.append(img)
            boxes_batch.append(box)
            labels_batch.append(lab)
            ids_batch.append('_'.join(img_ids[i:i+4]))
        return images_batch, boxes_batch, labels_batch, ids_batch


def dataloader(dataset, batch_size=1, resize: Tuple[int, int]=None, anchors_assigner=None, shuffle=True, num_workers=0,
               use_mosaic=False):
    """
    数据加载器。
    通过指定的anchor分配器，将
    可指定使用mosaic数据增强，将四张图组合成mosaic图片。

    :param dataset:             数据集，torch.data.dataset，输出为(image:ndarry, boxes:ndarry, labels:ndarry, id:str)
    :param batch_size:          批次
    :param resize:              归一化后的尺寸(使用mosaic时，mosaic图片尺寸；不使用mosaic时，resize填充灰边，保持宽高比)
    :param anchors_assigner:    anchor分配器, AnchorsAssigner,
    :param shuffle:             是否打乱数据集
    :param num_workers:         线程数
    :param use_mosaic:          是否使用mosaic数据增强
    :return:                    提供anchor分配器:
                                    images: [B, C, W, H], boxes: [num_anchors, 4], labels: [num_anchors],
                                不提供anchor分配器:
                                    images: [1, C, W, H], boxes: [num_object, 4], labels: [num_object],

    """
    if resize is None:
        assert batch_size == 1 and not use_mosaic, 'When batch size is not 1 or use mosaic, must provide resize'

    if use_mosaic:  # 使用mosaic时，然后四张图合成一张图，批次乘4
        batch_size = batch_size * 4

    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = InfiniteBatchSampler(sampler, batch_size)

    loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler,
                        collate_fn=MosaicBatchCollator(size=resize, anchors_assigner=anchors_assigner,
                                                       use_mosaic=use_mosaic),
                        num_workers=num_workers)
    return loader

