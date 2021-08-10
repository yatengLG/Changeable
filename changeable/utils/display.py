# -*- coding: utf-8 -*-
# @Author  : LG

from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, Union, Tuple
from numpy import ndarray
from torch import Tensor

def draw_boxes(image:Union[str, ndarray, Image.Image],
               boxes:ndarray,
               labels:ndarray = None,
               scores:ndarray = None,
               label_name:Union[Dict[int, str], Tuple[str, ...]]=None) -> ndarray:
    """
    绘框
    :param image:
    :param boxes:       np.array([[xmin, ymin, xmax, ymax], ...])
    :param labels:
    :param scores:
    :param label_name:
    :param font_path:
    :return:
    """

    IMAGE_FONT = ImageFont.load_default()
    if label_name is not None:
        if not isinstance(label_name, dict):
            label_name = {i:k for i, k in enumerate(label_name)}
        if label_name[0] != '__background__':
            label_name = {i+1:k for i,k in label_name.items()}
            label_name[0] = '__background__'
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    d = ImageDraw.Draw(image, mode='RGBA')

    for i in range(boxes.shape[0]):
        box = boxes[i]
        color = (0, 255, 0)
        text = ''

        if labels is not None:
            label = labels[i]
            color = tuple([int(i * (label ** 2 - label) + i) % 255 for i in (170, 65, 37)])
            text = '{}'.format(label)

            if label_name is not None:
                if label in label_name:
                    label = label_name[label]
                    text = '{}'.format(label)
        if scores is not None:
            text += ':{:.2f}'.format(scores[i])

        d.rectangle(xy=((box[0], box[1]), (box[2], box[3])), fill=None, outline=color, width=1)

        text_w, text_h = IMAGE_FONT.getsize(text)
        # if text != '':
        d.rectangle(xy=((box[0], box[3] - text_h), (box[0] + text_w, box[3])),
                    fill=color + (int(255 * 0.5),), width=0)
        d.text(xy=(box[0], box[3] - text_h), text=text, fill='black', font=IMAGE_FONT)
    return np.array(image)

def plot_image(image:Union[str, ndarray, Image.Image], save_path:str=None):
    """
    显示图片
    :param image:
    :return:
    """
    if isinstance(image, str):
        image = np.array(Image.open(image))
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = np.uint8(image)
    fig, ax = plt.subplots()
    ax.imshow(image, aspect="equal")
    plt.axis("off")
    height, width, _ = image.shape
    fig.set_size_inches(width / 100.0, height / 100.0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    return True
