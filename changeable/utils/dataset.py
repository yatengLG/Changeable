# -*- coding: utf-8 -*-
# @Author  : LG

"""
coco 文件夹结构：
    root -| images -| trainxxx              训练集图片   jpg
                    | valxxx                测试集图片   jpg
         -| labels -| trainxxx              训练集标签   txt
                   -| valxxx                测试集标签   txt
         -| classes.txt                     类别名文件,每类一行

voc 文件夹结构：
    root -| JPEGImages                      图片数据    jpg
         -| Annotations                     标签数据    xml
         -| ImageSets -| Main -| train.txt  训练文件
                              -| val.txt    测试文件
"""

import os
import shutil
import random
import tqdm
from xml.etree import ElementTree as ET
from xml.dom import minidom
from matplotlib import pyplot as plt

def generate_coco(root:str, img_dir:str, txt_dir:str, classes_txt:str, train_rate:float=0.8) -> str:
    """
    将标注数据组织为coco数据集结构
    :param root:        生成数据集的根目录
    :param img_dir:     图片目录
    :param txt_dir:     标注文件目录
    :param classes_txt: 标注类别文件
    :param train_rate:  训练数据占比
    :return:
    """
    coco_root = root
    if os.path.split(root)[-1] != 'coco':
        coco_root = os.path.join(root, 'coco')
    train_image_root = os.path.join(coco_root, 'images', 'train')
    val_image_root = os.path.join(coco_root, 'images', 'val')
    train_label_root = os.path.join(coco_root, 'labels', 'train')
    val_label_root = os.path.join(coco_root, 'labels', 'val')
    classes_txt_path = os.path.join(coco_root, 'classes.txt')

    if os.path.exists(coco_root):
        raise FileExistsError('{} already existed, please delete and try again.'.format(coco_root))
    else:
        os.mkdir(coco_root)
        os.makedirs(train_image_root)
        os.makedirs(val_image_root)
        os.makedirs(train_label_root)
        os.makedirs(val_label_root)
        shutil.copy(classes_txt, classes_txt_path)

    imgs = [f.rstrip('.jpg') for f in os.listdir(img_dir) if f.endswith('.jpg')]
    txts = [f.rstrip('.txt') for f in os.listdir(txt_dir) if f.endswith('.txt')]

    imgs = [f for f in imgs if f in txts]
    txts = [f for f in txts if f in imgs]

    assert len(imgs) == len(txts)
    file_num = len(imgs)
    train_num = int(file_num * train_rate)
    random.shuffle(imgs)
    train_files = imgs[:train_num]
    val_files = imgs[train_num:]
    train_files_bar = tqdm.tqdm(train_files)
    for train_file in train_files_bar:
        train_files_bar.set_description("Copy {}".format(train_file))
        shutil.copy(os.path.join(img_dir, train_file+'.jpg'), os.path.join(train_image_root, train_file+'.jpg'))
        shutil.copy(os.path.join(txt_dir, train_file+'.txt'), os.path.join(train_label_root, train_file+'.txt'))
    val_files_bar = tqdm.tqdm(val_files)
    for val_file in val_files_bar:
        val_files_bar.set_description("Copy {}".format(val_file))

        shutil.copy(os.path.join(img_dir, val_file+'.jpg'), os.path.join(val_image_root, val_file+'.jpg'))
        shutil.copy(os.path.join(txt_dir, val_file+'.txt'), os.path.join(val_label_root, val_file+'.txt'))

    print('Generated coco dataset in {}'.format(coco_root))
    print('    have train image {}.'.format(train_num))
    print('    have val image {}.'.format(file_num-train_num))
    return coco_root

def generate_voc(root:str, img_dir:str, xml_dir:str, train_rate:float=0.8):
    """
    将标注数据组织为voc数据集结构
    :param root:        生成数据集的根目录
    :param img_dir:     图片目录
    :param xml_dir:     标注文件目录
    :param train_rate:  训练数据占比
    :return:
    """
    voc_root = os.path.join(root, 'voc')
    image_root = os.path.join(voc_root, 'JPEGImages')
    label_root = os.path.join(voc_root, 'Annotations')
    txt_root = os.path.join(voc_root, 'ImageSets', 'Main')
    train_txt_path = os.path.join(voc_root, 'ImageSets', 'Main', 'train.txt')
    val_txt_path = os.path.join(voc_root, 'ImageSets', 'Main', 'val.txt')

    if os.path.exists(voc_root):
        raise FileExistsError('{} already existed, please delete and try again.'.format(voc_root))
    else:
        os.mkdir(voc_root)
        os.makedirs(image_root)
        os.makedirs(label_root)
        os.makedirs(txt_root)

    imgs = [f.rstrip('.jpg') for f in os.listdir(img_dir) if f.endswith('.jpg')]
    xmls = [f.rstrip('.xml') for f in os.listdir(xml_dir) if f.endswith('.xml')]

    imgs = [f for f in imgs if f in xmls]
    xmls = [f for f in xmls if f in imgs]

    assert len(imgs) == len(xmls)
    imgs_bar = tqdm.tqdm(imgs)
    for img in imgs_bar:
        imgs_bar.set_description('Copy {}:'.format(img))
        shutil.copy(os.path.join(img_dir, img + '.jpg'), os.path.join(image_root, img + '.jpg'))
        shutil.copy(os.path.join(xml_dir, img + '.xml'), os.path.join(label_root, img + '.xml'))

    file_num = len(imgs)
    train_num = int(file_num * train_rate)
    random.shuffle(imgs)
    train_files = imgs[:train_num]
    val_files = imgs[train_num:]

    with open(train_txt_path, 'w') as f:
        for train_file in train_files:
            f.write(train_file+'\n')

    with open(val_txt_path, 'w') as f:
        for val_file in val_files:
            f.write(val_file+'\n')

    print('Generated voc dataset in {}'.format(voc_root))
    print('    have train image {}.'.format(train_num))
    print('    have val image {}.'.format(file_num - train_num))
    return voc_root

def voc_to_coco(voc_path:str, keep_truncated:bool=True, keep_difficult:bool=True):
    """
    voc数据集转coco数据集
    :param voc_path:        voc数据集目录
    :param keep_truncated:  保留截断目标
    :param keep_difficult:  保留难识别目标
    :return:
    """
    coco_root = os.path.join(os.path.split(voc_path)[0], 'coco')
    train_image_root = os.path.join(coco_root, 'images', 'train')
    val_image_root = os.path.join(coco_root, 'images', 'val')
    train_label_root = os.path.join(coco_root, 'labels', 'train')
    val_label_root = os.path.join(coco_root, 'labels', 'val')
    classes_txt_path = os.path.join(coco_root, 'classes.txt')

    if os.path.exists(coco_root):
        raise FileExistsError('{} already existed, please delete and try again.'.format(coco_root))
    else:
        os.mkdir(coco_root)
        os.makedirs(train_image_root)
        os.makedirs(val_image_root)
        os.makedirs(train_label_root)
        os.makedirs(val_label_root)

    xml_path = os.path.join(voc_path, 'Annotations')
    xmls = os.listdir(xml_path)
    label_set = set()
    for xml in xmls:
        tree = ET.parse(os.path.join(xml_path, xml))
        objects = tree.findall('object')
        for obj in objects:
            name = obj.find('name').text
            if name not in label_set:
                label_set.add(name)

    label_id_dict = {n:i for i, n in enumerate(label_set)}

    with open(classes_txt_path, 'w') as f:
        id_label_dict = {i:k for k,i in label_id_dict.items()}
        for i in range(len(label_id_dict)):
            f.write(id_label_dict[i]+'\n')

    train_txt = os.path.join(voc_path, 'ImageSets', 'Main', 'train.txt')
    val_txt = os.path.join(voc_path, 'ImageSets', 'Main', 'val.txt')

    train_ids = [line.rstrip('\n') for line in open(train_txt, 'r').readlines()]
    val_ids = [line.rstrip('\n') for line in open(val_txt, 'r').readlines()]

    train_bar = tqdm.tqdm(train_ids)
    for train_id in train_bar:
        train_bar.set_description('Copy train data {}'.format(train_id))
        try:
            shutil.copy(os.path.join(voc_path, 'JPEGImages', train_id+'.jpg'),
                        os.path.join(train_image_root, train_id+'.jpg'))

            tree = ET.parse(os.path.join(voc_path, 'Annotations', train_id+'.xml'))
            width = int(tree.find('size').find('width').text)
            height = int(tree.find('size').find('height').text)
            objects = tree.findall('object')

            with open(os.path.join(train_label_root, train_id+'.txt'), 'w') as f:
                for obj in objects:
                    name = obj.find('name').text

                    truncated = obj.find('truncated').text
                    difficult = obj.find('difficult').text
                    if keep_truncated==False and truncated == '1':
                        continue
                    if keep_difficult==False and difficult == '1':
                        continue

                    xmin = int(obj.find('bndbox').find('xmin').text)
                    ymin = int(obj.find('bndbox').find('ymin').text)
                    xmax = int(obj.find('bndbox').find('xmax').text)
                    ymax = int(obj.find('bndbox').find('ymax').text)
                    f.write('{} {} {} {} {}\n'.format(label_id_dict[name],
                                                    (xmax+xmin)/2/(width),
                                                    (ymax+ymin)/2/height,
                                                    (xmax-xmin)/width,
                                                    (ymax-ymin)/height
                                                    ))
        except:
            pass

    val_bar = tqdm.tqdm(val_ids)
    for val_id in val_bar:
        val_bar.set_description('Copy val data {}'.format(val_id))

        try:
            shutil.copy(os.path.join(voc_path, 'JPEGImages', val_id+'.jpg'),
                        os.path.join(val_image_root, val_id+'.jpg'))

            tree = ET.parse(os.path.join(voc_path, 'Annotations', val_id+'.xml'))
            width = int(tree.find('size').find('width').text)
            height = int(tree.find('size').find('height').text)
            objects = tree.findall('object')

            with open(os.path.join(val_label_root, val_id+'.txt'), 'w') as f:
                for obj in objects:
                    name = obj.find('name').text

                    truncated = obj.find('truncated').text
                    difficult = obj.find('difficult').text
                    if keep_truncated==False and truncated == '1':
                        continue
                    if keep_difficult==False and difficult == '1':
                        continue

                    xmin = int(obj.find('bndbox').find('xmin').text)
                    ymin = int(obj.find('bndbox').find('ymin').text)
                    xmax = int(obj.find('bndbox').find('xmax').text)
                    ymax = int(obj.find('bndbox').find('ymax').text)
                    f.write('{} {} {} {} {}\n'.format(label_id_dict[name],
                                                    (xmax+xmin)/2/(width),
                                                    (ymax+ymin)/2/height,
                                                    (xmax-xmin)/width,
                                                    (ymax-ymin)/height
                                                    ))
        except:
            pass

    return coco_root

def coco_to_voc(coco_path:str):
    """
    coco数据集转voc数据集
    :param coco_path:
    :return:
    """
    voc_root = os.path.join(os.path.split(coco_path)[0], 'voc')
    image_root = os.path.join(voc_root, 'JPEGImages')
    label_root = os.path.join(voc_root, 'Annotations')
    txt_root = os.path.join(voc_root, 'ImageSets', 'Main')
    train_txt_path = os.path.join(voc_root, 'ImageSets', 'Main', 'train.txt')
    val_txt_path = os.path.join(voc_root, 'ImageSets', 'Main', 'val.txt')

    if os.path.exists(voc_root):
        raise FileExistsError('{} already existed, please delete and try again.'.format(voc_root))
    else:
        os.mkdir(voc_root)
        os.makedirs(image_root)
        os.makedirs(label_root)
        os.makedirs(txt_root)

    train_ids = [f.rstrip('.txt') for f in os.listdir(os.path.join(coco_path, 'labels', 'train')) if f.endswith('.txt')]
    with open(train_txt_path, 'w') as f:
        for train_id in train_ids:
            f.write(train_id+'\n')
    val_ids = [f.rstrip('.txt') for f in os.listdir(os.path.join(coco_path, 'labels', 'val')) if f.endswith('.txt')]
    with open(val_txt_path, 'w') as f:
        for val_id in val_ids:
            f.write(val_id+'\n')

    train_image_root = os.path.join(coco_path, 'images', 'train')
    val_image_root = os.path.join(coco_path, 'images', 'val')
    images_path = [os.path.join(train_image_root, img) for img in os.listdir(train_image_root) if img.endswith('.jpg')]
    images_path.extend([os.path.join(val_image_root, img) for img in os.listdir(val_image_root) if img.endswith('.jpg')])

    for img_path in images_path:
        shutil.copy(img_path, os.path.join(image_root, os.path.split(img_path)[-1]))

    coco_index_label_dict = {}
    with open(os.path.join(coco_path, 'classes.txt'), 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            coco_index_label_dict[i] = line.rstrip('\n')

    train_label_root = os.path.join(coco_path, 'labels', 'train')
    val_label_root = os.path.join(coco_path, 'labels', 'val')
    labels_path = [os.path.join(train_label_root, img) for img in os.listdir(train_label_root) if img.endswith('.txt')]
    labels_path.extend(
        [os.path.join(val_label_root, img) for img in os.listdir(val_label_root) if img.endswith('.txt')])

    for txt in labels_path:
        try:
            h, w, c = plt.imread(os.path.join(image_root, os.path.split(txt)[-1].rstrip('.txt') + '.jpg')).shape
            with open(os.path.join(txt), 'r') as f:
                lines = f.readlines()

            domTree = minidom.Document()

            annotation_node = domTree.createElement('annotation')

            size_node = domTree.createElement('size')

            width_node = domTree.createElement('width')
            width_node.appendChild(domTree.createTextNode('{}'.format(w)))

            height_node = domTree.createElement('height')
            height_node.appendChild(domTree.createTextNode('{}'.format(h)))

            depth_node = domTree.createElement('depth')
            depth_node.appendChild(domTree.createTextNode('{}'.format(c)))

            size_node.appendChild(width_node)
            size_node.appendChild(height_node)
            size_node.appendChild(depth_node)

            annotation_node.appendChild(size_node)

            for line in lines:
                index, x_, y_, w_, h_ = line.split()
                label = coco_index_label_dict[int(index)]
                xmin = max(0, int((float(x_) - float(w_) / 2) * w))
                ymin = max(0, int((float(y_) - float(h_) / 2) * h))
                xmax = max(0, int((float(x_) + float(w_) / 2) * w))
                ymax = max(0, int((float(y_) + float(h_) / 2) * h))

                object_node = domTree.createElement('object')

                name_node = domTree.createElement('name')
                name_node.appendChild(domTree.createTextNode('{}'.format(label)))

                pose_node = domTree.createElement('pose')
                pose_node.appendChild(domTree.createTextNode('{}'.format(0)))

                truncated_node = domTree.createElement('truncated')
                truncated_node.appendChild(domTree.createTextNode('{}'.format(0)))

                difficult_node = domTree.createElement('difficult')
                difficult_node.appendChild(domTree.createTextNode('{}'.format(0)))

                bndbox_node = domTree.createElement('bndbox')
                xmin_node = domTree.createElement('xmin')
                xmin_node.appendChild(domTree.createTextNode('{}'.format(xmin)))
                ymin_node = domTree.createElement('ymin')
                ymin_node.appendChild(domTree.createTextNode('{}'.format(ymin)))
                xmax_node = domTree.createElement('xmax')
                xmax_node.appendChild(domTree.createTextNode('{}'.format(xmax)))
                ymax_node = domTree.createElement('ymax')
                ymax_node.appendChild(domTree.createTextNode('{}'.format(ymax)))

                bndbox_node.appendChild(xmin_node)
                bndbox_node.appendChild(ymin_node)
                bndbox_node.appendChild(xmax_node)
                bndbox_node.appendChild(ymax_node)

                object_node.appendChild(name_node)
                object_node.appendChild(pose_node)
                object_node.appendChild(truncated_node)
                object_node.appendChild(difficult_node)
                object_node.appendChild(bndbox_node)

                annotation_node.appendChild(object_node)
            domTree.appendChild(annotation_node)

            with open(os.path.join(label_root, os.path.split(txt)[-1].rstrip('.txt') + '.xml'), 'w') as f:
                domTree.writexml(f, addindent='\t', newl='\n', encoding='utf-8')
        except  Exception as e:
            print("error file: {}, {}".format(os.path.split(txt)[-1], e))
    return voc_root

