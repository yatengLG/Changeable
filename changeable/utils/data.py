# -*- coding: utf-8 -*-
# @Author  : LG

import os
from xml.etree import ElementTree as ET
from multiprocessing import Pool
import shutil

class ImageChecker(object):
    """
    图片检测器基类
    """
    def __init__(self, root:str, recursion:bool=False):
        self.all_num = 0
        self.invalid_num = 0

        self.root = root
        self.recursion = recursion
        self.format = ''
        self.start_char = None
        self.end_char = None

    def run(self, root:str):
        files = os.listdir(root)
        for file in files:
            file = os.path.join(root, file)
            if os.path.isfile(file) and file.endswith('.{}'.format(self.format)):
                self.all_num += 1
                with open(file, 'rb')as f:
                    f.seek(-len(self.end_char), 2)
                    if f.read() != self.end_char:
                        print('Found error {} file:{}'.format(self.format, file))
                        self.invalid_num += 1
            elif os.path.isdir(file) and self.recursion:
                self.run(file)
        return True

class JpgChecker(ImageChecker):
    def __init__(self, root:str, recursion:bool=False):
        """
        检查jpg图像完整性，通过设置recursion递归处理子文件夹文件
        :param root:        根目录
        :param recursion:   是否递归处理子文件夹
        """
        super(JpgChecker, self).__init__(root, recursion)
        self.format = 'jpg'
        self.start_char = b'\xff\xd8'
        self.end_char = b'\xff\xd9'

        self.run(self.root)
        print('Found error {} file: {}/{}'.format(self.format, self.invalid_num, self.all_num))

class PngChecker(ImageChecker):
    def __init__(self, root:str, recursion:bool=False):
        """
         检查png图像完整性，通过设置recursion递归处理子文件夹文件
         :param root:        根目录
         :param recursion:   是否递归处理子文件夹
         """
        super(PngChecker, self).__init__(root, recursion)
        self.format = 'png'
        self.start_char = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'
        self.end_char = b'\x00\x00\x00\x00\x49\x45\x4E\x44\xAE\x42\x60\x82'

        self.run(self.root)
        print('Found {} file: {}/{}'.format(self.format, self.invalid_num, self.all_num))

class XmlOp(object):
    """
    xml检测器基类
    """
    def __init__(self, root:str, recursion:bool=False):
        self.root = root
        self.recursion = recursion

    def work(self, xml:str):
        # raise NotImplementedError
        pass
    def run(self, root:str):
        files = os.listdir(root)
        for file in files:
            file = os.path.join(root, file)
            if os.path.isfile(file) and file.endswith('.xml'):
                self.work(file)
            elif os.path.isdir(file) and self.recursion:
                self.run(file)

class XmlReplaceName(XmlOp):
    def __init__(self, root:str, old_name:str, new_name:str, recursion:bool=False):
        """
        替换xml文件类别名
        :param root:        根目录
        :param old_name:    旧类别名
        :param new_name:    新类别名
        :param recursion:   是否递归处理子文件夹
        """
        super(XmlReplaceName, self).__init__(root, recursion)
        self.old_name = old_name
        self.new_name = new_name

        self.run(self.root)

    def work(self, xml:str):
        replace = False
        try:
            tree = ET.parse(xml)
            objs = tree.findall('object')
            for obj in objs:
                name = obj.find('name').text
                if name == self.old_name:
                    obj.find('name').text = self.new_name
                    replace = True
            tree.write(xml)
            if replace:
                print("Replace {} {} from: {}".format(self.old_name, self.new_name, xml))
        except:
            pass

class XmlDelObjectByName(XmlOp):
    def __init__(self, root:str, name:str, recursion:bool=False):
        """
        删除xml中指定类别名的目标
        :param root:        根目录
        :param name:        待删除的目标类别名
        :param recursion:   是否递归处理子文件夹
        """
        super(XmlDelObjectByName, self).__init__(root, recursion)
        self.name = name

        self.run(self.root)

    def work(self, xml:str):
        delete = False
        try:
            tree = ET.parse(xml)
            root = tree.getroot()
            objs = tree.findall('object')
            for obj in objs:
                name = obj.find('name').text
                if name == self.name:
                    root.remove(obj)
                    delete = True
            tree.write(xml)
            if delete:
                print("Delete {} Object from: {}".format(self.name, xml))
        except:
            pass

class XmlIsInvalid(XmlOp):
    def __init__(self, root:str, remove:bool=False, recursion:bool=False):
        """
        查找或删除目录下空的以及无目标的xml文件
        :param root:        根目录
        :param remove:      是否删除查找到的xml文件
        :param recursion:   是否递归处理子文件夹
        """
        super(XmlIsInvalid, self).__init__(root, recursion)
        self.remove = remove

        self.run(self.root)

    def work(self, xml:str):
        try:
            tree = ET.parse(xml)
            objs = tree.findall('object')
            if len(objs) < 1:
                if self.remove:
                    try:
                        os.remove(xml)
                        print('Found and remove no object xml file: {}'.format(xml))
                    except:
                        print('Found no object xml file: {}, but remove failed.'.format(xml))
                else:
                    print('Found no object xml file: {}'.format(xml))

        except:
            if self.remove:
                try:
                    os.remove(xml)
                    print('Found and remove invalid xml file: {}'.format(xml))
                except:
                    print('Found invalid xml file: {}, but remove failed.'.format(xml))
            else:
                print('Found invalid xml file: {}'.format(xml))

class CopyMultiprocess(object):
    def __init__(self, from_root:str, to_root:str, suffix:str=None, num_processes:int=10, recursion:bool=False):
        """
        对文件夹下指定文件执行多线程复制
        :param from_root:       文件夹
        :param to_root:         目标文件夹
        :param suffix:          文件后缀，不指定则复制全部文件
        :param num_processes:   线程数
        :param recursion:       是否递归处理子文件夹(递归处理时，同名文件会被覆盖!)
        """
        self.from_root = from_root
        self.to_root = to_root
        self.suffix = suffix
        self.pool = Pool(num_processes)
        self.recursion = recursion

        self.run()

    def work(self, from_root:str, to_root:str):
        files = os.listdir(from_root)
        for file in files:
            file = os.path.join(from_root, file)
            if os.path.isfile(file):
                if self.suffix is not None and not file.endswith(self.suffix):
                    continue
                to_path = os.path.join(to_root, os.path.split(file)[-1])
                self.pool.apply_async(shutil.copy, (file, to_path, ))

            elif os.path.isdir(file) and self.recursion:
                self.work(file, to_root)

    def run(self):
        self.work(self.from_root, self.to_root)
        self.pool.close()
        self.pool.join()
