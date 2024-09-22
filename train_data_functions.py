from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #


# coding=utf-8

import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from PIL import Image


class TrainData(Dataset):

    def __init__(self, root, crop_size):

        super(TrainData, self).__init__()
        self.root = root
        self.crop_size = crop_size
        self.inp_root, self.gt_root = os.path.join(root, 'input'), os.path.join(root, 'gt')

        self.name2label = {}
        sorted(os.listdir(os.path.join(self.gt_root)))
        for name in sorted(os.listdir(self.inp_root)):
            self.name2label[name] = len(self.name2label.keys())
        print(self.name2label)

        # image, label
        self.images_inp, self.images_gt, self.labels = self.load_csv("images.csv")

    def load_csv(self, filename):
        """
        :param filename:
        :return:
        """
        if not os.path.exists(os.path.join(self.root, filename)):
            images_inp = []

            for name in self.name2label.keys():
                images_inp += glob.glob(os.path.join(self.inp_root, name, "*.png"))
                images_inp += glob.glob(os.path.join(self.inp_root, name, "*.jpg"))
                images_inp += glob.glob(os.path.join(self.inp_root, name, "*.jpeg"))

            # 将元素打乱
            random.shuffle(images_inp)
            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img_inp in images_inp:
                    name = img_inp.split(os.sep)[-1]
                    kind = img_inp.split(os.sep)[-2]
                    img_gt = os.path.join(self.gt_root, kind, name)
                    writer.writerow([img_inp, img_gt, kind])
                print("writen into csv file: ", filename)

        # 如果已经存在了csv文件，则读取csv文件
        images_inp, images_gt, labels = [], [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img_inp, img_gt, label = row
                images_gt.append(img_gt)
                images_inp.append(img_inp)
                labels.append(label)
        assert len(images_gt) == len(labels) == len(images_inp)

        return images_inp, images_gt, labels

    def resize_img(self, input_img, gt_img, crop_width, crop_height):

        width, height = input_img.size

        if width < crop_width and height < crop_height:
            input_img = input_img.resize((crop_width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width:
            input_img = input_img.resize((crop_width, height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, height), Image.ANTIALIAS)
        elif height < crop_height:
            input_img = input_img.resize((width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        return input_crop_img, gt_crop_img

    def __len__(self):
        return len(self.images_inp)

    def __getitem__(self, idx):
        crop_width, crop_height = self.crop_size

        img_inp_path, img_gt_path, label = self.images_inp[idx], self.images_gt[idx], self.labels[idx]

        img_inp, img_gt = Image.open(img_inp_path).convert("RGB"), Image.open(img_gt_path).convert("RGB")

        input_crop_img, gt_crop_img = self.resize_img(img_inp, img_gt, crop_width, crop_height)

        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])

        input_im = transform_input(input_crop_img)
        gt_im = transform_gt(gt_crop_img)

        if list(input_im.shape)[0] is not 3 or list(gt_im.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(img_gt_path.split[os.sep][-1]))

        return input_im, gt_im, label

# datasrt = TrainData(r'D:\DEALL\deall_val\tsne_deall', (256,256))
# dataldr = DataLoader(datasrt, batch_size=16, shuffle=True)
# batch = next(iter(dataldr))
# input_im, gt_im, label = batch
# print(input_im.shape)
# print(gt_im.shape)
# print(label)

