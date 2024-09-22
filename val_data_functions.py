import torch.utils.data as data
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os, glob
import random, csv
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np

# --- Validation/test dataset --- #
class ValData(Dataset):

    def __init__(self, root):

        super(ValData, self).__init__()
        self.root = root
        self.inp_root, self.gt_root = os.path.join(root, 'input'), os.path.join(root, 'gt')
        print(self.root)
        self.images_inp, self.images_gt = self.load_csv("images.csv")

    def load_csv(self, filename):
        """
        :param filename:
        :return:
        """
        if not os.path.exists(os.path.join(self.root, filename)):
            images_inp = []


            images_inp += glob.glob(os.path.join(self.inp_root, "*.png"))
            images_inp += glob.glob(os.path.join(self.inp_root, "*.jpg"))
            images_inp += glob.glob(os.path.join(self.inp_root, "*.jpeg"))

            random.shuffle(images_inp)
            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img_inp in images_inp:
                    name = img_inp.split(os.sep)[-1]
                    img_gt = os.path.join(self.gt_root, name)
                    writer.writerow([img_inp, img_gt])
                print("writen into csv file: ", filename)

        images_inp, images_gt = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img_inp, img_gt = row
                images_gt.append(img_gt)
                images_inp.append(img_inp)

        assert len(images_gt) == len(images_inp)

        return images_inp, images_gt

    def resize_img(self, input_img, gt_img):

        input_img = input_img.resize((256, 256), Image.ANTIALIAS)
        gt_img = gt_img.resize((256, 256), Image.ANTIALIAS)

        return input_img, gt_img

    def __len__(self):
        return len(self.images_inp)

    def __getitem__(self, idx):

        img_inp_path, img_gt_path = self.images_inp[idx], self.images_gt[idx]

        img_inp, img_gt = Image.open(img_inp_path).convert("RGB"), Image.open(img_gt_path).convert("RGB")

        input_img, gt_img = self.resize_img(img_inp, img_gt)

        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])

        input_im = transform_input(input_img)
        gt_im = transform_gt(gt_img)


        if list(input_im.shape)[0] is not 3 or list(gt_im.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(img_gt_path.split[os.sep][-1]))

        return input_im, gt_im, img_gt_path


# dataset = ValData(r'DEALL\deall_val\deall_val\derain')
# dataldr = DataLoader(dataset, batch_size=16, shuffle=True)
# batch = next(iter(dataldr))
# input_im, gt_im = batch
# print(input_im.shape)
# print(gt_im.shape)
