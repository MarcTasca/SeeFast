import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random
import matplotlib.pyplot as plt

class CamVid(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, csv_path, scale, loss='dice', mode='train', normalize=True):
        super().__init__()
        self.mode = mode
        self.normalize = normalize
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.png')))
        self.image_list.sort()
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()
        # self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        # self.label_list = [os.path.join(label_path, x + '_L.png') for x in self.image_list]
        self.label_info = get_label_info(csv_path)
        # resize
        # self.resize_label = transforms.Resize(scale, Image.NEAREST)
        # self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        # self.crop = transforms.RandomCrop(scale, pad_if_needed=True)
        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])

        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        # =====================================
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        # =====================================
        img = np.array(img)

        # load label
        label = Image.open(self.label_list[index])

        # randomly resize label and random crop
        # =====================================
        if self.mode == 'train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        label = np.array(label)

        # image -> [C, H, W]
        img = Image.fromarray(img)

        # they will be normalized after being preprocessed with FDA
        if self.normalize == True:
            img = self.to_tensor(img).float()
        else: 
            img = transforms.ToTensor()(img)

        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label).long()
            return img, label

    def __len__(self):
        return len(self.image_list)

def saveimlab(img, label):
    #csv_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/CamVid/class_dict.csv'
    #label = colour_code_segmentation(np.array(label),get_label_info(csv_path))
    #label = Image.fromarray(label, 'RGB')
    #label.save('/content/images/label.png')
    input('press enter')


if __name__ == '__main__':

    csv_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/CamVid/class_dict.csv'
    train_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/CamVid/train'
    val_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/CamVid/val'
    train_labels_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/CamVid/train_labels'
    val_labels_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/CamVid/val_labels'

    data = CamVid([train_path, val_path],
                  [train_labels_path, val_labels_path], csv_path,
                  (720, 960), loss='crossentropy', mode='val')
    from model.build_BiSeNet import BiSeNet
    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

    label_info = get_label_info(csv_path)
    for i, (img, label) in enumerate(data):
        saveimlab(img,label)

