import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np
from utils import  one_hot_it, RandomCrop, reverse_one_hot, get_label_info
from utils_idda import one_hot_it_v11_idda, mapper_get_label_info
from model.build_BiSeNet import BiSeNet
from utils import colour_code_segmentation, compute_global_accuracy
from torch.utils.data import DataLoader
import random
import sys
import matplotlib.pyplot as plt


# augment images with spatial transformation: Flip
# def augmentation(image, label):
#    if random.random() > 0.5:
#        image = np.fliplr(image)
#        label = np.fliplr(label)
#    return image, label

# augment images with pixel intensity transformation: GaussianBlur
# def augmentation_pixel(image):
#    if random.random() > 0.5:
#        imagePIL = Image.fromarray(image)
#        imageBlur = imagePIL.filter(ImageFilter.GaussianBlur(radius = 2))
#        #returned blurred image
#        image = np.array(imageBlur)
#    return image

class IDDA(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, json_path, scale, loss='crossentropy', mode='train'):
        super().__init__()
        self.mode = mode

        # build a list of image paths
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.jpg')))
        self.image_list.sort()

        # build a list of label paths
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()
        self.label_info = mapper_get_label_info(json_path)

        # resize -> (720, 1280)
        scale = (720, 1280)
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)

        # params for normalization
        # mean, std -> computed with 'compute_mean_std.py'

        mean = (0.5014, 0.4765, 0.4517)
        std = (0.2484, 0.2489, 0.2495)

        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        self.image_size = (720, 960)
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):

        # load image
        seed = random.random()
        seed1 = random.random()
        img = Image.open(self.image_list[index])

        # resize and crop img
        img = self.resize_img(img)
        img = RandomCrop(self.image_size, seed1, pad_if_needed=True)(img)

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

        # resize and crop label
        label = self.resize_label(label)
        label = RandomCrop(self.image_size, seed1, pad_if_needed=True)(label)

        # randomly resize label and random crop
        # =====================================
        if self.mode == 'train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        label = np.array(label)

        # augment image and label (flip it)
        # if self.mode == 'train':
            # set a probability of 0.5
            # img, label = augmentation(img, label)

        # augment pixel image (blur it)
        # if self.mode == 'train':
            # set a probability of 0.5
            # img = augmentation_pixel(img)

        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        if self.loss == 'crossentropy':
            label = one_hot_it_v11_idda(label, self.label_info).astype(np.uint8)
            label = torch.from_numpy(label).long()
            return img, label

    def __len__(self):
        return len(self.image_list)

def saveimlab(img, label):
    # debug
    img = transforms.ToPILImage()(img)
    img.save('/content/images/imageidda.png')
    # end
    input('press enter')

if __name__ == '__main__':
    
    json_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/IDDA/classes_info.json'
    train_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/IDDA/rgb'
    labels_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/IDDA/labels'
    data = IDDA(train_path, labels_path, json_path,
                (1080, 1920), loss='crossentropy', mode='val')

    dataloader_train = DataLoader(
        data,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        )
    
    # do somethings


        
        
