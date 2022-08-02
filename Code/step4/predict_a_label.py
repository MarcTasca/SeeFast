from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from utils import colour_code_segmentation, one_hot_it_v11, get_label_info, reverse_one_hot
from dataset.CamVid import CamVid
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from model.build_BiSeNet import BiSeNet
import torch
import numpy as np

def prediction():
    # path to modify
    data = './data/CamVid'
    pretrained_model_path = '/a_safe_place/my_wonderful_model.pth'
    csv_path = os.path.join(data, 'class_dict.csv')

    # dataloader
    dataset = CamVid(image_path = os.path.join(data, 'test'),
                    label_path = os.path.join(data, 'test_labels'),
                    csv_path = csv_path,
                    scale=(720, 960), loss='crossentropy', mode='test', normalize=False)

    dataloader = DataLoader(dataset, batch_size=1)

    #model
    num_classes = 12
    context_path = 'resnet101'
    cuda = '0'
    # build model network
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    model = BiSeNet(num_classes, context_path)
    model = torch.nn.DataParallel(model).cuda()

    # load pretrained model
    path = pretrained_model_path    
    print('load model from %s ...' % path)
    state = torch.load(path)
    model.module.load_state_dict(state['model_state'])

    # get label infos
    label_info = get_label_info(csv_path)

    # retrieve image and label
    dataloader_iter = enumerate(dataloader)
    _, batch = next(dataloader_iter)
    image_original, label = batch
    # normalize to feed the model
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = normalize(image_original)

    # feed the model with the image and store the label
    model.eval()
    predict = model(image).squeeze()
    predict = reverse_one_hot(predict)
    predict = np.array(predict.cpu())

    # paint the labels with the CamVid palette
    label_rgb = colour_code_segmentation(np.array(label)[0,:,:], label_info)
    predict_rgb = colour_code_segmentation(predict, label_info)

    # convert and save labels
    label_pil = Image.fromarray(label_rgb.astype(np.uint8))
    predict_pil = Image.fromarray(predict_rgb.astype(np.uint8))

    label_pil.save('/somewhere/groundtruth.png')
    predict_pil.save('/over_the_rainbow/predicted.png')

if __name__ == '__main__':
    prediction()