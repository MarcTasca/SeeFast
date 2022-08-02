import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import random
import numbers
import torchvision

# this is just to check if the mapping of labels values from idda to camvid notation is correct
# this is a very eavy and basic function
def check(labelbefore,labelafter, label_info):
  labelbefore = labelbefore[:,:,0]
  for i in range(labelbefore.shape[0]):
    for j in range(labelbefore.shape[1]):
      if labelafter[i,j] != label_info[labelbefore[i,j]]:
        return False 
  return True

# this returns a numpy array which maps from idda notation to camvid notation
# we are considering 10 common classes, and 11th class for void or background, like in camvid mapping
def mapper_get_label_info(json_path):
	# common classes
	N = 10
	# create df
	df = pd.read_json(json_path)
	df['camvid'] = [x[1] for x in df['label2camvid']]
	df.drop(columns=['classes','label2camvid','label','palette'], inplace=True)
	mask = df['camvid'] == 255
	df['camvid'][mask] = N+1
	return df.to_numpy()

# return the label using camvid notation
def one_hot_it_v11_idda(label, label_info):
  map = np.zeros(label.shape[:-1])
  map = label_info[label[:,:,0]]
  return map[:,:,0]