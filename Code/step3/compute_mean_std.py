import torch
import glob
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

# compute mean and std over the entire dataset of images, 
# do it without normalize the images, and then apply the values in line 80
# run it just once to retrieve the infos
# quite eavy
def compute_mean_std(loader):

    # mean over entire dataset, pixel*pixel
    mean = 0.0
    for images in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    # standard deviation over entire dataset, pixel*pixel
    var = 0.0
    for images in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*1080*1920))
    return mean, std

# create a semplified dataset and dataloader for IDDA
class IDDA(torch.utils.data.Dataset):
    def __init__(self, image_path):
        super().__init__()

        # build a list of image paths
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.jpg')))
        self.image_list.sort()

        # convert to tensor
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # get image
        img = Image.open(self.image_list[index])
        img = np.array(img)
        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()
        # return image modified
        return img

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    train_path = '/content/drive/MyDrive/progetto mldl/BiseNetv1-master/data/IDDA/rgb'
    data = IDDA(train_path)

    dataloader_train = DataLoader(
        data,
        batch_size=40,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        )

    mean, std = compute_mean_std(dataloader_train)
    print(mean, std)

    # the complete execution of this code takes roughly 9 minutes